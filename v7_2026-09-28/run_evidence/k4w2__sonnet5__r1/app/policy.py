"""Test-time sampling policy for solver-7b.

Key facts learned from dev_sample exploration (see /app/docs/policy_api.md for the API):

* The verifier score is NOT monotonically related to correctness. Accuracy rises with
  score up to roughly score~0.7-0.8 (where samples are almost always correct), but then
  *falls back down* for very high scores (>~1.7-2.0) which are often confidently-wrong
  answers (a reward-hacking-like failure mode, seen mostly in geometry/combinatorics).
  A plain sum-of-raw-scores or argmax-of-score aggregator gets fooled by this. Mapping
  the raw score through an empirical calibration curve (score -> P(correct)) before
  aggregating fixes this and is the single biggest lever found.
* The 5 prompt-template variants are genuinely interchangeable in accuracy, but their
  *errors are only weakly correlated* with each other: sampling across variants finds
  the correct answer far more often than resampling one variant repeatedly (presence of
  the correct answer among samples rises from ~52% with a single variant to ~89% at
  budget 10 spread across variants). So each question should be sampled round-robin
  across all variants rather than repeating variant 0.
* Given calibrated per-sample log-odds, summing them per candidate answer and picking
  the argmax is a good aggregator, and the gap between the best and second-best summed
  log-odds is a good confidence signal for adaptive early stopping: easy questions
  resolve in ~4-6 samples, freeing budget for harder questions to use more (up to a cap).

The calibration table below was fit (equal-frequency binning + Laplace smoothing) on
~4500 labelled dev samples spanning all 4 topics.
"""

import math
from collections import defaultdict

# (score midpoint, empirical P(correct)) pairs, sorted ascending by score.
_CALIBRATION_BINS = [
    (-1.1516, 0.0031),
    (-0.8746, 0.0031),
    (-0.7185, 0.0031),
    (-0.5918, 0.0031),
    (-0.4578, 0.0031),
    (-0.2975, 0.0061),
    (0.0359, 0.2263),
    (0.6036, 0.9541),
    (0.8415, 0.9908),
    (1.0148, 0.9572),
    (1.1946, 0.9021),
    (1.4036, 0.6820),
    (1.6693, 0.3089),
    (2.1038, 0.0520),
]

_GAP_THRESH = 5.0
_MIN_FRAC = 0.667   # min samples per question, as a fraction of the average budget/question
_MAX_FRAC = 3.0      # per-question sample cap, as a fraction of the average budget/question


def _calibrate(score):
    bins = _CALIBRATION_BINS
    if score <= bins[0][0]:
        return bins[0][1]
    if score >= bins[-1][0]:
        return bins[-1][1]
    for i in range(len(bins) - 1):
        x0, y0 = bins[i]
        x1, y1 = bins[i + 1]
        if x0 <= score <= x1:
            if x1 == x0:
                return y0
            t = (score - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return bins[-1][1]


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _top_gap(sums):
    ranked = sorted(sums.items(), key=lambda kv: -kv[1])
    top_a, top_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else float('-inf')
    return top_a, top_s - second_s


def run(env):
    n_questions = len(env.questions)
    avg_budget = env.budget / n_questions if n_questions else 0

    min_samples = max(2, round(avg_budget * _MIN_FRAC))
    base_max_cap = max(min_samples, round(avg_budget * _MAX_FRAC))

    n_variants = env.n_variants
    variant_order = list(range(n_variants))

    out = {}
    for i, q in enumerate(env.questions):
        qid = q["id"]
        remaining_questions = n_questions - i
        remaining_budget = env.left()

        if remaining_budget <= 0:
            out[qid] = ""
            continue

        # Guarantee every remaining question (including this one) at least
        # `min_samples`, and let this question spend whatever slack budget is left
        # beyond that floor, up to base_max_cap. This makes the per-question cap
        # self-throttling so we can never run the shared pool dry.
        floor_for_this_q = min(min_samples, remaining_budget)
        slack = remaining_budget - remaining_questions * floor_for_this_q
        max_cap = floor_for_this_q + max(0, min(base_max_cap - floor_for_this_q, slack))
        max_cap = max(1, min(max_cap, remaining_budget))
        this_min_samples = min(floor_for_this_q, max_cap)

        sums = defaultdict(float)
        cost = 0
        rv = 0
        ans = None
        while cost < max_cap and env.left() > 0:
            v = variant_order[rv % n_variants]
            rv += 1
            a, s = env.sample(qid, v)
            sums[a] += _logit(_calibrate(s))
            cost += 1
            if cost >= this_min_samples:
                top_a, gap = _top_gap(sums)
                if gap > _GAP_THRESH:
                    ans = top_a
                    break
        if ans is None:
            if sums:
                ans, _ = _top_gap(sums)
            else:
                ans = ""
        out[qid] = ans

    env.submit(out)
