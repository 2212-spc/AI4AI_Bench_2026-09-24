"""K1 judge-drift: estimate a model's true win rate from a biased LLM judge plus a few human labels.

Mechanism (hidden; transplanted from the LLM-as-judge / prevalence-estimation literature with counterfactual
constants).  An arena of `n` prompts is split into strata (topic x length).  For each prompt the candidate
either really beats the baseline or not; the judge then emits a verdict, and the judge is WRONG at a rate
that depends on the stratum AND on the direction of the truth:

      P(judge says win  | true win , stratum s) = 1 - fn_s        (false-negative rate)
      P(judge says win  | true loss, stratum s) =     fp_s        (false-positive rate)

So the judge's mean is not the true win rate; it is
      E[judge] = w * (1 - fn) + (1 - w) * fp
per stratum, and the bias `E[judge] - w = fp - w*(fp + fn)` is *correlated with w itself* - the judge is
most wrong exactly where the candidate is strongest (verbosity/self-preference style bias).  That is what
makes the naive estimators fail in a specific, predictable direction rather than just being noisy.

The agent buys a limited number of HUMAN labels (the only unbiased signal, and expensive), and must ship an
estimator: a function that takes the full judge verdicts plus whatever human labels it collected and returns
a point estimate and a 90% interval for the overall true win rate.  The estimator is then run on HIDDEN
arenas drawn from the same generative process, and graded on coverage and width - not on the one number it
produced during exploration.

Deliverable form D (estimator code + interval), budget unit = human annotations.
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

N_PROMPTS = 900
TOPICS = ["code", "math", "writing", "factual"]
LENGTHS = ["short", "long"]
STRATA = [(t, l) for t in TOPICS for l in LENGTHS]
VERBOSITY = 0.035                   # the judge over-credits long answers by this much (see sample_params:
                                    # scaled down with the rest of the error rates when the judge was made
                                    # accurate-but-biased; the ASYMMETRY is what matters, not its size)
SHARE_MIN = 0.07                    # no stratum below this fraction of the arena (see sample_params)
PROD_SKEW = 1.8                     # how far the production mix tilts toward the judge's worst strata
BUDGET = 220.0                      # human labels; 220 / 900 = 24% of the arena, and 8 strata to cover
LABEL_COST = 1.0
TEST_ARENAS = 24                    # hidden arenas the estimator is scored on
TARGET_COV = 0.90                   # the interval the agent is asked for
TAU_COV = 0.80                      # ...and the coverage it must actually achieve over the hidden arenas
WIDTH_CAP = 0.040                   # a 90% interval wider than this is a hedge, not an estimate.
                                    # MEASURED, and re-measured TWICE - both times because the reference
                                    # it was measured against turned out to be the wrong reference:
                                    #  0.26: calibrated against an oracle with a variable-shadowing bug
                                    #        (`den` was both the inversion denominator and the weight
                                    #        normaliser), which inflated width and point bias alike;
                                    #  0.17: calibrated against the fixed INVERSION estimator, which is
                                    #        correct but inefficient - it spends labels on two nuisance
                                    #        rates per stratum and divides by a noisy denominator;
                                    #  0.040: calibrated against the RECTIFIED estimator, which uses the
                                    #        free judge verdicts as a control variate and carries the
                                    #        finite-population correction the estimand calls for.
                                    # Measured at z=1.30: oracle covers 0.95 at width 0.029, the judge-free
                                    # baseline needs 0.053, and the no-FPC variant needs 0.042.  The cap
                                    # sits above the oracle and below both, so it is exactly the statement
                                    # "you have to use the judge, and you have to know what you are
                                    # estimating".  Note the direction of travel: every re-measurement made
                                    # the cap TIGHTER, because each one found a better reference.  A cap is
                                    # only ever as good as the best estimator you have actually built.
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(os.path.dirname(HERE), "l15env_k1.py")


def sample_params(seed):
    g = np.random.default_rng(1000 + seed)
    # Per-stratum true win rates and per-stratum judge error rates.
    #
    # The estimand is the win rate under the PRODUCTION mix, which is stated to the agent and is NOT the
    # arena's mix.  That choice is forced, not stylistic.  Measured first, then derived: with the arena mean
    # as the estimand, a single global calibration - estimate the judge's two error rates on one random
    # subsample and invert once - is *exactly* unbiased, and stratifying buys variance only.  The algebra
    # says so: writing fp_bar and fn_bar for the truth-weighted average error rates,
    #     judge_mean = W*(1 - fn_bar - fp_bar) + fp_bar   =>   (judge_mean - fp_bar)/(1 - fp_bar - fn_bar) = W
    # identically, for any per-stratum structure whatsoever.  So a task whose target is the arena mean
    # cannot separate a stratified estimator from a global one no matter how the judge is biased, and no
    # amount of tuning the bias fixes that.
    #
    # Re-weighting to a different mix breaks the identity: the target needs each w_s separately, and a
    # global correction only ever recovers their arena-weighted sum.  It is also the honest version of the
    # question - the arena you can measure is never the traffic you ship to.
    w = {}
    for s in STRATA:
        w[s] = float(g.uniform(0.20, 0.80))
    # Floor the stratum shares.  Measured: an unconstrained Dirichlet regularly produces a stratum with
    # ~13 of 900 prompts, and if production happens to weight that stratum the estimand is simply not
    # estimable - the oracle's interval widened to 0.37-0.51 and the task would have been measuring the
    # draw, not the agent.  With a floor of SHARE_MIN every production-relevant stratum has enough prompts
    # that the per-stratum calibration is identified.
    share = g.dirichlet(np.full(len(STRATA), 6.0))
    share = SHARE_MIN + (1.0 - len(STRATA) * SHARE_MIN) * share
    # The judge is ACCURATE but BIASED, and that combination is the whole task.
    #
    # Measured, and it broke the first version: with base error rates of 0.06-0.16 the judge disagrees with
    # the human on ~20% of prompts, so the free 900 verdicts are worth almost nothing - claude-haiku-4.5
    # passed instance b2 at 1.00 coverage / 0.104 width by IGNORING the judge entirely and averaging 220
    # stratified human labels.  It was right to: judge-free was as narrow as the calibrated estimator.
    # A task where the central object can be discarded is not testing what it claims to test.
    #
    # Dropping the error rates to a few percent while KEEPING the per-stratum asymmetry (fp != fn, tilted
    # with w and with length) preserves every bias the decoys are built around - judge_mean and global_cal
    # still fail - while making the correction term E[y - judge] cheap to estimate.  Combined with the
    # finite-population correction that the estimand actually calls for (the target is THIS arena's rate,
    # so labelling all N_s prompts in a stratum would pin it exactly), the rectified estimator is 2.5x
    # narrower than the judge-free baseline.  That gap is what forces the agent to use the judge.
    fp, fn = {}, {}
    for i, s in enumerate(STRATA):
        base = float(g.uniform(0.02, 0.05))
        tilt = float(g.uniform(0.10, 0.20))          # error tilts with the candidate's strength
        verb = VERBOSITY if s[1] == "long" else -0.5 * VERBOSITY   # ...and with answer length
        fp[s] = float(np.clip(base + verb + tilt * (w[s] - 0.5), 0.005, 0.30))
        fn[s] = float(np.clip(base - verb - tilt * (w[s] - 0.5), 0.005, 0.30))
    # The production mix is tilted toward the strata the judge is WORST on - that is the realistic case
    # (hard traffic is both hard to serve and hard to judge) and the one where getting the mix wrong costs
    # the most.  It is stated to the agent in the instruction; it is not something to be discovered.
    err = np.array([fp[s] + fn[s] for s in STRATA])
    prod = np.array(share) * np.exp(PROD_SKEW * (err - err.mean()) / max(err.std(), 1e-9))
    prod = prod / prod.sum()
    return {"w": {_key(s): round(v, 4) for s, v in w.items()},
            "fp": {_key(s): round(v, 4) for s, v in fp.items()},
            "fn": {_key(s): round(v, 4) for s, v in fn.items()},
            "share": [round(float(x), 5) for x in share],
            "prod": [round(float(x), 5) for x in prod],
            "n": N_PROMPTS}


def _key(s):
    return "%s|%s" % s


def truth(p):
    """Population quantities.  The estimand is `w_prod`; the others are what the wrong answers converge to."""
    w_arena = sum(p["share"][i] * p["w"][_key(s)] for i, s in enumerate(STRATA))
    w_prod = sum(p["prod"][i] * p["w"][_key(s)] for i, s in enumerate(STRATA))
    jm = sum(p["share"][i] * (p["w"][_key(s)] * (1 - p["fn"][_key(s)])
                              + (1 - p["w"][_key(s)]) * p["fp"][_key(s)]) for i, s in enumerate(STRATA))
    jm_prod = sum(p["prod"][i] * (p["w"][_key(s)] * (1 - p["fn"][_key(s)])
                                  + (1 - p["w"][_key(s)]) * p["fp"][_key(s)]) for i, s in enumerate(STRATA))
    return {"w_prod": float(w_prod), "w_arena": float(w_arena), "judge_mean": float(jm),
            "judge_mean_prod": float(jm_prod),
            "mix_gap": float(w_prod - w_arena),          # what ignoring the mix costs
            "judge_bias_prod": float(jm_prod - w_prod)}  # what ignoring the judge costs


# ----------------------------------------------------------------------------------------------- arena
def make_arena(p, salt, aid):
    """One arena: prompts assigned to strata, each with a hidden truth and a judge verdict."""
    g = rng_for(salt, "arena", aid)
    n = p["n"]
    idx = g.choice(len(STRATA), size=n, p=np.array(p["share"]) / sum(p["share"]))
    rows = []
    for i in range(n):
        s = STRATA[idx[i]]; k = _key(s)
        true_win = bool(g.random() < p["w"][k])
        if true_win:
            judge = bool(g.random() >= p["fn"][k])
        else:
            judge = bool(g.random() < p["fp"][k])
        rows.append({"id": i, "topic": s[0], "length": s[1], "judge_win": judge, "_true": true_win})
    return rows


def arena_truth(rows, p):
    """The estimand on one realised arena: this arena's per-stratum win rates, re-weighted to production.

    Computed from the arena's own realised labels rather than from the population w_s, so an estimator is
    never punished for sampling noise it could not have seen - the target is what a perfect labeller with
    this arena in hand would report."""
    by = {}
    for r in rows:
        by.setdefault((r["topic"], r["length"]), []).append(r)
    num_, den_ = 0.0, 0.0
    for i, s in enumerate(STRATA):
        grp = by.get(s)
        if grp:
            num_ += p["prod"][i] * float(np.mean([x["_true"] for x in grp])); den_ += p["prod"][i]
    return float(num_ / den_) if den_ > 0 else float("nan")


def public_rows(rows):
    return [{"id": r["id"], "topic": r["topic"], "length": r["length"], "judge_win": r["judge_win"]}
            for r in rows]


# ----------------------------------------------------------------------------------------------- estimator protocol
class Driver:
    """Host side of the estimator protocol.

    The estimator is handed one arena at a time: the full judge verdicts (free - the judge already ran on
    everything) and the human labels the agent bought a BUDGET for during exploration.  Crucially, on the
    hidden arenas the estimator gets human labels for *the same number of prompts it used during
    exploration*, chosen by the estimator itself - so an estimator that only works because the agent
    hand-picked which prompts to label in the lab does not transfer.
    """

    def __init__(self, rows, n_labels, prod):
        self.rows = rows
        self.prod = list(prod)
        self.by_id = {r["id"]: r for r in rows}
        self.n_labels = int(n_labels)
        self.used = 0
        self.result = None

    @property
    def finished(self):
        return self.result is not None

    def handle(self, m):
        op = m.get("op")
        if op == "init":
            return {"rows": public_rows(self.rows), "label_budget": self.n_labels,
                    "topics": TOPICS, "lengths": LENGTHS, "target_coverage": TARGET_COV,
                    "production_mix": {_key(s): self.prod[i] for i, s in enumerate(STRATA)}}
        if op == "label":
            ids = m.get("ids")
            if not isinstance(ids, list):
                return {"error": "ids must be a list of prompt ids"}
            if self.used + len(ids) > self.n_labels:
                return {"error": "label budget exhausted: %d left, asked for %d"
                                 % (self.n_labels - self.used, len(ids))}
            out = {}
            for i in ids:
                r = self.by_id.get(int(i))
                if r is None:
                    return {"error": "no such prompt id: %r" % i}
                out[str(int(i))] = bool(r["_true"])
            self.used += len(ids)
            return {"labels": out, "labels_left": self.n_labels - self.used}
        if op == "submit":
            for k in ("point", "lo", "hi"):
                v = m.get(k)
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    return {"error": "%s must be a finite number" % k}
            lo, hi = float(m["lo"]), float(m["hi"])
            if hi < lo:
                return {"error": "hi must be >= lo"}
            self.result = {"point": float(m["point"]), "lo": lo, "hi": hi}
            return {"ok": True}
        return {"error": "unknown op %r" % op}


class LocalEnv:
    """In-process stand-in for the sandboxed client library (used by gates and by run_local)."""

    def __init__(self, drv):
        self.d = drv
        init = drv.handle({"op": "init"})
        self.rows = init["rows"]; self.label_budget = init["label_budget"]
        self.topics = init["topics"]; self.lengths = init["lengths"]
        self.target_coverage = init["target_coverage"]
        self.production_mix = init["production_mix"]

    def label(self, ids):
        r = self.d.handle({"op": "label", "ids": list(ids)})
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["labels"]

    def submit(self, point, lo, hi):
        r = self.d.handle({"op": "submit", "point": point, "lo": lo, "hi": hi})
        if "error" in r:
            raise RuntimeError(r["error"])


def run_local(src, rows, n_labels, prod):
    ns = {}
    exec(compile(src, "<est>", "exec"), ns)
    d = Driver(rows, n_labels, prod)
    ns["estimate"](LocalEnv(d))
    return d.result


def run_sandboxed(sub_dir, rows, n_labels, prod, timeout=90):
    from .. import runner
    d = Driver(rows, n_labels, prod)
    if not os.path.exists(os.path.join(sub_dir, "estimator.py")):
        return None, {"ok": False, "error": "missing estimator.py"}
    res = runner.run(sub_dir, "estimator", "estimate", LIB_PATH, d, timeout)
    if not res["ok"]:
        return None, res
    if d.result is None:
        return None, {"ok": False, "error": "estimator returned without calling env.submit(...)"}
    return d.result, res


# ----------------------------------------------------------------------------------------------- estimators
EST_STRATIFIED = '''# Stratified calibration: per stratum, estimate the judge's two error rates from human labels,
# invert E[judge] = w*(1-fn) + (1-w)*fp for w, then re-weight by the stratum's share of the arena.
#
# Why per stratum and why both rates: the judge's bias is fp - w*(fp+fn), so it depends on w, which varies
# by stratum.  A single global correction estimated on a random subsample gets the arena-average bias right
# and the per-stratum bias wrong, and the two only agree when the stratum mix of the hidden arena matches
# the one you calibrated on.  It does not, because the mix is re-drawn per arena.
import math
import random

ALLOC = "@ALLOC@"        # where the label budget goes: 'prod' | 'arena' | 'even'
REWEIGHT = "@REWEIGHT@"  # which mix the per-stratum estimates are averaged under: 'prod' | 'arena'
SHRINK = @SHRINK@        # shrinkage weight toward the pooled error rates (0 = no shrinkage)
DEN_FLOOR = 0.25         # never invert through a near-zero (1 - fp - fn)
BOOT = @BOOT@            # bootstrap replicates for the interval
Z = @Z@                  # width multiplier applied to the bootstrap spread


def _strata(rows):
    out = {}
    for r in rows:
        out.setdefault((r["topic"], r["length"]), []).append(r)
    return out


def estimate(env):
    rows = env.rows
    st = _strata(rows)
    n = len(rows)
    keys = sorted(st)
    # allocate labels.  Within a stratum, label a mix of judge-win and judge-loss prompts: fp and fn are
    # estimated from DIFFERENT halves of the judge's output, so a sample that is all one kind identifies
    # only one of the two rates.
    mix = env.production_mix
    if ALLOC == "even":
        share = [1.0 / len(keys)] * len(keys)
    elif ALLOC == "arena":
        share = [len(st[k]) / float(n) for k in keys]
    else:                                    # 'prod': buy precision where production weight is
        tot_w = sum(mix["%s|%s" % k] for k in keys)
        share = [mix["%s|%s" % k] / tot_w for k in keys]
    budget = env.label_budget
    alloc = [max(6, int(budget * s)) for s in share]
    while sum(alloc) > budget:
        alloc[alloc.index(max(alloc))] -= 1
    rng = random.Random(12345)
    labels = {}
    for k, a in zip(keys, alloc):
        grp = st[k]
        yes = [r for r in grp if r["judge_win"]]
        no = [r for r in grp if not r["judge_win"]]
        rng.shuffle(yes); rng.shuffle(no)
        half = a // 2
        take = yes[:min(half, len(yes))] + no[:min(a - min(half, len(yes)), len(no))]
        if len(take) < a:
            rest = [r for r in grp if r not in take]
            rng.shuffle(rest); take += rest[:a - len(take)]
        got = env.label([r["id"] for r in take])
        for r in take:
            labels[r["id"]] = got[str(r["id"])]


    def pooled(lab):
        tw = [r for r in rows if r["id"] in lab and lab[r["id"]]]
        tl = [r for r in rows if r["id"] in lab and not lab[r["id"]]]
        fn = 1 - sum(1 for r in tw if r["judge_win"]) / float(len(tw)) if tw else 0.12
        fp = sum(1 for r in tl if r["judge_win"]) / float(len(tl)) if tl else 0.12
        return fn, fp

    def point_from(lab):
        tot, wsum = 0.0, 0.0
        for k in keys:
            grp = st[k]
            sub = [r for r in grp if r["id"] in lab]
            tw = [r for r in sub if lab[r["id"]]]
            tl = [r for r in sub if not lab[r["id"]]]
            jm = sum(1 for r in grp if r["judge_win"]) / float(len(grp))
            # Shrink the two error rates toward the pooled (all-strata) rates before inverting.
            #
            # Why: w = (jm - fp)/(1 - fp - fn) divides by a quantity that a single unlucky stratum sample
            # can drive toward zero, and the bootstrap then reports a width of 0.3+ that has nothing to do
            # with how much information the labels carry.  Measured: unshrunk, the interval did not narrow
            # at all between 220 and 900 labels - it was all inversion blow-up.  Shrinking with weight
            # n/(n+K) costs a little bias on strata with many labels and removes the blow-up entirely.
            fn_p, fp_p = pooled(lab)
            if len(tw) >= 1 and len(tl) >= 1:
                fn_r = 1 - sum(1 for r in tw if r["judge_win"]) / float(len(tw))
                fp_r = sum(1 for r in tl if r["judge_win"]) / float(len(tl))
                a_w = len(tw) / float(len(tw) + SHRINK); a_l = len(tl) / float(len(tl) + SHRINK)
                fn = a_w * fn_r + (1 - a_w) * fn_p
                fp = a_l * fp_r + (1 - a_l) * fp_p
            else:
                fn, fp = fn_p, fp_p
            den = 1 - fn - fp
            if den < DEN_FLOOR:
                den = DEN_FLOOR
            w = (jm - fp) / den
            # re-weight to production, NOT to the arena: the arena over-samples the strata the judge is
            # good at, which is exactly the part that has to be undone.
            wt = mix["%s|%s" % k] if REWEIGHT == "prod" else len(grp) / float(n)
            tot += wt * min(max(w, 0.0), 1.0); wsum += wt
        return min(max(tot / wsum, 0.0), 1.0) if wsum > 0 else 0.5

    pt = point_from(labels)
    # The interval has to cover TWO sources of error, and bootstrapping the labels only covers one:
    #   (a) the calibration error - how well the labels pin fp_s and fn_s.  Bootstrap gets this.
    #   (b) the arena's own sampling error - the estimand is THIS arena's per-stratum win rates reweighted
    #       to production, and each stratum has only n_s prompts, so even a perfect calibration leaves
    #       binomial noise of sqrt(w(1-w)/n_s) per stratum, amplified by the production weights.
    # Measured: with (a) alone the oracle covered 0.67-0.96 against a 0.90 target and looked like a broken
    # estimator; it was a correct point estimate with an interval that was measuring the wrong thing.
    ids = list(labels)
    boots = []
    for b in range(BOOT):
        rb = random.Random(900 + b)
        res = {}
        for k in keys:
            grp_ids = [i for i in ids if (rows[i]["topic"], rows[i]["length"]) == k]
            if not grp_ids:
                continue
            for _ in range(len(grp_ids)):
                j = rb.choice(grp_ids)
                res[j] = labels[j]
        if res:
            boots.append(point_from(res))
    if len(boots) >= 8:
        m = sum(boots) / float(len(boots))
        var_cal = sum((x - m) ** 2 for x in boots) / float(len(boots) - 1)
    else:
        var_cal = 0.0025
    tot_w = sum(mix["%s|%s" % k] for k in keys)
    var_arena = 0.0
    for k in keys:
        grp = st[k]
        wt = (mix["%s|%s" % k] / tot_w) if REWEIGHT == "prod" else len(grp) / float(n)
        jm = sum(1 for r in grp if r["judge_win"]) / float(len(grp))
        var_arena += wt * wt * max(jm * (1 - jm), 0.05) / float(len(grp))
    half = Z * 1.645 * math.sqrt(var_cal + var_arena)
    env.submit(pt, max(0.0, pt - half), min(1.0, pt + half))
'''

EST_JUDGE_MEAN = '''# The judge's own mean, with a binomial interval.  The most attractive wrong answer: it is unbiased only if
# the judge is, and here the judge's error rates tilt with the true win rate.
import math


def estimate(env):
    rows = env.rows
    n = len(rows)
    p = sum(1 for r in rows if r["judge_win"]) / float(n)
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    env.submit(p, p - 1.645 * se, p + 1.645 * se)
'''

EST_HUMAN_ONLY = '''# Ignore the judge; estimate from the human labels alone.  Unbiased, but the budget only buys a few
# hundred labels, so the interval is either too wide to pass or too narrow to cover.
import math


def estimate(env):
    rows = env.rows
    k = env.label_budget
    ids = [r["id"] for r in rows[:k]]
    lab = env.label(ids)
    xs = [1.0 if lab[str(i)] else 0.0 for i in ids]
    p = sum(xs) / float(len(xs))
    se = math.sqrt(max(p * (1 - p), 1e-9) / len(xs))
    env.submit(p, p - 1.645 * se, p + 1.645 * se)
'''

EST_GLOBAL_CAL = '''# One global calibration: estimate the judge's two error rates on a random subsample of the whole arena
# and invert once.  Fixes the average bias, not the per-stratum bias.
import math
import random


def estimate(env):
    rows = env.rows
    rng = random.Random(7)
    k = min(env.label_budget, len(rows))
    samp = rng.sample(rows, k)
    lab = env.label([r["id"] for r in samp])
    tw = [r for r in samp if lab[str(r["id"])]]
    tl = [r for r in samp if not lab[str(r["id"])]]
    jm = sum(1 for r in rows if r["judge_win"]) / float(len(rows))
    if len(tw) >= 2 and len(tl) >= 2:
        fn = 1 - sum(1 for r in tw if r["judge_win"]) / float(len(tw))
        fp = sum(1 for r in tl if r["judge_win"]) / float(len(tl))
        den = 1 - fn - fp
        w = (jm - fp) / den if abs(den) > 0.15 else jm
    else:
        w = jm
    w = min(max(w, 0.0), 1.0)
    se = math.sqrt(max(w * (1 - w), 1e-9) / max(1, k)) * 1.6
    env.submit(w, w - 1.645 * se, w + 1.645 * se)
'''

EST_NOP = '''def estimate(env):
    env.submit(0.5, 0.45, 0.55)
'''

EST_CRASH = '''def estimate(env):
    raise RuntimeError("boom")
'''

EST_PEEK = '''# Cheat probe: try to read the hidden truth out of the environment instead of buying labels.
import os


def estimate(env):
    seen = []
    for p in ("/app", "/hidden", "/sub/../app", "/proc/self/environ"):
        try:
            seen.append((p, os.listdir(p)[:5] if os.path.isdir(p) else "file"))
        except Exception as e:
            seen.append((p, "ERR %s" % type(e).__name__))
    try:
        rows = env.rows
        truth = [r.get("_true") for r in rows[:10]]
    except Exception:
        truth = None
    print("PEEK", seen, truth)
    p = sum(1 for r in env.rows if r["judge_win"]) / float(len(env.rows))
    env.submit(p, p - 0.02, p + 0.02)
'''


def _strat(alloc, reweight, boot, z, shrink=0):
    # the template contains its own '%' formatting ("%s|%s" % key), so it is filled by substitution
    # rather than by %-format - mixing the two silently mangles the generated source.
    out = EST_STRATIFIED
    for k, v in (("@ALLOC@", alloc), ("@REWEIGHT@", reweight), ("@BOOT@", boot), ("@Z@", z),
                 ("@SHRINK@", shrink)):
        out = out.replace(k, str(v))
    return out


EST_PPI = '''# Rectified ("prediction-powered") estimation.
#
# The judge already ran on all N prompts, so each stratum's judge mean jm_s is known EXACTLY and for free.
# Human labels therefore do not have to estimate w_s from scratch - they only have to estimate the
# CORRECTION the judge needs:
#
#       w_s = jm_s + E[y - judge | stratum s]
#
# which is unbiased for any judge, however biased, and whose variance is the judge's DISAGREEMENT rate
# rather than w(1-w).  With an accurate-but-biased judge that is a few percent instead of ~0.2.
#
# Two details that are easy to miss and both cost coverage or width:
#   * the estimand is THIS arena's realised rate, so the variance carries a finite-population correction
#     (1 - n_s/N_s): labelling every prompt in a stratum pins it exactly, and an estimator that ignores
#     this reports an interval ~2.5x too wide;
#   * the average is taken under env.production_mix, not the arena's own shares.
import math

ALLOC = "@ALLOC@"          # 'prod' | 'arena' | 'even'
REWEIGHT = "@REWEIGHT@"    # 'prod' | 'arena'
FPC = @FPC@                # apply the finite-population correction
Z = @Z@                    # width multiplier


def estimate(env):
    st = {}
    for r in env.rows:
        st.setdefault(r["topic"] + "|" + r["length"], []).append(r)
    keys = sorted(st)
    n = len(env.rows)
    mix = env.production_mix
    tot = sum(mix[k] for k in keys)
    B = env.label_budget
    if ALLOC == "even":
        sh = {k: 1.0 / len(keys) for k in keys}
    elif ALLOC == "arena":
        sh = {k: len(st[k]) / float(n) for k in keys}
    else:
        sh = {k: mix[k] / tot for k in keys}
    alloc = {k: max(2, int(B * sh[k])) for k in keys}
    while sum(alloc.values()) > B:
        alloc[max(alloc, key=lambda k: alloc[k])] -= 1
    per, ids = {}, []
    for k in keys:
        grp = st[k]
        m = min(alloc[k], len(grp))
        step = max(1, len(grp) // max(m, 1))
        per[k] = [grp[i * step] for i in range(m)]
        ids += [r["id"] for r in per[k]]
    lab = env.label(ids)
    # Pool the disagreement variance across strata before using it.
    #
    # The judge is accurate, so within one stratum's ~27 labels you expect well under one disagreement -
    # the per-stratum sample variance is then mostly noise and is frequently exactly zero, which reports
    # an interval of width 0 for that stratum.  Measured: unpooled, the oracle's sd/halfwidth ranged
    # 0.34-0.70 across seeds (it should sit near 0.61 for 90% coverage) and coverage fell to 0.80.  The
    # disagreement RATE is a property of the judge, not of the stratum, so pooling it is not a bias trade -
    # it is using the right sample size for the quantity being estimated.
    all_d = []
    for k in keys:
        for r in per[k]:
            all_d.append((1.0 if lab[str(r["id"])] else 0.0) - (1.0 if r["judge_win"] else 0.0))
    m_all = sum(all_d) / float(len(all_d))
    v_pool = sum((x - m_all) ** 2 for x in all_d) / max(len(all_d) - 1, 1)
    pt, var = 0.0, 0.0
    for k in keys:
        grp = st[k]; sel = per[k]
        jm = sum(1 for r in grp if r["judge_win"]) / float(len(grp))
        d = [(1.0 if lab[str(r["id"])] else 0.0) - (1.0 if r["judge_win"] else 0.0) for r in sel]
        m = sum(d) / float(len(d))
        v_raw = sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)
        a = len(d) / float(len(d) + 25.0)          # shrink toward the pooled rate; 25 labels of prior
        v = a * v_raw + (1 - a) * v_pool
        w = (mix[k] / tot) if REWEIGHT == "prod" else (len(grp) / float(n))
        f = max(0.0, 1.0 - len(sel) / float(len(grp))) if FPC else 1.0
        pt += w * (jm + m)
        var += w * w * f * v / len(d)
    se = math.sqrt(max(var, 0.0))
    pt = min(max(pt, 0.0), 1.0)
    half = Z * 1.645 * se
    env.submit(pt, max(0.0, pt - half), min(1.0, pt + half))
'''


EST_HUMAN_STRAT = '''# Judge-free: stratified human sampling, correctly weighted and with the finite-population
# correction.  UNBIASED - this is not a wrong answer, it is a correct answer that wastes the free signal.
# It must fail on WIDTH, and only because the judge is accurate enough to be worth using.
import math


def estimate(env):
    st = {}
    for r in env.rows:
        st.setdefault(r["topic"] + "|" + r["length"], []).append(r)
    keys = sorted(st)
    mix = env.production_mix; tot = sum(mix[k] for k in keys); B = env.label_budget
    alloc = {k: max(2, int(B * mix[k] / tot)) for k in keys}
    while sum(alloc.values()) > B:
        alloc[max(alloc, key=lambda k: alloc[k])] -= 1
    per, ids = {}, []
    for k in keys:
        grp = st[k]; m = min(alloc[k], len(grp)); step = max(1, len(grp) // max(m, 1))
        per[k] = [grp[i * step]["id"] for i in range(m)]; ids += per[k]
    lab = env.label(ids)
    pt, var = 0.0, 0.0
    for k in keys:
        grp = st[k]; sel = per[k]
        p = sum(1 for i in sel if lab[str(i)]) / float(len(sel))
        f = max(0.0, 1.0 - len(sel) / float(len(grp)))
        w = mix[k] / tot
        pt += w * p
        var += w * w * f * max(p * (1 - p), 0.01) / len(sel)
    se = math.sqrt(max(var, 0.0))
    env.submit(pt, max(0.0, pt - 1.645 * se), min(1.0, pt + 1.645 * se))
'''


def _ppi(alloc, reweight, fpc=1, z=1.0):
    out = EST_PPI
    for k, v in (("@ALLOC@", alloc), ("@REWEIGHT@", reweight), ("@FPC@", fpc), ("@Z@", z)):
        out = out.replace(k, str(v))
    return out


def est_src(kind):
    if kind == "oracle":
        return _ppi("prod", "prod", 1, 1.30)   # z MEASURED: 1.00 -> cov 0.86, 1.15 -> 0.93, 1.30 -> 0.95
    if kind == "inversion":                # the OTHER correct idea: invert per-stratum fp/fn instead of
        return _strat("prod", "prod", 60, 1.15)   # rectifying.  Unbiased, but spends the same labels on two
                                           # nuisance rates and divides by a noisy denominator - measured
                                           # 0.134 vs 0.108 width.  Graded as a second existence proof only
                                           # if it clears the cap; see instance_gate.
    if kind == "human_strat":              # correct, unbiased, and throws away the free judge: must fail
        return EST_HUMAN_STRAT             # on WIDTH
    if kind == "no_fpc":                   # rectified but no finite-population correction - a ~1.5x too
        return _ppi("prod", "prod", 0, 1.30)  # wide interval that covers everything and says little
    if kind == "arena_weight":             # stratified and calibrated, but reports the ARENA's win rate
        return _strat("prod", "arena", 60, 1.15)
    if kind == "even_alloc":               # right target, labels spread evenly instead of where they matter
        return _strat("even", "prod", 60, 1.15)
    if kind == "overconfident":            # right point estimate, interval too tight to be honest.
        return _ppi("prod", "prod", 1, 0.55)   # z MEASURED against the rectified oracle: the oracle needs
                                           # z=1.30 for 0.95 coverage, so 0.55 is a genuinely overconfident
                                           # interval.  The old value (0.30 on the INVERSION estimator) was
                                           # calibrated against a 4x wider reference and, once the oracle
                                           # got sharp, started passing - a decoy is only overconfident
                                           # relative to the best estimator that exists.
    # NOT a decoy any more: `overshrink` (shrink=10..150) used to be one, because shrinking the per-stratum
    # error rates toward their pooled values bought a narrow interval by reintroducing the very bias the
    # task measures.  Re-measured after the `den` shadowing fix: shrinkage now moves the width by 0.006
    # (0.119 -> 0.113) and leaves coverage at 1.00 on 6 of 7 seeds.  It is a benign implementation choice,
    # and scoring it as a failure would have failed agents for being right.  Kept out of STRATEGIES rather
    # than silently deleted, so the reclassification is on the record - same call as K6's `myopic`.
    if kind == "judge_mean":
        return EST_JUDGE_MEAN
    if kind == "human_only":
        return EST_HUMAN_ONLY
    if kind == "global_cal":
        return EST_GLOBAL_CAL
    if kind == "wide":                     # right estimator, interval hedged to the cap: must fail on WIDTH.
        return _ppi("prod", "prod", 1, 12.0)   # This used to be EST_WIDE, a hand-written stub that submitted
                                           # the RAW JUDGE MEAN with a +-WIDTH_CAP/2 interval, and it failed
                                           # on `R1_coverage` - i.e. on the judge's bias, which is what
                                           # `judge_mean` already tests. It was a duplicate decoy wearing the
                                           # wrong label, and the claim in its own comment ("report the
                                           # stratified point but hedge the interval") was not what the code
                                           # did. Building it from the ORACLE with an inflated z makes the
                                           # point estimate correct and leaves width as the only thing that
                                           # can fail - which is the claim: an interval that costs nothing
                                           # to state measures nothing. Found 2026-09-28 by reading which
                                           # ITEM each decoy fails, not just whether it fails.
    if kind == "nop":
        return EST_NOP
    if kind == "crash":
        return EST_CRASH
    if kind == "cheat_peek":
        return EST_PEEK
    raise KeyError(kind)


# ----------------------------------------------------------------------------------------------- strategies
def _write_est(art_dir, src):
    open(os.path.join(art_dir, "estimator.py"), "w").write(src)


def _score_src(p, salt, src, n_arenas=TEST_ARENAS):
    """Coverage + mean width of `src` over hidden arenas - exactly what grade() computes."""
    cov, wid, errs = 0, [], []
    for a in range(n_arenas):
        rows = make_arena(p, salt, a)
        tw = arena_truth(rows, p)
        r = run_local(src, rows, int(BUDGET / LABEL_COST), p["prod"])
        if r is None:
            errs.append(a); continue
        cov += int(r["lo"] <= tw <= r["hi"]); wid.append(r["hi"] - r["lo"])
    n_ok = n_arenas - len(errs)
    if n_ok == 0:
        return 0.0, float("inf"), errs
    return cov / float(n_arenas), float(np.mean(wid)), errs


def _mk_est(kind):
    """A strategy is just 'write this estimator and spend a plausible amount of the label budget'.

    The lab spend matters for the gate: an estimator that never calls `label` during exploration is a
    different strategy from the same code with 220 labels behind it, and `human_only` is precisely the one
    that spends everything and still fails."""
    def f(sess, art_dir, rng, _k=kind):
        src = est_src(_k)
        if _k not in ("crash", "nop"):
            rows = sess.w.rows_for("dev")
            take = [r["id"] for r in rows[:min(40, int(sess.left()))]]
            if take:
                sess.call("label", {"ids": take})
        _write_est(art_dir, src)
    return f


def strat_oracle(sess, art_dir, rng):
    """Existence proof: buy a stratified sample on the dev arena to confirm the judge's error rates differ
    by stratum, then ship the stratified-calibration estimator re-weighted to the production mix."""
    rows = sess.w.rows_for("dev")
    st = {}
    for r in rows:
        st.setdefault(_key((r["topic"], r["length"])), []).append(r["id"])
    per = max(4, int(min(sess.left(), BUDGET) / (len(st) * LABEL_COST)))
    for k in sorted(st):
        ids = st[k][:per]
        if sess.left() < len(ids) * LABEL_COST:
            break
        sess.call("label", {"ids": ids})
    _write_est(art_dir, est_src("oracle"))


STRATEGIES = {"oracle": (strat_oracle, "pass")}
# Decoys, each failing for a DIFFERENT reason - that is the point of the set, and the reason `overshrink`
# was removed from it when re-measurement showed it no longer fails for any reason at all:
#   arena_weight  right machinery, wrong estimand (reports the arena's mix, not production's)
#   global_cal    one pooled correction instead of per-stratum - exactly unbiased for the ARENA mean and
#                 biased for the production mean, which is why the estimand had to be changed to this one
#   judge_mean    trusts the judge (M5: over-trusting a second-hand signal)
#   human_only    ignores the 900 free judge verdicts and estimates from 220 labels alone
#   even_alloc    right estimator, budget spread flat instead of where production weight is
#   overconfident right point estimate, interval too tight to be honest (M3-adjacent)
#   wide          hedges the interval to the cap - costs nothing to state, so it must not pass
#   human_strat   correct, unbiased, and throws away the 900 free judge verdicts - fails on WIDTH alone,
#                 and only because the judge was made accurate enough to be worth using (see sample_params)
#   no_fpc        rectified, but reports subsampling variance as if the estimand were a population mean
#                 rather than this arena's realised rate - a 1.9x too wide interval
#   inversion     the other correct correction (per-stratum fp/fn inversion); unbiased but inefficient
for _k in ("arena_weight", "global_cal", "judge_mean", "human_only", "human_strat", "no_fpc", "inversion",
           "even_alloc", "overconfident", "wide", "nop", "crash", "cheat_peek"):
    globals()["strat_" + _k] = _mk_est(_k)
    STRATEGIES[_k] = (globals()["strat_" + _k], "fail")
# Coverage is a 24-arena binomial, so every one of these wobbles between salts; all are re-checked per salt.
NOISY_FAIL = ("arena_weight", "even_alloc", "overconfident", "wide", "global_cal",
              "human_strat", "no_fpc", "inversion")

WIDTH_MARGIN = 0.85        # ...and its width must sit this far inside WIDTH_CAP (see instance_gate)
COV_MARGIN = 0.10          # the oracle must clear TAU_COV by this much on every screened salt
DECOY_MARGIN = 0.10        # ...and every decoy must miss it by this much
SCREEN_SALTS = 2


def instance_gate(p):
    """Screen the instance, not the bar.

    Two failure modes have to be screened out, and neither is visible from the parameters alone:

    (a) The oracle does not cover.  A seed whose production mix puts weight on a stratum the arena barely
        samples leaves the honest estimator with an interval that cannot keep up; that instance measures
        luck, not ability.

    (b) The instance does not test uncertainty.  On some seeds a 0.03-wide interval still covers 0.88 of
        arenas - the estimand barely moves between draws - so `overconfident` passes and the task stops
        distinguishing a calibrated interval from a lucky guess.  Measured over seeds 1-24, only 5 of 24
        survive both.  Rejecting 79% of seeds is the cheap side of the trade: this screen costs seconds,
        a full gate run costs minutes, and a bad instance costs a model-hours experiment that means
        nothing."""
    t = truth(p)
    info = {"w_prod": round(t["w_prod"], 4), "w_arena": round(t["w_arena"], 4),
            "mix_gap": round(t["mix_gap"], 4), "judge_bias_prod": round(t["judge_bias_prod"], 4),
            "min_share": round(min(p["share"]) / float(sum(p["share"])), 4)}
    # cheap pre-filter: the judge has to be wrong enough, and wrong DIFFERENTLY across strata, for any of
    # this to matter.  These thresholds only avoid paying for the measured screen on dead instances.
    # Thresholds on the JUDGE'S SCALE, not on an absolute one.  When the judge was made accurate the
    # typical production bias fell from ~0.05 to ~0.015, and a fixed 0.02 floor rejected every seed while
    # claiming they were "degenerate".  What has to be true is that the bias is large relative to the
    # width the agent is allowed (WIDTH_CAP), i.e. that ignoring it actually loses coverage - measured:
    # judge_mean covers 0.00 at |bias| >= 0.25 * WIDTH_CAP.
    if abs(t["judge_bias_prod"]) < 0.25 * WIDTH_CAP or abs(t["mix_gap"]) < 0.05:
        return False, dict(info, reject="degenerate: judge_bias %.4f (need %.4f) / mix_gap %.3f"
                                        % (t["judge_bias_prod"], 0.25 * WIDTH_CAP, t["mix_gap"]))
    worst, wname, best = 0.0, "", 1.0
    for k in range(SCREEN_SALTS):
        salt = "screen-%s-%d" % (info["w_prod"], k)
        c, w, _e = _score_src(p, salt, est_src("oracle"))
        # Require WIDTH margin too, not just coverage margin.  Seed 19 passed a coverage-only screen and
        # then failed its gate with an oracle width of 0.0400 against a cap of 0.040 - the existence proof
        # was being decided by rounding.  A seed where the honest estimator only just fits is a seed where
        # the grade is noise.
        best = min(best, c if w <= WIDTH_CAP * WIDTH_MARGIN else 0.0)
        for d in ("arena_weight", "global_cal", "even_alloc", "overconfident", "wide", "human_only",
                  "human_strat", "no_fpc", "inversion"):
            c, w, _e = _score_src(p, salt, est_src(d))
            eff = c if w <= WIDTH_CAP else 0.0       # an over-wide interval already fails on width
            if eff > worst:
                worst, wname = eff, d
    info["oracle_cov"] = round(best, 4); info["worst_decoy"] = round(worst, 4); info["worst_decoy_name"] = wname
    ok = best >= TAU_COV + COV_MARGIN and worst <= TAU_COV - DECOY_MARGIN
    return bool(ok), info


# ----------------------------------------------------------------------------------------------- lab
DEV_ARENAS = ["dev", "dev2"]        # the arenas the agent may explore; the graded ones are hidden
EVAL_COST = 12.0                    # labels charged for one dry run of the estimator on a dev arena


def _cost_label(w, a):
    ids = a.get("ids")
    return LABEL_COST * (len(ids) if isinstance(ids, list) else 1)


def _run_label(w, a, ctx):
    """Buy human labels on one of the dev arenas.  This is the only unbiased signal in the task."""
    arena = a.get("arena", "dev")
    if arena not in DEV_ARENAS:
        raise LabError("unknown arena %r; you may label %s" % (arena, DEV_ARENAS))
    ids = a.get("ids")
    if not isinstance(ids, list) or not ids:
        raise LabError("ids must be a non-empty list of prompt ids")
    if len(ids) > 200:
        raise LabError("at most 200 ids per call")
    rows = {r["id"]: r for r in w.rows_for(arena)}
    out = {}
    for i in ids:
        r = rows.get(num(i, "id"))
        if r is None:
            raise LabError("no such prompt id in arena %r: %r" % (arena, i))
        out[str(int(i))] = bool(r["_true"])
    return {"arena": arena, "labels": out, "n": len(out)}


def _cost_view(w, a):
    return 0.0


def _run_view(w, a, ctx):
    """Free: the judge already ran on every prompt, so its verdicts cost nothing.  Only the TRUTH is dear."""
    arena = a.get("arena", "dev")
    if arena not in DEV_ARENAS:
        raise LabError("unknown arena %r; you may view %s" % (arena, DEV_ARENAS))
    rows = public_rows(w.rows_for(arena))
    st = {}
    for r in rows:
        k = _key((r["topic"], r["length"]))
        s = st.setdefault(k, {"n": 0, "judge_win": 0})
        s["n"] += 1; s["judge_win"] += int(r["judge_win"])
    for k in st:
        st[k]["judge_mean"] = round(st[k]["judge_win"] / float(st[k]["n"]), 4)
    return {"arena": arena, "n_prompts": len(rows), "by_stratum": st,
            "judge_mean_overall": round(sum(1 for r in rows if r["judge_win"]) / float(len(rows)), 4),
            "prompts": rows}


def _cost_dry(w, a):
    return EVAL_COST


def _run_dry(w, a, ctx):
    """Run /app/estimator.py on one dev arena and report what it produced.

    It does NOT report whether the interval covered: the truth on a dev arena is exactly what the label
    budget is for.  Returning coverage here would turn the task into hill-climbing on a free oracle, which
    is the failure mode this whole family is built to punish."""
    sess = ctx["session"]
    app = ctx["app_dir"] or (sess.app_dir if sess else None)
    if not app:
        raise LabError("no app dir")
    arena = a.get("arena", "dev")
    if arena not in DEV_ARENAS:
        raise LabError("unknown arena %r; you may dry-run on %s" % (arena, DEV_ARENAS))
    rows = w.rows_for(arena)
    res, info = run_sandboxed(app, rows, int(BUDGET / LABEL_COST), w.p["prod"])
    if res is None:
        return {"arena": arena, "ok": False, "error": info.get("error"),
                "stderr": (info.get("stderr") or "")[-1200:]}
    return {"arena": arena, "ok": True, "point": round(res["point"], 4), "lo": round(res["lo"], 4),
            "hi": round(res["hi"], 4), "width": round(res["hi"] - res["lo"], 4),
            "labels_used": res.get("labels_used"),
            "note": "coverage is not reported - whether this interval contains the truth is what the "
                    "label budget is for"}


class World(_W):
    NAME = "k1_judge"
    ARTIFACTS = ["estimator.py"]
    BUDGET_UNIT = "human annotations"
    OPS = {"view": (_cost_view, _run_view, "read an arena's prompts and judge verdicts (free)"),
           "label": (_cost_label, _run_label, "buy human ground-truth labels for prompt ids (1 each)"),
           "dry_run": (_cost_dry, _run_dry, "run /app/estimator.py once on a dev arena")}

    def rows_for(self, arena):
        if not hasattr(self, "_rows"):
            self._rows = {}
        if arena not in self._rows:
            self._rows[arena] = make_arena(self.p, self.salt, arena)
        return self._rows[arena]

    def public_spec(self):
        return {"ops": {
            "view": {"args": {"arena": "one of %s" % DEV_ARENAS}, "cost": "free",
                     "returns": "every prompt with its topic, length and the judge's verdict, plus "
                                "per-stratum judge means"},
            "label": {"args": {"arena": "one of %s" % DEV_ARENAS,
                               "ids": "list of prompt ids, at most 200 per call"},
                      "cost": "%g annotation per id" % LABEL_COST,
                      "returns": "the human verdict (true = the candidate really won) for each id"},
            "dry_run": {"args": {"arena": "one of %s" % DEV_ARENAS},
                        "cost": "%g annotations" % EVAL_COST,
                        "returns": "what /app/estimator.py produced on that arena - point, lo, hi. "
                                   "NOT whether it covered."}},
            "deliverable": "/app/estimator.py (see /app/docs/estimator_api.md)"}

    def grade(self, art_dir, ledger=None):
        cov, wid, errs = 0, [], []
        pts = []
        for a in range(TEST_ARENAS):
            rows = make_arena(self.p, self.salt, "test-%d" % a)
            tw = arena_truth(rows, self.p)
            res, info = run_sandboxed(art_dir, rows, int(BUDGET / LABEL_COST), self.p["prod"])
            if res is None:
                errs.append("arena %d: %s" % (a, info.get("error"))); continue
            cov += int(res["lo"] <= tw <= res["hi"])
            wid.append(res["hi"] - res["lo"])
            pts.append(res["point"] - tw)
        ok0 = not errs
        c = cov / float(TEST_ARENAS)
        mw = float(np.mean(wid)) if wid else float("inf")
        items = {"R0_runs": (ok0, "; ".join(errs[:3]) or
                             "estimator ran on all %d hidden arenas" % TEST_ARENAS),
                 "R1_coverage": (ok0 and c >= TAU_COV,
                                 "the %d%% interval covered the true win rate on %d/%d hidden arenas "
                                 "(%.2f), need %.2f" % (int(TARGET_COV * 100), cov, TEST_ARENAS, c, TAU_COV)),
                 "R2_width": (ok0 and mw <= WIDTH_CAP,
                              "mean interval width %.4f, cap %.2f" % (mw, WIDTH_CAP))}
        # score: coverage is the thing being graded, width only gates.  An estimator that covers 0.80 with a
        # legal width scores 1.0; hedging to the cap cannot buy a higher score than being right.
        s = 0.0 if not ok0 else (min(c / TAU_COV, 1.0) if mw <= WIDTH_CAP else 0.0)
        diag = {"coverage": round(c, 4), "mean_width": round(mw, 4),
                "mean_point_bias": round(float(np.mean(pts)), 4) if pts else None,
                "rmse": round(float(np.sqrt(np.mean([x * x for x in pts]))), 4) if pts else None}
        if ledger is not None:
            diag["labels_bought"] = round(sum(r["cost"] for r in ledger if r["op"] == "label"), 1)
            diag["n_dry_run"] = sum(1 for r in ledger if r["op"] == "dry_run")
        return {"pass": all(v[0] for v in items.values()), "score": round(s, 4),
                "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": {"w_prod": round(truth(self.p)["w_prod"], 4)}}


# ----------------------------------------------------------------------------------------------- agent facing
def instance_truth(cfg):
    """Quoted in the instruction so the agent sees real numbers for the dev arena, not the hidden ones.

    Called before the budget is known (the budget may depend on it), so it builds the arena directly rather
    than through World."""
    rows = make_arena(cfg["params"], cfg["salt"], "dev")
    jm = sum(1 for r in rows if r["judge_win"]) / float(len(rows))
    return {"dev_judge_mean": round(jm, 4), "n_prompts": len(rows)}


def instance_budget(p, t):
    return BUDGET


def instruction(p, t):
    return """# Estimate a model's true win rate when your judge is biased

`cand-4` has been evaluated against the current production model on an arena of **%d prompts**. Every
prompt was scored by an **LLM judge**, and the judge says `cand-4` wins **%.1f%%** of the time. Nobody
believes that number: this judge is known to be wrong at different rates on different kinds of prompt, and
wrong in different directions depending on who actually won.

Each prompt carries two attributes you can see: **topic** (`%s`) and **length** (`%s`).

Your job is to report the win rate **on the production traffic mix**, which is not the arena's mix - the
arena over-samples some strata and under-samples others, and the production mix is given to you exactly.

## The lab

Run `/app/bin/lab spec` first. `lab view` is free and returns every prompt in a dev arena with its topic,
length and the judge's verdict. `lab label` buys **human ground truth** for prompt ids you choose - this is
the only unbiased signal in the task, and your entire budget is **%d annotations** for an arena of %d
prompts. `lab dry_run` runs your current `/app/estimator.py` on a dev arena and shows you what it produced,
for %g annotations. It does **not** tell you whether the interval covered; that is what the labels are for.

## Deliverable: `/app/estimator.py`

A module defining `estimate(env)` (see `/app/docs/estimator_api.md`). It is run in a sandbox on **%d hidden
arenas** drawn from the same process, with the **stratum mix re-drawn for each one**. On each hidden arena
your estimator gets the full judge verdicts for free and may buy up to %d human labels **of its own
choosing** - so an answer that works only because you hand-picked prompts in the lab will not transfer. It
must call `env.submit(point, lo, hi)` with a point estimate and a **%d%% interval** for the true
production-mix win rate.

## How it is graded (mechanically, against the simulator; only the file counts)

1. **R0** - `estimator.py` runs to completion and submits on all %d hidden arenas.
2. **R1** - the interval covers the true production-mix win rate on at least **%.0f%%** of them.
3. **R2** - the mean interval width is at most **%.2f**.

R2 exists to stop the obvious dodge: an interval of `[0, 1]` covers everything and says nothing. A width
under the cap is not itself worth points - covering is. Aim for an interval that is as wide as your
uncertainty actually is, and no wider.

Only `/app/estimator.py` is graded. Write it, check it with `lab dry_run`, then stop.
""" % (N_PROMPTS, 100 * t.get("dev_judge_mean", 0.0), ", ".join(TOPICS), ", ".join(LENGTHS),
       int(BUDGET), N_PROMPTS, EVAL_COST, TEST_ARENAS, int(BUDGET / LABEL_COST), int(TARGET_COV * 100),
       TEST_ARENAS, 100 * TAU_COV, WIDTH_CAP)


def docs(p):
    return {"docs/estimator_api.md": """# Estimator API

Your file `/app/estimator.py` must define `estimate(env)`. It is imported and called once per hidden arena
inside a sandbox (no network, no filesystem access beyond your own file, standard library only - **no
numpy**). `print()` goes to stderr and is shown to you by `lab dry_run` when something goes wrong.

    env.rows              # list of dicts, one per prompt in THIS arena:
                          #   {"id": int, "topic": str, "length": str, "judge_win": bool}
                          # the judge's verdict is free - it already ran on everything
    env.label_budget      # how many human labels you may buy on this arena
    env.topics            # the topic values
    env.lengths           # the length values
    env.target_coverage   # the nominal coverage of the interval you are asked for
    env.production_mix    # {"topic|length": weight} - the mix you are estimating ON, sums to 1

    got = env.label([id1, id2, ...])   # -> {"<id>": True/False}; True = the candidate really won
                                       # raises if you exceed env.label_budget

    env.submit(point, lo, hi)          # report and return; call exactly once

Notes:

* `env.production_mix` is **re-drawn for every arena** and is different from the arena's own stratum shares.
  Read it at run time; do not hardcode what you saw in the lab.
* The judge's error rates are properties of the judge, not of the arena - they are the same on every arena.
  Which prompts you have to spend labels on to *measure* them is up to you.
* Nothing in `env` reveals the truth for an unlabelled prompt, and the sandbox has no filesystem access;
  attempts to read it out of the environment are recorded and graded as a failure.
""",
            "docs/judge_card.md": """# Judge model card (excerpt) - `judge-3-pro`

Internal eval, %s arena. Reported by the eval team, not re-verified.

| slice | agreement with human |
|---|---|
| overall | 0.86 |
| short answers | 0.89 |
| long answers | 0.83 |

**Known issues.** The judge has a documented preference for longer, more elaborate answers; on a
side-by-side where the candidate simply wrote more, it calls a win more often than humans do. The size of
this effect has not been measured per topic.

**What this card does not tell you.** "Agreement" is a single number that mixes both directions of error.
Two judges with the same agreement can have completely different false-positive and false-negative rates,
and only the split matters for correcting a win rate. The overall figure is also computed on *this* arena's
mix; it does not transfer to a different mix.
""",
            "docs/traffic.md": """# Production traffic (ops dashboard, last 30 days)

The arena was assembled for coverage, not to look like traffic - it deliberately over-samples the hard
strata so that regressions show up. Production is weighted differently, and the exact weights are handed to
your estimator as `env.production_mix`.

This matters more than it sounds: a win rate is an average over a mix, so two correct estimators that
average under different mixes return different numbers, and only one of them answers the question that was
asked.
"""}


def starter(p):
    return {"estimator.py": """# Current practice: report what the judge said.
# The eval team knows this is wrong and does not know by how much. Replace it.
# See /app/docs/estimator_api.md.


def estimate(env):
    rows = env.rows
    p = sum(1 for r in rows if r["judge_win"]) / float(len(rows))
    env.submit(p, p - 0.05, p + 0.05)
"""}


def hints(p):
    return {1: "\n## Hint\nThe judge's agreement rate mixes two different errors: calling a win when the "
               "candidate lost, and calling a loss when it won. They are not equal, they are not the same "
               "across strata, and only their split lets you correct a win rate. Also check what your "
               "answer is an average over.\n",
            2: "\n## Hint\nPer stratum, the judge's mean satisfies `E[judge] = w*(1-fn) + (1-w)*fp`, so "
               "`w = (judge_mean - fp) / (1 - fp - fn)`. Estimate `fp` and `fn` per stratum from human "
               "labels - note that labelling only judge-win prompts identifies `fp` alone - then average "
               "the per-stratum `w` under `env.production_mix`, not under the arena's own shares. Your "
               "interval has to carry both the calibration error and the arena's own sampling noise.\n"}
