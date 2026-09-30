"""
Estimator for true candidate win rate correcting for biased judge.

Strategy: Empirical Bayes shrinkage combining labeled true rates with
judge rates, weighted by confidence in the labeled sample size.
"""

from collections import defaultdict
import math


def estimate(env):
    """Main estimator function called on each arena."""

    # Organize by stratum
    strata = defaultdict(list)
    for row in env.rows:
        key = f"{row['topic']}|{row['length']}"
        strata[key].append(row)

    stratum_keys = sorted(strata.keys())
    budget = env.label_budget

    # Allocate labels proportionally to stratum size
    allocation = {}
    total_size = sum(len(strata[k]) for k in stratum_keys)

    for k in stratum_keys:
        base = max(15, int(budget * len(strata[k]) / total_size))
        allocation[k] = base

    # Scale to fit budget
    total_alloc = sum(allocation.values())
    if total_alloc > budget:
        scale = budget / total_alloc
        for k in stratum_keys:
            allocation[k] = max(10, int(allocation[k] * scale))

    # Label selection
    labels = {}
    for k in stratum_keys:
        target = min(allocation[k], budget - len(labels), len(strata[k]))
        if target > 0:
            ids = [p['id'] for p in strata[k][:target]]
            try:
                result = env.label(ids)
                for id_str, v in result.items():
                    labels[int(id_str)] = v
            except:
                break

    # Compute shrunken estimates for each stratum
    estimates = {}
    mix = env.production_mix

    for k in stratum_keys:
        prompts = strata[k]
        labeled = [p for p in prompts if p['id'] in labels]

        if labeled:
            # True rate from labels
            true_wins = sum(1 for p in labeled if labels[p['id']])
            true_rate = true_wins / len(labeled)

            # Judge rate on full stratum
            judge_wins = sum(1 for p in prompts if p['judge_win'])
            judge_rate = judge_wins / len(prompts) if len(prompts) > 0 else 0.5

            # Shrinkage: combine true rate with judge rate
            # Weight = n / (n + λ) gives more weight to true rate with more labels
            n = len(labeled)
            lambda_param = 20  # Regularization parameter
            w = n / (n + lambda_param)

            shrunk_rate = w * true_rate + (1 - w) * judge_rate

            # Variance of the shrunken estimate
            var_true = true_rate * (1 - true_rate) / n
            var_shrunk = (w ** 2) * var_true

            estimates[k] = {
                'rate': shrunk_rate,
                'var': var_shrunk,
            }
        else:
            # No labels: use judge rate with uncertainty
            judge_wins = sum(1 for p in prompts if p['judge_win'])
            judge_rate = judge_wins / len(prompts) if len(prompts) > 0 else 0.5

            estimates[k] = {
                'rate': judge_rate,
                'var': 0.25 / max(1, len(prompts)),
            }

    # Compute weighted estimate
    point = sum(mix.get(k, 0) * estimates[k]['rate'] for k in stratum_keys)

    # Variance of weighted estimate
    total_var = sum(
        (mix.get(k, 0) ** 2) * estimates[k]['var']
        for k in stratum_keys
    )

    # Confidence interval
    se = math.sqrt(total_var) if total_var > 0 else 0.025
    z = 1.645
    margin = z * se

    lo = max(0.0, point - margin)
    hi = min(1.0, point + margin)

    # Enforce minimum width
    min_w = 0.015
    if hi - lo < min_w:
        c = point
        half_w = min_w / 2
        lo = max(0.0, c - half_w)
        hi = min(1.0, c + half_w)

    env.submit(point, lo, hi)
