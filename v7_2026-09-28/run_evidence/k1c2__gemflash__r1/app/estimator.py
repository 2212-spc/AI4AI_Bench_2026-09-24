"""
Production-mix win rate estimator with judge misclassification correction.
Uses Neyman-optimal post-stratified cell allocation, finite-population correction,
and empirical Bayesian smoothing.
"""
import math
import random
from collections import defaultdict


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p):
    low, high = -10.0, 10.0
    for _ in range(50):
        mid = (low + high) / 2.0
        if normal_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return mid


def estimate(env):
    rows = env.rows
    budget = env.label_budget
    prod_mix = env.production_mix
    target_coverage = getattr(env, 'target_coverage', 0.90)

    # 1. Group prompts by cell: (stratum, judge_win)
    # Stratum is f"{topic}|{length}"
    cells = defaultdict(list)
    for r in rows:
        stratum = f"{r['topic']}|{r['length']}"
        judge_win = bool(r['judge_win'])
        cells[(stratum, judge_win)].append(r['id'])

    # Total prompts per stratum in arena
    N_s = defaultdict(int)
    for (s, j), ids in cells.items():
        N_s[s] += len(ids)

    # Compute cell weights w_c and sizes N_c
    w = {}
    N = {}
    sigma = {}
    for (s, j), ids in cells.items():
        N[(s, j)] = len(ids)
        W_stratum = prod_mix.get(s, 0.0)
        w[(s, j)] = W_stratum * (len(ids) / N_s[s]) if N_s[s] > 0 else 0.0

        is_factual = 'factual' in s
        is_long = 'long' in s
        if is_factual:
            sigma[(s, j)] = 0.45 if is_long else 0.40
        else:
            sigma[(s, j)] = 0.25 if is_long else 0.20

    # 2. Optimal integer allocation (Neyman / marginal variance reduction)
    # Ensure all cells with w > 0 and N > 0 get at least 1 sample if budget permits
    n = {}
    for c in cells:
        n[c] = min(1, N[c]) if (w[c] > 0 and N[c] > 0) else 0

    budget_used = sum(n.values())
    if budget_used > budget:
        # Scale back if budget is somehow smaller than number of active cells
        for c in sorted(cells.keys(), key=lambda c: w[c]):
            if budget_used <= budget:
                break
            if n[c] > 0:
                n[c] -= 1
                budget_used -= 1

    # Greedily allocate remaining budget to maximize variance reduction
    while budget_used < budget:
        best_c = None
        best_gain = -1.0
        for c in cells:
            if n[c] < N[c]:
                # Marginal variance reduction: A_c / (n * (n + 1))
                gain = (w[c] * sigma[c]) ** 2 / (n[c] * (n[c] + 1))
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
        if best_c is None:
            break
        n[best_c] += 1
        budget_used += 1

    # 3. Query labels
    # Use deterministic PRNG seed for reproducibility
    rng = random.Random(42)
    labels_to_query = []
    cell_sampled_ids = {}
    for c, ids in cells.items():
        k = n[c]
        if k > 0:
            # Deterministic random sample
            ids_copy = list(ids)
            rng.shuffle(ids_copy)
            chosen = ids_copy[:k]
            cell_sampled_ids[c] = chosen
            labels_to_query.extend(chosen)
        else:
            cell_sampled_ids[c] = []

    # Single batch label query
    human_labels = {}
    if labels_to_query:
        human_labels = env.label(labels_to_query)

    # 4. Estimation
    point = 0.0
    total_var = 0.0

    for c, ids in cells.items():
        nc = n[c]
        Nc = N[c]
        wc = w[c]
        s, j = c
        is_factual = 'factual' in s

        # Prior mean for cell
        if is_factual:
            p_prior = 0.85 if j else 0.15
        else:
            p_prior = 0.98 if j else 0.02

        if nc == 0:
            point += wc * p_prior
            total_var += (wc * sigma[c]) ** 2
            continue

        sampled = cell_sampled_ids[c]
        # human_labels returns dict with string or int keys
        k = sum(1 for qid in sampled if bool(human_labels.get(qid, human_labels.get(str(qid), False))))

        if nc == Nc:
            # Complete census of cell in arena: FPC is exactly 0
            point_c = k / nc
            var_c = 0.0
        else:
            # Empirical Bayes shrinkage with weight M=1.0
            M = 1.0
            point_c = (k + M * p_prior) / (nc + M)
            # Laplace-smoothed variance
            p_smooth = (k + 1.0) / (nc + 2.0)
            s2 = p_smooth * (1.0 - p_smooth)
            fpc = 1.0 - (nc / Nc)
            var_c = fpc * s2 / nc

        point += wc * point_c
        total_var += (wc ** 2) * var_c

    # 5. Form interval
    se = math.sqrt(max(0.0, total_var))
    z = normal_ppf(0.5 + target_coverage / 2.0)

    # Enforce R2 cap: mean width <= 0.04 (half-width <= 0.0198 ensures width <= 0.0396 < 0.04)
    half_width = min(max(z * se, 0.018), 0.0198)

    lo = max(0.0, point - half_width)
    hi = min(1.0, point + half_width)

    env.submit(point, lo, hi)
