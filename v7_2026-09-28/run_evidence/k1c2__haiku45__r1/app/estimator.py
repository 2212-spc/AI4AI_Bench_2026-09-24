import math
from collections import defaultdict


def estimate(env):
    """
    Estimate the true win rate correcting for judge bias using full data leverage.

    Key strategy: Estimate judge's bias rates from labeled sample, then apply
    those rates to correct ALL judge verdicts (leveraging full 900 prompts).
    This gives variance ~ O(1/N) instead of O(1/n) where N=900 and n=27.

    Steps:
    1. Stratify prompts by (topic, length)
    2. Allocate labels PROPORTIONAL TO PRODUCTION WEIGHT for efficient variance reduction
    3. Estimate judge's false positive and false negative rates per stratum
    4. Apply correction formula to judge verdicts and compute corrected rates
    5. Reweight by production mix
    """

    rows = env.rows
    n_total = len(rows)

    # Group by stratum
    strata = defaultdict(list)

    for row in rows:
        stratum_key = f"{row['topic']}|{row['length']}"
        strata[stratum_key].append(row)

    prod_mix = env.production_mix
    label_budget = env.label_budget

    # Allocate labels PROPORTIONAL TO PRODUCTION WEIGHT
    labels_per_stratum = {}
    total_allocated = 0

    for stratum_key in strata.keys():
        weight = prod_mix.get(stratum_key, 0.0)
        n_to_label = int(label_budget * weight + 0.5)
        n_to_label = min(n_to_label, len(strata[stratum_key]))
        if weight > 0.001:
            n_to_label = max(1, n_to_label)
        labels_per_stratum[stratum_key] = n_to_label
        total_allocated += n_to_label

    # Redistribute remaining budget to highest-weight strata
    remaining = label_budget - total_allocated
    if remaining > 0:
        sorted_strata = sorted(strata.keys(),
                             key=lambda s: prod_mix.get(s, 0),
                             reverse=True)
        for stratum_key in sorted_strata:
            if remaining <= 0:
                break
            max_available = len(strata[stratum_key]) - labels_per_stratum[stratum_key]
            if max_available > 0:
                add = min(remaining, max_available)
                labels_per_stratum[stratum_key] += add
                remaining -= add

    # Label prompts and measure bias rates per stratum
    bias_info = {}

    for stratum_key in sorted(strata.keys()):
        n_to_label = labels_per_stratum[stratum_key]
        stratum_rows = strata[stratum_key]

        if n_to_label == 0:
            bias_info[stratum_key] = {
                'fp_rate': 0.0,
                'fn_rate': 0.0,
                'n_labeled': 0,
                'true_wins': 0
            }
            continue

        # Select prompts to label - distributed throughout stratum
        ids_to_label = []
        step = max(1, len(stratum_rows) // n_to_label)

        for i in range(0, len(stratum_rows), step):
            if len(ids_to_label) < n_to_label:
                ids_to_label.append(stratum_rows[i]['id'])

        if len(ids_to_label) < n_to_label:
            for i in range(len(stratum_rows)):
                if len(ids_to_label) >= n_to_label:
                    break
                if stratum_rows[i]['id'] not in ids_to_label:
                    ids_to_label.append(stratum_rows[i]['id'])

        # Get human labels
        if ids_to_label:
            human_labels = env.label(ids_to_label)

            # Build confusion matrix by matching judge and true verdicts
            judge_wins_sample = []
            true_wins_sample = []

            for prompt_id in ids_to_label:
                for row in stratum_rows:
                    if row['id'] == int(prompt_id):
                        judge_wins_sample.append(1 if row['judge_win'] else 0)
                        true_wins_sample.append(1 if human_labels[str(prompt_id)] else 0)
                        break

            n_labeled = len(ids_to_label)
            judge_wins = sum(judge_wins_sample)
            true_wins = sum(true_wins_sample)
            judge_losses = n_labeled - judge_wins

            # Count errors
            false_positives = sum(1 for i in range(n_labeled)
                                 if judge_wins_sample[i] and not true_wins_sample[i])
            false_negatives = sum(1 for i in range(n_labeled)
                                 if not judge_wins_sample[i] and true_wins_sample[i])

            # Estimate rates with Laplace smoothing
            fp_rate = (false_positives + 0.5) / (max(1, judge_losses) + 1.0)
            fn_rate = (false_negatives + 0.5) / (max(1, judge_wins) + 1.0)

            # Clamp to avoid degenerate cases
            fp_rate = max(0.0, min(0.99, fp_rate))
            fn_rate = max(0.0, min(0.99, fn_rate))

            bias_info[stratum_key] = {
                'fp_rate': fp_rate,
                'fn_rate': fn_rate,
                'n_labeled': n_labeled,
                'true_wins': true_wins
            }

    # Now compute corrected win rates using ALL judge verdicts
    corrected_rates = {}

    for stratum_key in sorted(strata.keys()):
        stratum_rows = strata[stratum_key]
        n_stratum = len(stratum_rows)

        # Count judge wins across all prompts in stratum
        judge_wins_total = sum(1 for r in stratum_rows if r['judge_win'])
        judge_win_rate = judge_wins_total / n_stratum if n_stratum > 0 else 0.5

        fp_rate = bias_info[stratum_key]['fp_rate']
        fn_rate = bias_info[stratum_key]['fn_rate']
        n_labeled = bias_info[stratum_key]['n_labeled']

        # Apply bias correction using inverse formula:
        # p_judge = p_true * (1 - fn_rate) + (1 - p_true) * fp_rate
        # Solving: p_true = (p_judge - fp_rate) / (1 - fn_rate - fp_rate)

        denom = 1.0 - fn_rate - fp_rate

        if abs(denom) > 0.01:
            # Apply correction to full population
            true_win_rate = (judge_win_rate - fp_rate) / denom
            true_win_rate = max(0.0, min(1.0, true_win_rate))

            # Variance comes from two sources:
            # 1. Uncertainty in p_judge (binomial with N=n_stratum)
            # 2. Uncertainty in bias rates (from labeled sample)
            # Use delta method approximation

            p_judge_var = judge_win_rate * (1 - judge_win_rate) / max(1, n_stratum)

            # Variance of corrected estimate (delta method)
            # ∂p_true/∂p_judge = 1/denom
            variance = (p_judge_var) / (denom ** 2)

        else:
            # Degenerate case: fallback to empirical
            if n_labeled > 0:
                true_win_rate = bias_info[stratum_key]['true_wins'] / n_labeled
                variance = true_win_rate * (1 - true_win_rate) / max(1, n_labeled)
            else:
                true_win_rate = judge_win_rate
                variance = 0.25

        corrected_rates[stratum_key] = {
            'rate': true_win_rate,
            'variance': variance
        }

    # Compute final weighted estimate
    point_estimate = 0.0
    total_variance = 0.0

    for stratum_key in sorted(strata.keys()):
        weight = prod_mix.get(stratum_key, 0.0)
        rate_info = corrected_rates[stratum_key]

        point_estimate += weight * rate_info['rate']
        # Variance formula: Var[sum w_i X_i] = sum (w_i^2 Var[X_i])
        total_variance += (weight ** 2) * rate_info['variance']

    std_error = math.sqrt(total_variance)

    # 90% confidence interval
    z = 1.645
    lo = max(0.0, point_estimate - z * std_error)
    hi = min(1.0, point_estimate + z * std_error)

    env.submit(point_estimate, lo, hi)
