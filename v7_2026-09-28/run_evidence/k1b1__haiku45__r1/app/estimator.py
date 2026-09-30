# Estimate true win rate by correcting for judge bias.
# Strategy:
# 1. Stratify by (topic, length) pairs
# 2. Allocate labels with minimum per stratum for coverage, then proportionally to size
# 3. Estimate true win rate per stratum from labeled samples
# 4. Combine via production mix
# 5. Compute 90% CI from sampling variance


def estimate(env):
    rows = env.rows
    production_mix = env.production_mix
    label_budget = env.label_budget

    # Group rows by stratum
    strata = {}
    for row in rows:
        stratum = f"{row['topic']}|{row['length']}"
        if stratum not in strata:
            strata[stratum] = []
        strata[stratum].append(row)

    n_strata = len(strata)

    # Allocate labels: ensure good coverage of all strata
    stratum_sizes = {s: len(rows_list) for s, rows_list in strata.items()}

    # Allocate most of budget with minimum per stratum for tight estimates
    min_per_stratum = 27  # 8 strata * 27 = 216, very tight allocation
    base_allocation = min_per_stratum * n_strata

    # Use essentially all of the budget for variance reduction
    labels_to_allocate = min(label_budget - 1, 219)

    all_to_label = []

    if base_allocation <= labels_to_allocate:
        # Baseline coverage everywhere, then proportional allocation
        remaining_budget = labels_to_allocate - base_allocation
        total_size = sum(stratum_sizes.values())

        for stratum in sorted(strata.keys()):
            rows_list = strata[stratum]
            n_base = min_per_stratum

            # Allocate share of remaining budget proportionally to size
            if total_size > 0:
                size_weight = stratum_sizes[stratum] / total_size
                n_extra = int(size_weight * remaining_budget)
            else:
                n_extra = 0

            n_labels = n_base + n_extra
            n_labels = min(n_labels, len(rows_list))

            sample_ids = [r['id'] for r in rows_list[:n_labels]]
            all_to_label.extend(sample_ids)
    else:
        # Fallback: allocate all available
        total_size = sum(stratum_sizes.values())

        for stratum in sorted(strata.keys()):
            rows_list = strata[stratum]
            if total_size > 0:
                weight = stratum_sizes[stratum] / total_size
                n_labels = max(2, int(weight * labels_to_allocate))
            else:
                n_labels = 2

            n_labels = min(n_labels, len(rows_list))
            sample_ids = [r['id'] for r in rows_list[:n_labels]]
            all_to_label.extend(sample_ids)

    # Get human labels
    labels_dict = {}
    if all_to_label:
        labels_dict = env.label(all_to_label)

    # Estimate true win rate for each stratum
    point_estimate = 0.0
    total_variance = 0.0

    for stratum, rows_list in strata.items():
        weight = production_mix.get(stratum, 0.0)

        # Get labeled rows in this stratum
        labeled_in_stratum = [r for r in rows_list
                             if str(r['id']) in labels_dict]

        if len(labeled_in_stratum) > 0:
            # Estimate from labeled sample
            true_wins = sum(1 for r in labeled_in_stratum
                           if labels_dict[str(r['id'])])
            p_hat = float(true_wins) / len(labeled_in_stratum)
            n_labeled = len(labeled_in_stratum)
        else:
            # Fallback to judge (should be rare with min per stratum)
            judge_wins = sum(1 for r in rows_list if r['judge_win'])
            p_hat = float(judge_wins) / len(rows_list) if rows_list else 0.5
            n_labeled = len(rows_list)

        # Add to weighted estimate
        point_estimate += weight * p_hat

        # Variance of estimate for this stratum
        if n_labeled > 0:
            var_stratum = p_hat * (1.0 - p_hat) / n_labeled
            total_variance += (weight ** 2) * var_stratum

    # 90% confidence interval
    se = total_variance ** 0.5
    z_critical = 1.645
    margin = z_critical * se

    lo = max(0.0, point_estimate - margin)
    hi = min(1.0, point_estimate + margin)

    env.submit(point_estimate, lo, hi)
