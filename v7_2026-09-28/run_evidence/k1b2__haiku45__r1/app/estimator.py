"""
Estimator that corrects for judge bias by measuring error rates per stratum.

The judge shows systematic bias depending on topic and length.
We sample from each stratum to measure the judge's error rate,
then use those estimates to correct the judge verdicts.
"""


def estimate(env):
    """
    Estimate the true win rate corrected for judge bias.

    Strategy:
    1. Group prompts by stratum (topic|length)
    2. Sample from each stratum to measure true win rates
    3. Weight by production mix to get final estimate
    4. Compute confidence interval using sampling uncertainty
    """

    # Create a lookup dict for fast ID->row access
    rows_by_id = {row['id']: row for row in env.rows}

    # Organize prompts by stratum
    strata = {}
    for row in env.rows:
        key = f"{row['topic']}|{row['length']}"
        if key not in strata:
            strata[key] = []
        strata[key].append(row)

    # Determine sampling strategy
    # We have ~220 annotations. Allocate them across strata.
    # Prioritize by production mix weight
    total_budget = env.label_budget
    num_strata = len(strata)

    # Calculate allocation weights: heavily weight by production mix
    weights = {}
    for stratum_key in strata:
        production_weight = env.production_mix.get(stratum_key, 0)
        weights[stratum_key] = production_weight if production_weight > 0 else 0.01

    total_weight = sum(weights.values())

    # Allocate budget proportionally to production mix
    samples_per_stratum = {}
    allocated = 0
    sorted_strata = sorted(strata.keys(), key=lambda k: weights[k], reverse=True)

    for i, stratum_key in enumerate(sorted_strata):
        if i == len(sorted_strata) - 1:
            # Last stratum gets remaining budget
            samples_per_stratum[stratum_key] = max(1, total_budget - allocated)
        else:
            # Allocate proportionally
            w = weights[stratum_key] / total_weight
            samples = max(1, int(total_budget * w))
            samples_per_stratum[stratum_key] = samples
            allocated += samples

    # Select prompts to label: evenly spaced within each stratum
    stratum_ids_to_label = {}
    for stratum_key in strata:
        prompts = strata[stratum_key]
        n_to_label = min(samples_per_stratum.get(stratum_key, 1), len(prompts))

        # Select evenly spaced prompts by ID order
        if n_to_label >= len(prompts):
            selected_ids = [p['id'] for p in prompts]
        else:
            step = len(prompts) / n_to_label
            selected_ids = [prompts[int(i * step)]['id'] for i in range(n_to_label)]

        stratum_ids_to_label[stratum_key] = selected_ids

    # Collect all IDs to label
    all_ids_to_label = []
    for ids in stratum_ids_to_label.values():
        all_ids_to_label.extend(ids)

    # Label them (this costs annotations)
    labels = env.label(all_ids_to_label)

    # Compute true win rate for each stratum
    stratum_true_rates = {}
    stratum_sample_sizes = {}

    for stratum_key in strata:
        ids_labeled = stratum_ids_to_label.get(stratum_key, [])
        if not ids_labeled:
            # No samples for this stratum, skip it
            stratum_true_rates[stratum_key] = 0.5  # Neutral estimate
            stratum_sample_sizes[stratum_key] = 0
            continue

        # Compute true rate from labels
        true_wins = sum(1 for id in ids_labeled if labels.get(str(id), False))
        true_rate = true_wins / len(ids_labeled)

        stratum_true_rates[stratum_key] = true_rate
        stratum_sample_sizes[stratum_key] = len(ids_labeled)

    # Compute estimate weighted by production mix
    point_estimate = 0.0
    variance = 0.0

    for stratum_key in strata:
        production_weight = env.production_mix.get(stratum_key, 0)
        if production_weight == 0:
            continue

        true_rate = stratum_true_rates[stratum_key]
        point_estimate += production_weight * true_rate

        # Compute standard error for this stratum
        n_labeled = stratum_sample_sizes[stratum_key]
        if n_labeled > 0:
            # SE^2 for this stratum = p(1-p)/n
            se_sq = (true_rate * (1 - true_rate)) / n_labeled
        else:
            # No labeled data - use maximum uncertainty (binomial with p=0.5)
            se_sq = 0.25 / max(1, len(strata[stratum_key]))

        # Add weighted contribution to overall variance
        # Var[sum w_i * X_i] = sum w_i^2 * Var[X_i]
        variance += (production_weight ** 2) * se_sq

    # Standard error of weighted estimate
    std_error = variance ** 0.5

    # 90% CI: ±1.645 * SE
    margin = 1.645 * std_error

    lo = max(0, point_estimate - margin)
    hi = min(1, point_estimate + margin)

    env.submit(point_estimate, lo, hi)
