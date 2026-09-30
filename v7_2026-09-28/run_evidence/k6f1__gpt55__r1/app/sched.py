def run(env):
    n = env.n_segments

    # Batch 128 is the best cheap probe on phase 1 in the dev replica.  Once
    # the switch window opens, 192 is a cheap hedge: close to neutral before the
    # switch, but better if phase 2 has started mildly.
    probe = 128
    hedge = 192
    candidates = [128, 192, 256, 512]

    # Relative phase-1 step counts, normalized to batch 128.  The curve is flat
    # enough near the optimum that this table is intentionally conservative.
    pre_rel = {
        128: 1.00,
        192: 0.91,
        256: 0.84,
        512: 0.71,
    }

    # The phase-2 excess is more batch-sensitive than the phase-1 floor, but
    # not enough to justify very large batches except on clearly hard runs.
    excess_rel = {
        128: 1.00,
        192: 0.75,
        256: 0.55,
        512: 0.42,
    }

    def clamp_batch(b):
        if b < env.b_min:
            return env.b_min
        if b > env.b_max:
            return env.b_max
        return b

    def step_time(b):
        return env.t_overhead + float(b) / env.throughput

    def choose_batch(base128, excess128):
        best_b = probe
        best_cost = None
        for b in candidates:
            if b == 512 and excess128 < 1.6 * base128:
                continue
            b = clamp_batch(b)
            pr = pre_rel.get(b, pre_rel[probe])
            er = excess_rel.get(b, excess_rel[probe])
            steps = base128 * pr + excess128 * er
            cost = steps * step_time(b)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_b = b
        return best_b

    baseline_samples = []
    base128 = None
    excess128 = 0.0
    in_phase2 = False
    batch = probe

    for i in range(n):
        # The switch cannot happen before segment 16, so use the known-good
        # probe there to get a clean baseline for this noisy run.
        if i < 16:
            batch = probe
        elif in_phase2:
            batch = choose_batch(base128, excess128)
        else:
            batch = hedge

        r = env.run_segment(clamp_batch(batch))
        steps = float(r["steps"])

        if i < 16:
            baseline_samples.append(steps)
            if len(baseline_samples) >= 6:
                s = sorted(baseline_samples)
                mid = len(s) // 2
                if len(s) % 2:
                    base128 = s[mid]
                else:
                    base128 = 0.5 * (s[mid - 1] + s[mid])
            else:
                base128 = sum(baseline_samples) / len(baseline_samples)
            continue

        if base128 is None:
            base128 = steps

        # Convert the observation back to an estimated batch-128 excess.  This
        # lets the controller keep adapting after it has moved off the probe.
        pr = pre_rel.get(batch, pre_rel[probe])
        er = excess_rel.get(batch, excess_rel[probe])
        obs_excess = (steps - base128 * pr) / er
        if obs_excess < 0.0:
            obs_excess = 0.0

        # A few percent of noise is normal; the phase-2 jump is materially
        # larger on runs where changing batch is worth it.  From segment 32 on,
        # the switch is guaranteed, so even a mild excess is treated as real.
        ratio = 1.0 + obs_excess / base128
        threshold = 1.22 if i < 32 else 1.05
        if in_phase2 or ratio > threshold:
            in_phase2 = True
            if excess128 == 0.0:
                excess128 = obs_excess
            else:
                # Track growth without overreacting to one noisy low segment.
                blended = 0.55 * excess128 + 0.45 * obs_excess
                floor = 0.82 * excess128
                excess128 = blended if blended > floor else floor

