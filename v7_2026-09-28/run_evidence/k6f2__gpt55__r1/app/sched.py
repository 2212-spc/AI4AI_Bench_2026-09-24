def run(env):
    # Fitted from the dev replica profiles.  The switch location and phase-2
    # scale vary by run, so only the batch-response shape is baked in here.
    p = 0.973
    floor_steps = 0.0257

    def clamp_batch(b):
        b = int(round(b))
        if b < env.b_min:
            return env.b_min
        if b > env.b_max:
            return env.b_max
        return b

    def choose_batch(obs_steps, anchor_batch):
        # Model steps(b) = floor + a / b**p around the latest observation and
        # minimize steps(b) * (overhead + b / throughput) by direct search.
        a = max(1e-9, obs_steps - floor_steps) * (float(anchor_batch) ** p)
        lo, hi = float(env.b_min), float(env.b_max)
        for _ in range(55):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            c1 = (floor_steps + a / (m1 ** p)) * (env.t_overhead + m1 / env.throughput)
            c2 = (floor_steps + a / (m2 ** p)) * (env.t_overhead + m2 / env.throughput)
            if c1 < c2:
                hi = m2
            else:
                lo = m1
        # The fitted curve is least certain at the extremes.  Keeping the
        # adaptive setting in this practical range also limits the penalty of
        # an unusually cheap or expensive phase-2 draw.
        return clamp_batch(max(384.0, min(1800.0, (lo + hi) / 2.0)))

    batch = clamp_batch(640)
    phase1 = []
    post_switch = False
    smooth_steps = None
    smooth_batch = batch

    for i in range(env.n_segments):
        r = env.run_segment(batch)
        steps = float(r["steps"])

        if i < 16:
            phase1.append(steps)
            # Phase 1 is common across runs and quite flat; stay near its
            # profiled optimum instead of chasing segment noise.
            batch = clamp_batch(640)
            continue

        if not post_switch:
            base = phase1[-8:] if len(phase1) >= 8 else phase1
            baseline = sum(base) / len(base) if base else steps
            if steps > baseline * 1.4:
                post_switch = True
                smooth_steps = steps
                smooth_batch = batch
            else:
                # The switch may be as late as segment 32.  Until the large
                # jump appears, keep the low-cost phase-1 setting.
                batch = clamp_batch(640)
                continue
        else:
            # Convert the observation back to the current anchor batch before
            # smoothing, so batch changes do not masquerade as corpus changes.
            equivalent = floor_steps + max(0.0, steps - floor_steps) * ((batch / smooth_batch) ** p)
            smooth_steps = 0.65 * smooth_steps + 0.35 * equivalent

        next_batch = choose_batch(smooth_steps, smooth_batch)
        # Avoid large swings from the few-percent segment noise.
        if next_batch > batch * 1.35:
            next_batch = clamp_batch(batch * 1.35)
        elif next_batch < batch * 0.75:
            next_batch = clamp_batch(batch * 0.75)
        batch = next_batch
