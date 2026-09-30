def run(env):
    # The small-batch phase is close to a step-count floor.  In the later
    # corpus phase, step counts scale strongly with batch, so amortising the
    # fixed per-step overhead is worthwhile.
    low_batch = min(env.b_max, max(env.b_min, 512))
    high_batch = env.b_max
    phase_two = False

    for _ in range(env.n_segments):
        batch = high_batch if phase_two else low_batch
        result = env.run_segment(batch)

        # At the low batch the two corpus phases are well separated in step
        # count.  Use a conservative threshold so ordinary run-to-run noise
        # cannot trigger the expensive phase prematurely.
        if not phase_two and result["steps"] > 0.10:
            phase_two = True
