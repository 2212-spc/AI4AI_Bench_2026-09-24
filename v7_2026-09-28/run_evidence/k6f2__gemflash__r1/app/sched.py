import math


def run(env):
    """
    Adaptive batch size scheduler for solver-7b pre-training.

    Phase 1:
      Model scaling follows steps = S_inf + C_p1 / B, with C_p1 ~= 0.938 and S_inf ~= 0.026.
      Optimal batch size B* = sqrt(C * t_overhead * throughput / S_inf).
      In segments 0..15, run is guaranteed to be in Phase 1. Online calibration of S_inf
      is performed from segment observations.

    Switch Window (segments 16..32):
      Switch occurs uniformly between segments 16 and 32.
      Before switch detection, Bayesian expectation over Phase 1 vs Phase 2 sets B.
      Switch is detected when realized (steps - S_inf) * B exceeds threshold.

    Phase 2:
      Once switched, C_run is estimated and tracked online with exponential moving average,
      setting optimal batch size B* = sqrt(C_est * t_overhead * throughput / S_inf).
    """
    n_segments = env.n_segments
    t_ov = env.t_overhead
    tp = env.throughput
    b_min = env.b_min
    b_max = env.b_max

    c_p1 = 0.938
    c_p2_prior = 22.4
    s_inf_est = 0.026
    s_inf_history = []

    switched = False
    c_est = None

    for seg in range(n_segments):
        # Choose batch size
        if not switched:
            if seg < 16:
                # Guaranteed Phase 1
                c_target = c_p1
            else:
                # Prior probability that switch has occurred at or before this segment
                # Switch point is drawn uniformly from {16, ..., 32}
                rem = max(1, 33 - seg)
                p_switch = 1.0 / rem
                c_target = (1.0 - p_switch) * c_p1 + p_switch * c_p2_prior

            b_ideal = math.sqrt(c_target * t_ov * tp / max(1e-4, s_inf_est))
        else:
            # Phase 2
            b_ideal = math.sqrt(c_est * t_ov * tp / max(1e-4, s_inf_est))

        # Clamp and cast to int
        b = int(round(max(b_min, min(b_max, b_ideal))))

        # Run segment
        res = env.run_segment(b)
        steps = res["steps"]

        # Update estimates
        if not switched:
            if seg < 16:
                s_sample = steps - c_p1 / b
                s_inf_history.append(s_sample)
                s_inf_est = sum(s_inf_history) / len(s_inf_history)

            c_obs = (steps - s_inf_est) * b
            # Switch detection threshold
            if c_obs > 2.2:
                switched = True
                c_est = max(3.0, c_obs)
        else:
            c_obs = (steps - s_inf_est) * b
            c_est = 0.5 * max(3.0, c_obs) + 0.5 * c_est
