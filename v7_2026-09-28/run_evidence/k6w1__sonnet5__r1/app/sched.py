import math


def run(env):
    n = env.n_segments
    b_min, b_max = env.b_min, env.b_max
    t_overhead, throughput = env.t_overhead, env.throughput

    def clip(b):
        return max(b_min, min(b_max, b))

    spike_ratio = 1.3     # observed/predicted steps ratio that triggers an immediate re-solve
    s_min_alpha = 0.25    # EMA smoothing for the floor-step estimate
    bc_alpha = 0.4        # EMA smoothing (in log space) for the critical-batch estimate
    root_k = math.sqrt(1.3)  # probe spread around the current batch estimate

    s_min_est = None
    bc_est = None
    b_star = clip(math.sqrt(b_min * b_max))

    def predict(b):
        if s_min_est is None or bc_est is None:
            return None
        return s_min_est * (1 + bc_est / b)

    def batch_for_bc(bc):
        return clip(math.sqrt(max(bc, 1e-9) * t_overhead * throughput))

    prev = None  # (batch, steps) of the previous segment
    toggle = False

    for _ in range(n):
        b = clip(b_star * root_k) if toggle else clip(b_star / root_k)
        toggle = not toggle

        pred = predict(b)
        r = env.run_segment(b)
        steps = r["steps"]

        spiked = False
        if pred is not None and pred > 0 and steps > spike_ratio * pred and s_min_est and s_min_est > 0:
            ratio = steps / s_min_est - 1.0
            if ratio > 0:
                bc_est = b * ratio
                b_star = batch_for_bc(bc_est)
                spiked = True

        if not spiked and prev is not None and prev[0] != b and prev[1] > 0 and steps > 0:
            x1, x2 = 1.0 / prev[0], 1.0 / b
            s1, s2 = prev[1], steps
            slope = (s1 - s2) / (x1 - x2)
            intercept = s2 - slope * x2
            if intercept > 0 and slope > 0:
                bc_new = slope / intercept
                if s_min_est is None:
                    s_min_est = intercept
                    bc_est = bc_new
                else:
                    s_min_est = (1 - s_min_alpha) * s_min_est + s_min_alpha * intercept
                    bc_est = math.exp((1 - bc_alpha) * math.log(bc_est) + bc_alpha * math.log(bc_new))
                b_star = batch_for_bc(bc_est)

        prev = (b, steps)
