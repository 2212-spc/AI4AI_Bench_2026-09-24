# Adaptive batch-size schedule for solver-7b.
#
# Model (fitted on the dev replica): steps(B) = S_MIN * (1 + B_crit / B), and the
# per-step time is t_overhead + B / throughput, so the time-optimal batch is
#     B* = sqrt(B_crit * t_overhead * throughput).
# Phase 1 (before the corpus switch) has a known, mild B_crit ramp; the switch
# raises B_crit by a run-specific factor and it then drifts, so after the switch
# B_crit is re-estimated from every realised step count and extrapolated.

import math

S_MIN = 0.0242            # steps per segment at infinite batch (property of the model)
PH1_BC0 = 36.0            # phase-1 B_crit ramp: BC0 + BC_SLOPE * segment
PH1_BC_SLOPE = 1.2
SWITCH_RATIO = 1.25       # observed/expected steps above this => corpus switch detected
FIT_WINDOW = 8            # phase-2 B_crit estimates used for the trend fit


def _clamp(env, b):
    b = int(round(b))
    return max(int(env.b_min), min(int(env.b_max), b))


def _opt_batch(env, bc):
    return math.sqrt(max(bc, 1.0) * env.t_overhead * env.throughput)


def _bc_from_obs(batch, steps):
    return batch * max(steps / S_MIN - 1.0, 0.02)


def _predict_bc(seg_idx, hist):
    """hist: list of (segment, bc_estimate) since the switch; predict bc for seg_idx."""
    pts = hist[-FIT_WINDOW:]
    n = len(pts)
    if n < 3:
        return sum(b for _, b in pts) / n
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    # extrapolate one step, but be conservative with the trend (noise is a few %)
    pred = my + 0.7 * slope * (seg_idx - mx)
    # never stray too far from the recent level
    return max(0.6 * my, min(1.6 * my, pred))


def run(env):
    n = env.n_segments
    switched = False
    hist = []
    for i in range(n):
        if not switched:
            bc = PH1_BC0 + PH1_BC_SLOPE * i
        else:
            bc = _predict_bc(i, hist)
        b = _clamp(env, _opt_batch(env, bc))
        r = env.run_segment(b)
        steps = r["steps"]
        print("%d b=%d s=%.4f bc=%.0f t=%.3f" % (i, b, steps, _bc_from_obs(b, steps), env.elapsed()))
        if not switched:
            expected = S_MIN * (1.0 + bc / b)
            if steps > SWITCH_RATIO * expected:
                switched = True
                hist.append((i, _bc_from_obs(b, steps)))
        else:
            hist.append((i, _bc_from_obs(b, steps)))
