"""Closed-loop batch controller with a run-local change-point posterior."""
import math


def run(env):
    # Profiles identify the batch response as steps = floor + demand / batch.
    # Only the initial ramp is transferable between runs.  The posterior below
    # marginalizes over the unknown switch, its jump, and the subsequent slope.
    floor = 0.02518
    initial = 1.113
    ramp = 0.0127
    scale = 1.0
    scale_var = 0.06 ** 2
    n = env.n_segments
    lo = int(math.floor(0.4 * n))
    hi = int(math.ceil(0.8 * n))
    hi = min(hi, n - 1)

    jumps = []
    for j in range(72):
        log_j = math.log(0.15) + j * math.log(40.0 / 0.15) / 71.0
        jump = math.exp(log_j)
        weight = math.exp(-0.5 * ((log_j - math.log(4.7)) / 0.75) ** 2)
        jumps.append((jump, weight))

    slopes = [0.7 * (j / 24.0) ** 2 for j in range(25)]
    slope_weights = []
    for j, slope in enumerate(slopes):
        left = 0.0 if j == 0 else (slopes[j-1] + slope) * 0.5
        right = slope if j == 24 else (slope + slopes[j+1]) * 0.5
        # A broad tail permits unusually steep phase-two ramps.
        density = 0.8 * math.exp(-slope / 0.065) / 0.065
        density += 0.2 * math.exp(-slope / 0.23) / 0.23
        slope_weights.append((right - left) * density)

    particles = []
    for change in range(lo, hi + 1):
        for jump, jw in jumps:
            for slope, sw in zip(slopes, slope_weights):
                particles.append([change, jump, slope, jw * sw])
    normalizer = sum(p[3] for p in particles)
    for p in particles:
        p[3] /= normalizer

    multiplier = max(0.0, env.t_overhead * env.throughput / floor)
    for i in range(n):
        base = scale * (initial + ramp * i)
        mean_demand = base
        if i >= lo:
            for change, jump, slope, weight in particles:
                if i >= change:
                    mean_demand += weight * (jump + slope * (i - change))
        batch = math.sqrt(max(0.0, multiplier * mean_demand))
        batch = max(env.b_min, min(env.b_max, int(round(batch))))
        result = env.run_segment(batch)
        observed = max(float(result['steps']), 1e-15)

        if i < lo:
            nominal = initial + ramp * i
            measurement = (observed - floor) * batch / nominal
            variance = (0.075 * (floor + base / batch) * batch / nominal) ** 2
            gain = scale_var / (scale_var + variance)
            scale += gain * (measurement - scale)
            scale = max(0.75, min(1.3, scale))
            scale_var *= 1.0 - gain
        else:
            total = 0.0
            for p in particles:
                change, jump, slope, weight = p
                demand = base
                if i >= change:
                    demand += jump + slope * (i - change)
                predicted = floor + demand / batch
                residual = math.log(observed / predicted) + 0.00245
                # Small contamination component prevents one anomalous segment
                # from irrevocably selecting the wrong change point.
                likelihood = math.exp(-0.5 * (residual / 0.07) ** 2)
                likelihood += 0.003 * math.exp(-0.5 * (residual / 0.25) ** 2)
                p[3] = weight * likelihood
                total += p[3]
            if total > 0.0:
                inv = 1.0 / total
                for p in particles:
                    p[3] *= inv
