"""Closed-loop, change-point-aware batch controller (standard library only)."""
import math


def run(env):
    # Replica measurements fit steps = work / batch + saturation_steps.
    # Only the phase-one ramp is shared across runs; phase-two work is
    # identified from the live run, not from the replica's change point.
    floor_prior = 0.0245
    saturation = floor_prior
    calibration = []
    changed = False
    level = 1.05
    growth = 0.20
    post = []
    first_switch = int(math.ceil(0.40 * env.n_segments))
    last_switch = int(math.floor(0.80 * env.n_segments))

    for i in range(env.n_segments):
        baseline = 1.05 + 0.014 * i
        if changed:
            work = max(baseline, level + growth)
        else:
            # The switch is unobservable until a segment has completed.
            # Minimize expected cost on that segment, rather than waiting
            # at the small phase-one optimum for a potentially large jump.
            if i < first_switch:
                hazard = 0.0
            elif i <= last_switch:
                hazard = 1.0 / (last_switch - i + 1)
            else:
                hazard = 1.0
            expected_jump_work = 32.0
            work = baseline + hazard * max(0.0, expected_jump_work - baseline)

        target = math.sqrt(max(1e-12, work * env.t_overhead *
                               env.throughput / saturation))
        batch = max(env.b_min, min(env.b_max, int(round(target))))
        result = env.run_segment(batch)
        steps = float(result['steps'])
        observed_work = max(0.01, batch * (steps - saturation))

        if not changed:
            # Allow generous noise tolerance. Even the low end of a
            # phase-two jump is readily separated from phase-one noise.
            expected_steps = saturation + baseline / batch
            if i >= first_switch and steps > expected_steps * 1.32:
                changed = True
                level = observed_work
                growth = max(0.02, 0.005 * level)
                post = [(i, observed_work, max(0.08, 0.055 * steps * batch))]
            else:
                # Calibrate the saturation floor using only confirmed
                # phase-one observations; the corpus jump must not leak
                # into this estimate.
                estimate = steps - baseline / batch
                calibration.append(estimate)
                ordered = sorted(calibration)
                trim = len(ordered) // 6
                retained = ordered[trim:len(ordered)-trim] if trim else ordered
                saturation = (6.0 * floor_prior + sum(retained)) / (6 + len(retained))
                saturation = min(0.029, max(0.020, saturation))
        else:
            post.append((i, observed_work, max(0.08, 0.055 * steps * batch)))
            post = post[-10:]
            # Weighted local linear regression: predict the next segment,
            # not the average of already completed segments. A weak slope
            # prior stabilizes the first few post-switch observations.
            slope_prior = max(0.02, 0.005 * post[0][1])
            slope_sd = max(0.20, 0.015 * post[0][1])
            sw = sx = sy = sxx = sxy = 0.0
            for index, value, sigma in post:
                x = index - i
                weight = 1.0 / (sigma * sigma)
                sw += weight
                sx += weight * x
                sy += weight * value
                sxx += weight * x * x
                sxy += weight * x * value
            precision = 1.0 / (slope_sd * slope_sd)
            denominator = sxx + precision - sx * sx / sw
            growth = (sxy + precision * slope_prior - sx * sy / sw) / denominator
            growth = max(0.0, growth)
            level = max(baseline, (sy - growth * sx) / sw)
