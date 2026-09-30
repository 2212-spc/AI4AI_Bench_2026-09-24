"""Online, change-point-aware batch-size controller.

The measured step curve is a floor plus an inverse-batch term.  Its
inverse-batch coefficient is inferred anew after each corpus transition;
no transition location or phase-two trajectory is taken from the replica.
"""
import math


def run(env):
    n = env.n_segments
    hardware_ratio = env.t_overhead * env.throughput
    # Phase one's curve is shared between jobs.  The floor is also calibrated
    # with the current job's observations before the change point.
    floor = 0.0253
    initial_critical = 34.2
    floor_samples = []
    changed = False
    phase_two = []
    first_possible = int(0.4 * n)
    last_possible = int(0.8 * n)

    for i in range(n):
        if not changed:
            critical = initial_critical
            if i >= first_possible:
                # Minimize expected *time*, not expected optimal batch.  An
                # unobserved change can happen on this very segment.  Its
                # conditional hazard rises as the switch window expires.
                remaining = max(1, last_possible - i + 1)
                mean_jump = 24.0 / floor
                critical += mean_jump / remaining
        else:
            recent = phase_two[-8:]
            if len(recent) == 1:
                critical = recent[-1][1] * 1.005
            else:
                # Regularized local linear prediction follows each run's
                # own post-change growth without amplifying step noise.
                count = len(recent)
                xbar = sum(x for x, y in recent) / count
                ybar = sum(y for x, y in recent) / count
                sxx = sum((x - xbar) ** 2 for x, y in recent)
                sxy = sum((x - xbar) * (y - ybar) for x, y in recent)
                slope_prior = 0.004 * ybar
                slope = (sxy + 24.0 * slope_prior) / (sxx + 24.0)
                slope = max(-0.015 * ybar, min(0.05 * ybar, slope))
                critical = ybar + slope * (i - xbar)
                critical = max(0.5 * ybar, min(1.5 * ybar, critical))

        batch = int(round(math.sqrt(max(0.0, hardware_ratio * critical))))
        batch = max(env.b_min, min(env.b_max, batch))
        result = env.run_segment(batch)
        steps = float(result['steps'])
        observed = max(0.1, batch * (steps / floor - 1.0))

        if not changed:
            # A phase change is a many-sigma residual, even at the smallest
            # supported phase-two severity.  Detection is based on this run.
            expected = floor * (1.0 + initial_critical / batch)
            if i >= first_possible and (steps > 1.30 * expected or
                                         observed > 3.0 * initial_critical):
                changed = True
                phase_two.append((i, observed))
            else:
                estimate = steps / (1.0 + initial_critical / batch)
                # Reject pathological observations and retain a modest prior.
                estimate = max(0.0125, min(0.05, estimate))
                floor_samples.append(estimate)
                floor = (4.0 * 0.0253 + sum(floor_samples)) / (4 + len(floor_samples))
        else:
            phase_two.append((i, observed))
