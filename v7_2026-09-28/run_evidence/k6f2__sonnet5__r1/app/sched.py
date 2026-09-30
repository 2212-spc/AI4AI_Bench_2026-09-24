import math


def run(env):
    t_over = env.t_overhead
    thr = env.throughput
    b_min = int(round(env.b_min))
    b_max = int(round(env.b_max))

    def clamp(b):
        return max(b_min, min(b_max, int(round(b))))

    # Phase-1 ramp (before the corpus switch) is a fixed property of the
    # model: it is the same on every run. Calibrated offline on the dev
    # replica by profiling a few batch sizes and fitting
    #   steps(b) = Smin * (1 + Bcrit / b)
    # The optimal batch minimising steps(b) * (t_over + b/thr) is
    #   b* = sqrt(Bcrit * t_over * thr)
    Smin1 = 0.0261
    Bcrit1 = 36.08

    def bstar(Bcrit):
        return math.sqrt(max(Bcrit, 1e-6) * t_over * thr)

    b1 = clamp(bstar(Bcrit1))

    switched = False
    phase2 = []       # (batch, steps) observed after the corpus switch
    Bcrit2_est = Bcrit1 * 10.0   # placeholder prior until we have real data
    b_current = b1

    for _ in range(env.n_segments):
        r = env.run_segment(b_current)
        steps = r["steps"]

        if not switched:
            predicted = Smin1 * (1 + Bcrit1 / b_current)
            if steps > predicted * 1.20:
                # The switch to the phase-2 corpus happened in this segment.
                # Its cost/growth are per-run and unknown - start
                # characterising them from here on.
                switched = True
                phase2.append((b_current, steps))
                probe = b_current * 3
                if probe > b_max:
                    probe = max(b_min, b_current // 3)
                b_current = clamp(probe)
            # else: still phase 1, keep the same batch (b1 is optimal)
        else:
            phase2.append((b_current, steps))
            if len(phase2) >= 2:
                xs = [1.0 / b for b, _ in phase2]
                ys = [s for _, s in phase2]
                n = len(xs)
                mx = sum(xs) / n
                my = sum(ys) / n
                num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                den = sum((x - mx) ** 2 for x in xs)
                if den > 1e-12 and my > 1e-12:
                    slope = num / den
                    intercept = my - slope * mx
                    if intercept > 1e-9 and slope > 0:
                        Bcrit2_est = slope / intercept
            b_current = clamp(bstar(Bcrit2_est))
