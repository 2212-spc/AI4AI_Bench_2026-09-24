"""Batch-size scheduler for solver-7b (closed loop).

Model (fitted on the dev replica): steps_i(b) = S_i * (1 + B_i / b), where B_i is the
noise/critical batch scale.  Per-step time is t_o + b/thr, so the cost-optimal batch is
b* = sqrt(B_i * t_o * thr), independent of S_i.  B_i grows smoothly within a phase and jumps
~15x when the data schedule switches to the phase-2 corpus (segment 24 on the dev replica).

Control loop: use the phase-1 schedule until the step count jumps well above the phase-1
prediction, then switch to the phase-2 schedule.  At the expected switch segment we go to
the phase-2 batch pre-emptively and fall back if the observed steps show phase 1 continues.
"""
import math

SWITCH_PRIOR = 24          # first segment of the phase-2 corpus on the dev replica
N_PRIOR = 40

# phase 1: log B = c0 + c1 * x, x = i / 23 ;  log S = a0 + a1 x + a2 x^2
P1_B = (math.log(49.4), math.log(270.4 / 49.4))
P1_S = (-3.6670, 0.0183, -0.0398)
# phase 2: x = (i - switch) / 15
P2_B = (math.log(3294.5), math.log(7332.0 / 3294.5))
P2_S = (-3.6375, 0.0913, 0.1567)


def _model(phase, i, switch):
    if phase == 1:
        x = i / float(max(1, SWITCH_PRIOR - 1))
        x = min(max(x, 0.0), 1.5)
        cb, cs = P1_B, P1_S
    else:
        x = (i - switch) / 15.0
        x = min(max(x, 0.0), 1.5)
        cb, cs = P2_B, P2_S
    B = math.exp(cb[0] + cb[1] * x)
    S = math.exp(cs[0] + cs[1] * x + cs[2] * x * x)
    return S, B


def run(env):
    n = env.n_segments
    bmin, bmax = env.b_min, env.b_max
    to, thr = env.t_overhead, env.throughput
    k = to * thr

    def clamp(b):
        b = int(round(b))
        return max(int(math.ceil(bmin)), min(int(math.floor(bmax)), b))

    phase = 1
    switch = None                      # segment index where phase 2 was confirmed
    prior_switch = SWITCH_PRIOR if n == N_PRIOR else int(round(n * SWITCH_PRIOR / float(N_PRIOR)))
    tried_prior = False

    for i in range(n):
        if phase == 1 and i >= prior_switch and not tried_prior:
            # expected switch: pre-emptively use the phase-2 batch for this segment
            tried_prior = True
            S2, B2 = _model(2, i, i)
            b = clamp(math.sqrt(B2 * k))
            r = env.run_segment(b)
            steps = float(r["steps"])
            S1, B1 = _model(1, i, None)
            pred1 = S1 * (1.0 + B1 / b)
            pred2 = S2 * (1.0 + B2 / b)
            # closer (in log space) to phase 2 than phase 1 -> switch confirmed
            if abs(math.log(max(steps, 1e-9) / pred2)) < abs(math.log(max(steps, 1e-9) / pred1)):
                phase, switch = 2, i
            continue

        if phase == 1:
            S1, B1 = _model(1, i, None)
            b = clamp(math.sqrt(B1 * k))
            r = env.run_segment(b)
            steps = float(r["steps"])
            pred1 = S1 * (1.0 + B1 / b)
            if steps > 2.5 * pred1:
                # phase-2 corpus arrived earlier/later than expected: switch from next segment
                phase, switch = 2, i
        else:
            S2, B2 = _model(2, i, switch)
            b = clamp(math.sqrt(B2 * k))
            env.run_segment(b)
