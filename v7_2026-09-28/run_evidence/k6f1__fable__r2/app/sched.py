# Closed-loop batch-size scheduler for solver-7b.
#
# Model (fitted on the dev replica): a segment at batch b needs
#     steps = S * (1 + Bn / b)
# with S ~ 0.0255 steps-units per segment and Bn a "noise batch" that is ~44
# before the corpus switch and jumps (then drifts) after it.  Per-step time is
# t_overhead + b / throughput, so the time-optimal batch is
#     b* = sqrt(Bn * t_overhead * throughput).
# Bn is tracked online from each segment's realised step count with a small
# Kalman-style filter plus a jump detector for the corpus switch.

import math

S_STEPS = 0.0255      # min steps per segment (batch -> infinity)
BN_PRE = 44.0         # noise batch before the switch (same every run)
NOISE = 0.05          # relative noise of realised step counts
PROC_REL = 0.03       # per-segment relative drift allowed post-switch
JUMP_SIGMA = 3.5      # innovation threshold (in sigmas) to declare a switch
SW_FIRST, SW_LAST = 16, 32   # the corpus switch happens at one of these segments
BN_POST_PRIOR = (300.0, 700.0, 1100.0, 1500.0, 1900.0, 2300.0, 2700.0)   # plausible noise batches after the switch


def _seg_time(b, bn, b0):
    # segment time up to the constant S * t_overhead
    return (1.0 + bn / b) * (1.0 + b / b0)


def _hedged_batch(p, bn_pre, bn_post, b0, lo, hi):
    # batch minimising expected time when the switch hits this segment w.p. p
    best, bb = None, lo
    b = lo
    while b <= hi:
        t = (1.0 - p) * _seg_time(b, bn_pre, b0)
        for bp in bn_post:
            t += p * _seg_time(b, bp, b0) / len(bn_post)
        if best is None or t < best:
            best, bb = t, b
        b *= 1.02
    return bb


def run(env):
    n = env.n_segments
    b0 = env.t_overhead * env.throughput
    bn_hat = BN_PRE
    bn_var = (0.10 * BN_PRE) ** 2
    switched = False

    for i in range(n):
        if i > SW_LAST:
            switched = True   # the switch has certainly happened by now
        if not switched and SW_FIRST <= i <= SW_LAST:
            # the switch has not happened yet: it is equally likely at any
            # remaining segment of the window, so hedge the batch upwards
            p = 1.0 / (SW_LAST - i + 1)
            b = _hedged_batch(p, bn_hat, BN_POST_PRIOR, b0, 32.0, 2048.0)
        else:
            b = math.sqrt(max(bn_hat, 1.0) * b0)
        b = int(round(min(env.b_max, max(env.b_min, b))))
        r = env.run_segment(b)
        steps = r["steps"]
        if not (steps > 0):
            continue

        # one observation of Bn from this segment
        bn_obs = b * (steps / S_STEPS - 1.0)
        # noise on bn_obs: NOISE * S(1+Bn/b) * b / S = NOISE * (b + Bn)
        sig = NOISE * (b + max(bn_hat, 0.0)) + 1.0
        innov = bn_obs - bn_hat

        if abs(innov) > JUMP_SIGMA * sig:
            # corpus switch (or a gross change): restart the estimate here
            bn_hat = max(bn_obs, 1.0)
            bn_var = sig * sig
            switched = True
            continue

        # Kalman update; allow drift only after the switch (pre-switch Bn is fixed)
        q = (PROC_REL * bn_hat) ** 2 if switched else (0.005 * bn_hat) ** 2
        p = bn_var + q
        k = p / (p + sig * sig)
        bn_hat = max(bn_hat + k * innov, 1.0)
        bn_var = (1.0 - k) * p
