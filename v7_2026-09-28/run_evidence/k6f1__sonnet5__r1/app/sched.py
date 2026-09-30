import math

# Phase-1 ("ramp") step-count model, fit offline from profiling the dev
# replica at segment 0 (always before the switch): steps(B) = Smin + C/B.
# This ramp is a fixed property of the model and is the same every run, so
# the optimal constant batch for phase 1 can be computed in closed form:
#   cost(B) = steps(B) * (t_overhead + B/throughput)
# is minimized at B = sqrt(Bcrit * t_overhead * throughput), independent of
# Smin, where Bcrit = C/Smin.
_S1_MIN = 0.02606
_S1_C = 1.1203
_BCRIT1 = _S1_C / _S1_MIN

# The corpus switch is guaranteed to land within segments 16-32 (1-indexed),
# i.e. 0-indexed [15, 31]. Before that window, phase-1 behavior is exactly
# known, so there's nothing to detect or adapt yet.
_DETECT_START = 15
_DETECT_FORCE = 31

# Phase-2 segments realise wildly more steps than phase-1 would predict at
# the same batch (observed 5-10x on profiling), far above the few-percent
# run-to-run noise on step counts - so a single segment this far off the
# phase-1 curve is decisive, no need to wait for confirmation.
_DEV_STRONG = 2.0

# Cost-per-step is fairly flat near its optimum (verified while profiling),
# so once the switch is confirmed we make exactly one deterministic jump to
# a bigger batch and hold it - no further probing. Multi-candidate
# exploration was tried and its own cost consistently outweighed the gains
# it found, because the whole available margin over a tuned constant batch
# is small (~8% on the dev replica) and easily eaten by even a couple of
# badly-placed probe segments.
_PHASE2_MULT = 5.0


def _bopt(bcrit, t_overhead, throughput, b_min, b_max):
    b = math.sqrt(max(bcrit, 1e-6) * t_overhead * throughput)
    return int(round(min(max(b, b_min), b_max)))


def _predicted_steps1(batch):
    return _S1_MIN + _S1_C / batch


def run(env):
    b_min, b_max = env.b_min, env.b_max
    t_overhead, throughput = env.t_overhead, env.throughput

    b1 = _bopt(_BCRIT1, t_overhead, throughput, b_min, b_max)
    b2 = int(max(b_min, min(b_max, round(b1 * _PHASE2_MULT))))

    switch_detected = False

    for i in range(env.n_segments):
        batch = b2 if switch_detected else b1
        batch = int(max(b_min, min(b_max, batch)))

        r = env.run_segment(batch)
        steps = r["steps"]

        if not switch_detected and i >= _DETECT_START:
            pred = _predicted_steps1(batch)
            ratio = steps / pred if pred > 0 else 1.0
            if ratio > _DEV_STRONG or ratio < 1.0 / _DEV_STRONG or i >= _DETECT_FORCE:
                switch_detected = True
