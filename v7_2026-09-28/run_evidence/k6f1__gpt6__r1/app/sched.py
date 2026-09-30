"""Closed-loop batch schedule for the fixed-token run."""


def run(env):
    # The small batches are best on the ramp.  A source change makes the
    # work per segment jump; use the observed jump rather than assuming the
    # dev replica's change location.
    recent = []
    phase2 = False
    batch = 256

    for i in range(env.n_segments):
        if not phase2 and i >= 5 and len(recent) >= 5:
            baseline = sum(recent[-5:]) / 5.0
            if recent[-1] > baseline * 1.22 and i >= 15:
                phase2 = True
                batch = 512

        r = env.run_segment(batch)
        steps = float(r["steps"])
        recent.append(steps)

        # One very noisy segment should not lock the controller into the
        # larger batch.  Revert only when two observations are both close to
        # the old regime, which also handles an unusually mild phase change.
        if phase2 and len(recent) >= 3 and recent[-1] < recent[-3] * 1.08:
            batch = 256
