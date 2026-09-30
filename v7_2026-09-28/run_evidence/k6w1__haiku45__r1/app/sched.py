def run(env):
    """
    Adaptive batch scheduler for solver-7b training.

    Strategy:
    - Early phase (segments 0-23): Batch 350 (optimal for easy data)
    - Hard phase (segments 24-39): Batch 1400 (optimal for hard data)

    The data schedule changes at segment 24, causing a significant increase
    in required steps. Larger batches help amortize this cost.
    """

    # Phase 1: Early segments with easy data
    for segment_idx in range(24):
        env.run_segment(350)

    # Phase 2: Hard segments with difficult data
    for segment_idx in range(24, 40):
        env.run_segment(1400)
