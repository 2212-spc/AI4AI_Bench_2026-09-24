# Batch size scheduler for fixed-token training
# Optimized for 40-segment run with corpus switch between segments 16-32

def run(env):
    """
    Scheduler strategy:
    - Segments 0-15: batch 128 (phase 1, before corpus switch)
    - Segments 16-39: batch 512 (phase 2, after corpus switch)

    The corpus switch happens somewhere in segments 16-32 per run.
    This strategy uses batch 128 for the first 16 segments (safely in phase 1)
    and batch 512 for the remaining 24 segments (safely in phase 2).
    """

    for segment_idx in range(env.n_segments):
        if segment_idx < 16:
            batch = 128
        else:
            batch = 512

        env.run_segment(batch)

    return
