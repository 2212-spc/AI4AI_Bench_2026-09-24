# Adaptive batch size scheduler for fixed-token training
# Strategy: Detect corpus switch via step count increase, then adjust batch size accordingly

def run(env):
    recent_steps = []
    window_size = 3
    step_baseline = None
    switch_detected = False

    for segment_idx in range(env.n_segments):
        # Initial batches: use smaller batch for better throughput
        if segment_idx < 4:
            batch = 128
        elif segment_idx < 16:
            batch = 256
        else:
            # Post-16 strategy: monitor for switch and adapt
            if not switch_detected and len(recent_steps) >= window_size:
                avg_recent = sum(recent_steps[-window_size:]) / window_size
                if avg_recent > 1.5 * step_baseline:
                    switch_detected = True

            # If switch detected or we're past typical switch zone, use larger batch
            if switch_detected or segment_idx >= 20:
                batch = 512
            else:
                batch = 256

        result = env.run_segment(batch)

        # Update baseline and recent history
        steps = result['steps']
        if step_baseline is None:
            step_baseline = steps
        recent_steps.append(steps)
        if len(recent_steps) > window_size:
            recent_steps.pop(0)
