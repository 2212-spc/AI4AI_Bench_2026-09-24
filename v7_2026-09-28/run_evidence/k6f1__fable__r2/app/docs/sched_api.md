# Scheduler API

Your file `/app/sched.py` must define `run(env)`. It is imported and called once per run inside a sandbox
(no network, no filesystem access beyond your own file, standard library only - **no numpy**). `print()`
goes to stderr and is shown to you by `lab evaluate` when something goes wrong.

    env.n_segments    # number of segments in the run
    env.b_min         # smallest batch you may request
    env.b_max         # largest batch you may request
    env.t_overhead    # fixed seconds per optimizer step
    env.throughput    # examples/second; per-step time is t_overhead + batch/throughput

    r = env.run_segment(batch)
        # trains one segment at that batch size and returns
        #   {"segment": i, "batch": b, "steps": s, "seconds": t, "segments_left": k, "done": bool}
        # `steps` is what the segment actually needed - it is noisy.
        # Call this exactly env.n_segments times, in order. There is no undo.

    env.elapsed()     # seconds used so far

Minimal example (the current production schedule - a constant batch):

    def run(env):
        for _ in range(env.n_segments):
            env.run_segment(512)
