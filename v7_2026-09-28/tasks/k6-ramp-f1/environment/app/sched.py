# Current production schedule: a constant batch size, tuned once on a short replica.
# Replace this with your own. See /app/docs/sched_api.md.


def run(env):
    for _ in range(env.n_segments):
        env.run_segment(512)
