import sys

def run(env):
    records = []
    for i in range(env.n_segments):
        res = env.run_segment(512)
        records.append(f"{res['segment']}:{res['steps']:.5f}")
    print(" ".join(records[:20]), file=sys.stderr)
    print(" ".join(records[20:]), file=sys.stderr)
