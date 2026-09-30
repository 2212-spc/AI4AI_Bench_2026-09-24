"""usage: e8_search.py <start> <n> <S> <out.jsonl> ; resumable (skips worlds already in out)"""
import sys, json, time, os
sys.path.insert(0, "/tmp/bench/gen")
import numpy as np
from multiprocessing import Pool
import e8

WORLDS = {}
def job(a):
    i, m, s = a
    if i not in WORLDS: WORLDS[i] = e8.build(e8.sample_params(i))
    return i, m, s, e8.run_cell(WORLDS[i], m, 2000 + s)

if __name__ == "__main__":
    s0, n, S, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    done = set()
    if os.path.exists(out): done = {json.loads(l)["i"] for l in open(out)}
    t = time.time()
    with Pool(4) as pool:
        for i in range(s0, s0 + n):
            if i in done: continue
            if time.time() - t > 140: break
            res = pool.map(job, [(i, m, s) for m in range(e8.NC) for s in range(S)], chunksize=32)
            V = np.zeros((e8.NC, S))
            for _, m, s, v in res: V[m, s] = v
            mu = V.mean(1); a = e8.analyse(mu); a["i"] = i; a["params"] = e8.sample_params(i); a["mu"] = mu.round(6).tolist()
            open(out, "a").write(json.dumps(a) + "\n")
            print(i, a["n_traps"], "opt", a["opt"], "gain %.3f" % a["opt_gain"], {k: int(v) for k, v in a["traps"].items()}, "t=%.0f" % (time.time() - t), flush=True)
