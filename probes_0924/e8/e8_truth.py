"""chunked truth table for E8 world i: usage e8_truth.py <i> <cache.npz> <n_seeds_target>; seeds 0.. (disjoint from search seeds 2000+)"""
import sys, time, os
sys.path.insert(0, "/tmp/bench/gen")
import numpy as np
from multiprocessing import Pool
import e8
I = int(sys.argv[1]); WORLD = e8.build(e8.sample_params(I))
def job(a):
    m, s = a; return m, s, e8.run_cell(WORLD, m, s)
if __name__ == "__main__":
    cache, target = sys.argv[2], int(sys.argv[3]); t = time.time()
    V = np.load(cache)["V"] if os.path.exists(cache) else np.zeros((e8.NC, 0))
    with Pool(4) as pool:
        while V.shape[1] < target and time.time() - t < 120:
            s = V.shape[1]
            res = pool.map(job, [(m, s + j) for j in range(2) for m in range(e8.NC)], chunksize=16)
            col = np.zeros((e8.NC, 2))
            for m, ss, v in res: col[m, ss - s] = v
            V = np.concatenate([V, col], 1); np.savez(cache, V=V)
    print("world", I, "seeds", V.shape[1], "t=%.0f" % (time.time() - t))
