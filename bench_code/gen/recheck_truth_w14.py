# Independent re-check of the w14 truth on fresh seeds (5000-5039), disjoint from truth seeds (1000-1079) and agent seeds.
import sys, json; sys.path.insert(0, "/tmp/bench/gen"); sys.path.insert(0, "/tmp/bench/lab")
import numpy as np
from multiprocessing import Pool
import e1_world as W
P = json.load(open("/tmp/bench/e1/w14.json")); WORLD = None
CELLS = {"base":0,"c1":1,"c2":2,"c3":4,"c4":8,"c5":16,"c6":32,"c1+c2+c3":7,"c1+c2+c3+c5":23,"c1+c2+c3+c5+c6":55,
         "c3+c4":12,"c3+c4+c5":28,"c1+c2+c3+c4+c5":31,"all":63,"c5+c6":48,"c1+c5":17}
def job(a):
    global WORLD
    if WORLD is None: WORLD = W.build_world(P)
    m, s = a; v, d = W.run_cell(WORLD, m, s); return m, s, v, d
if __name__ == "__main__":
    S = list(range(5000, 5040))
    with Pool(4) as p: R = p.map(job, [(m, s) for m in CELLS.values() for s in S], chunksize=8)
    T = np.load("/tmp/bench/e1/truth14.npz"); key = [k for k in T.files]; print("truth keys", key)
    mu_t = T[key[0]].mean(1) if T[key[0]].ndim == 2 else T[key[0]]
    base = np.mean([v for m, s, v, d in R if m == 0])
    for name, m in CELLS.items():
        vs = np.array([v for mm, s, v, d in R if mm == m]); dv = sum(d for mm, s, v, d in R if mm == m)
        print("%-18s fresh %.5f ±%.5f (div %2d/40)  vs base %+6.1f%%   truth %.5f" % (name, vs.mean(), vs.std(ddof=1)/np.sqrt(len(vs)), dv, 100*(vs.mean()/base-1), mu_t[m]))
