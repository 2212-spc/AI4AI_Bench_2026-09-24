"""Chunked, cached 64-cell factorial truth for an E1 world. usage: e1_truth.py <world_json> <cache.npz> <s0> <s1>"""
import sys, json, time, os
sys.path.insert(0, "/tmp/bench/gen"); sys.path.insert(0, "/tmp/bench/lab")
import numpy as np
from multiprocessing import Pool
import e1_world as W

P = json.load(open(sys.argv[1]))
def build(params):
    return W.build_world(params)
def _old_build(params):
    base = dict(hidden=96, lr=params["lr"], momentum=params["mom"], weight_decay=0.0, warmup_steps=params["warm"], steps=400,
                min_lr_ratio=0.05, grad_accum_steps=1, micro_batch_size=32, clip_norm=None)
    deltas = {"c1": {"lr": params["lr"] * params["lr_f"]}, "c2": {"momentum": params["mom2"]}, "c3": {"clip_norm": params["clip"]},
              "c4": {"hidden": params.get("hid2", 160)}, "c5": {"weight_decay": params.get("wd2", 3e-4)}, "c6": {}}
    return W.make_world(params["tseed"], base, deltas, dict(d_in=32, t_hidden=64, noise=0.1), params["dseed"])
WORLD = build(P)

def job(a):
    m, s = a
    v, d = W.run_cell(WORLD, m, s)
    return m, s, v, d

if __name__ == "__main__":
    t = time.time(); cache = sys.argv[2]; s0, s1 = int(sys.argv[3]), int(sys.argv[4])
    jobs = [(m, s) for s in range(s0, s1) for m in range(64)]
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=16)
    V = np.full((64, s1 - s0), np.nan); D = np.zeros((64, s1 - s0), bool)
    for m, s, v, d in res: V[m, s - s0] = v; D[m, s - s0] = d
    if os.path.exists(cache):
        z = np.load(cache); V = np.concatenate([z["V"], V], 1); D = np.concatenate([z["D"], D], 1); seeds = np.concatenate([z["seeds"], np.arange(s0, s1)])
    else: seeds = np.arange(s0, s1)
    np.savez(cache, V=V, D=D, seeds=seeds, params=json.dumps(P))
    print("cells", V.shape, "time %.1f" % (time.time() - t))
