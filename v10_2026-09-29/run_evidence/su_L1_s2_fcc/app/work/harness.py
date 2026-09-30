import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
sys.path.insert(0, "/app/repo")
import train as T

S = np.load("/app/data/sample.npz"); D = np.load("/app/data/dev.npz")
Xs, ys = S["X"].astype(np.float32), S["y"].astype(np.int64)
Xd, yd = D["X"].astype(np.float32), D["y"].astype(np.int64)
BASE = json.load(open("/app/repo/config.json"))

def run(cfg_over, n=4000, steps=None, seed=0, sub_seed=0):
    cfg = dict(BASE); cfg.update(cfg_over)
    if n < len(Xs):
        r = np.random.default_rng(sub_seed); idx = r.choice(len(Xs), n, replace=False)
        X, y = Xs[idx], ys[idx]
    else:
        X, y = Xs, ys
    if steps is None: steps = n * 32 // int(cfg["batch_size"])
    P, norm = T.train(X, y, steps, cfg, seed=seed)
    return (T.predict(P, norm, Xd) == yd).mean()
