import sys, json, numpy as np, os
sys.path.insert(0, '/app/repo')
import importlib
S = np.load('/app/data/sample.npz'); D = np.load('/app/data/dev.npz')
X, y = S['X'].astype(np.float32), S['y'].astype(np.int64)
Xd, yd = D['X'].astype(np.float32), D['y'].astype(np.int64)
BASE = json.load(open('/app/repo/config.json'))

def run(mod, cfg, n=4000, steps=None, seed=0, sub_seed=0):
    cfg = {**BASE, **cfg}
    if n < len(X):
        idx = np.random.default_rng(1000 + sub_seed).choice(len(X), n, replace=False)
        Xt, yt = X[idx], y[idx]
    else:
        Xt, yt = X, y
    if steps is None: steps = int(n / 128 * 32)
    P, norm = mod.train(Xt, yt, steps, cfg, seed=seed)
    p = mod.predict(P, norm, Xd)
    return (p == yd).mean()
