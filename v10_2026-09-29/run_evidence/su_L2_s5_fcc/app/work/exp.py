import sys, json, time, numpy as np
sys.path.insert(0, '/app/repo')
import importlib, train as T
S = np.load('/app/data/sample.npz'); D = np.load('/app/data/dev.npz')
X, y = S['X'].astype(np.float32), S['y'].astype(np.int64)
Xd, yd = D['X'].astype(np.float32), D['y'].astype(np.int64)
BASE = json.load(open('/app/repo/config.json'))

def run(cfg=None, n=None, steps=None, seed=0, passes=32, Xtr=None, ytr=None, mod=T, **over):
    c = dict(BASE); c.update(cfg or {}); c.update(over)
    if Xtr is None:
        if n is None or n >= len(X): Xtr, ytr = X, y
        else:
            idx = np.random.default_rng(1000 + seed).choice(len(X), n, replace=False); Xtr, ytr = X[idx], y[idx]
    if steps is None: steps = int(passes * len(Xtr) / c['batch_size'])
    t0 = time.time()
    P, norm = mod.train(Xtr, ytr, steps, c, seed=seed)
    p = mod.predict(P, norm, Xd)
    return (p == yd).mean(), time.time() - t0
