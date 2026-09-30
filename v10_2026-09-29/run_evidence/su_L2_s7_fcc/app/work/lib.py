import sys, json, numpy as np, time
sys.path.insert(0, '/app/repo')
import train as T
S = np.load('/app/data/sample.npz'); D = np.load('/app/data/dev.npz')
X, y = S['X'].astype(np.float32), S['y'].astype(np.int64)
Xd, yd = D['X'].astype(np.float32), D['y'].astype(np.int64)
BASE = json.load(open('/app/repo/config.json'))

def run(cfg=None, n=4000, steps=None, seed=0, sub_seed=0, mod=T, **kw):
    c = dict(BASE); c.update(cfg or {}); c.update(kw)
    if n < len(X):
        r = np.random.default_rng(1000 + sub_seed); idx = r.choice(len(X), n, replace=False)
        Xt, yt = X[idx], y[idx]
    else:
        Xt, yt = X, y
    if steps is None: steps = int(round(32 * n / c['batch_size']))
    P, norm = mod.train(Xt, yt, steps, c, seed=seed)
    p = mod.predict(P, norm, Xd)
    return (p == yd).mean(), P, norm

def acc(*a, seeds=(0,1), **kw):
    return np.mean([run(*a, seed=s, sub_seed=s, **kw)[0] for s in seeds])
