"""Experiment harness: learning curves / config comparisons on sample -> dev."""
import numpy as np, sys, json, time, itertools
sys.path.insert(0, '/app/repo')
import importlib

s = np.load('/app/data/sample.npz'); d = np.load('/app/data/dev.npz')
X, y = s['X'].astype(np.float32), s['y'].astype(np.int64)
Xd, yd = d['X'].astype(np.float32), d['y'].astype(np.int64)
BASE = json.load(open('/app/repo/config.json'))


def run(cfg, n=4000, passes=32, seed=0, module='train', X=X, y=y):
    T = importlib.import_module(module)
    rng = np.random.default_rng(1234 + seed)
    idx = rng.choice(len(X), n, replace=False) if n < len(X) else np.arange(len(X))
    steps = int(passes * n / cfg['batch_size'])
    t0 = time.time()
    P, norm = T.train(X[idx], y[idx], steps, cfg, seed=seed)
    acc = (T.predict(P, norm, Xd) == yd).mean()
    return acc, time.time() - t0


def sweep(grid, ns=(1000, 2000, 4000), seeds=(0, 1), module='train', passes=32):
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(BASE); cfg.update(dict(zip(keys, vals)))
        row = []
        for n in ns:
            accs = [run(cfg, n=n, seed=sd, module=module, passes=passes)[0] for sd in seeds]
            row.append(np.mean(accs))
        print(dict(zip(keys, vals)), ' '.join('n%d=%.3f' % (n, a) for n, a in zip(ns, row)), flush=True)
