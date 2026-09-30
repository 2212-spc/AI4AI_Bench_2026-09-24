"""Experimental training variants; evaluated by learning curves on subsamples of the sample."""
import numpy as np, sys, json, time
sys.path.insert(0, '/app/repo')
import train as T

s = np.load('/app/data/sample.npz'); d = np.load('/app/data/dev.npz')
X, y = s['X'], s['y']; Xd, yd = d['X'], d['y']
BASE = json.load(open('/app/repo/config.json'))


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def train_one(Xs, y, steps, cfg, rng, k=10, soft=None):
    """Xs standardised. soft: optional (n,k) soft targets replacing one-hot."""
    P = T.init_params(Xs.shape[1], int(cfg['hidden']), k, rng)
    opt = T.AdamW(P, cfg['lr'], cfg['weight_decay'])
    bs = int(cfg['batch_size']); n = len(Xs); ls = cfg['label_smoothing']
    for t in range(steps):
        idx = rng.integers(0, n, bs)
        xb = Xs[idx]
        if cfg['aug_sigma'] > 0:
            xb = xb + cfg['aug_sigma'] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = T.forward(P, xb)
        if soft is None:
            loss, dz = T.softmax_xent_grad(z, y[idx], k, ls)
        else:
            tb = soft[idx] * (1 - ls) + ls / k
            p = softmax(z); dz = (p - tb) / bs
        if cfg.get('mixup', 0) > 0:
            pass
        G = T.backward(P, cache, dz)
        opt.step(P, G, T.lr_multiplier(t, steps, cfg['warmup_frac']))
    return P


def fit_predict(Xtr, ytr, Xte, cfg, seed, passes=32):
    rng = np.random.default_rng(seed)
    norm = T.standardizer(Xtr); Xs = norm(Xtr); Xt = norm(Xte)
    steps = int(passes * len(Xtr) / cfg['batch_size'])
    k = 10
    n_models = int(cfg.get('n_models', 1))
    probs = np.zeros((len(Xte), k)); probs_tr = np.zeros((len(Xtr), k))
    for m in range(n_models):
        P = train_one(Xs, ytr, steps, cfg, rng, k)
        probs += softmax(T.forward(P, Xt)[0]); probs_tr += softmax(T.forward(P, Xs)[0])
    probs /= n_models; probs_tr /= n_models
    boot = cfg.get('bootstrap', 0.0)
    if boot > 0:  # second phase: soft targets mixing crowd label with ensemble prediction
        onehot = np.eye(k, dtype=np.float32)[ytr]
        soft = ((1 - boot) * onehot + boot * probs_tr).astype(np.float32)
        probs2 = np.zeros((len(Xte), k))
        for m in range(n_models):
            P = train_one(Xs, ytr, steps, cfg, rng, k, soft=soft)
            probs2 += softmax(T.forward(P, Xt)[0])
        probs = probs2 / n_models
    return probs.argmax(1)


def curve(cfg, ns=(1000, 2000, 4000), seeds=(0, 1, 2), passes=32):
    out = []
    for n in ns:
        accs = []
        for sd in seeds:
            r = np.random.default_rng(100 + sd)
            idx = r.choice(len(X), n, replace=False) if n < len(X) else np.arange(len(X))
            accs.append((fit_predict(X[idx], y[idx], Xd, cfg, sd, passes) == yd).mean())
        out.append(np.mean(accs))
    return out


if __name__ == '__main__':
    name = sys.argv[1]
    over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    ns = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [1000, 2000, 4000]
    cfg = dict(BASE, **over)
    t0 = time.time()
    c = curve(cfg, ns)
    print(name, over, [round(v, 4) for v in c], '%.0fs' % (time.time() - t0), flush=True)
