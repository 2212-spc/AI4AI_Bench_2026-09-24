import sys, numpy as np; sys.path.insert(0,'/app/repo'); import train as T
def loss_grad(z, y, k, smoothing, noise_rate):
    """Cross-entropy on forward-corrected probabilities q = (1-e) p + e/k."""
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    n = len(y); e = noise_rate
    t = np.full(z.shape, smoothing / k, np.float32); t[np.arange(n), y] += 1.0 - smoothing
    if e <= 0:
        return -(t * np.log(p + 1e-12)).sum(1).mean(), (p - t) / n
    q = (1 - e) * p + e / k
    loss = -(t * np.log(q)).sum(1).mean()
    dq = -(t / q) / n                     # dL/dq
    dp = (1 - e) * dq                      # dL/dp
    dz = p * (dp - (dp * p).sum(1, keepdims=True))
    return loss, dz.astype(np.float32)
def train(X, y, steps, cfg, seed=0, Xv=None, yv=None, eval_every=0):
    rng = np.random.default_rng(seed); k = 10
    norm = T.standardizer(X); Xs = norm(X)
    P = T.init_params(Xs.shape[1], int(cfg['hidden']), k, rng); opt = T.AdamW(P, cfg['lr'], cfg['weight_decay'])
    bs = int(cfg['batch_size']); n = len(Xs); hist=[]
    for t in range(steps):
        idx = rng.integers(0, n, bs); xb, yb = Xs[idx], y[idx]
        if cfg.get('aug_sigma',0) > 0: xb = xb + cfg['aug_sigma'] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = T.forward(P, xb)
        loss, dz = loss_grad(z, yb, k, cfg.get('label_smoothing',0.0), cfg.get('noise_rate',0.0))
        G = T.backward(P, cache, dz); opt.step(P, G, T.lr_multiplier(t, steps, cfg['warmup_frac']))
        if eval_every and (t+1)%eval_every==0 and Xv is not None:
            hist.append((t+1,(T.predict(P,norm,Xv)==yv).mean()))
    return P, norm, hist
