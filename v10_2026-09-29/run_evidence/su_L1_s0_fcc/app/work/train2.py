"""Experimental trainer: baseline + EMA + ensemble + robust loss + n-scaled weight decay."""
import sys, json, numpy as np
sys.path.insert(0, '/app/repo')
from train import init_params, forward, backward, AdamW, lr_multiplier, standardizer

def loss_grad(z, y, k, cfg):
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(logp)
    sm = cfg.get("label_smoothing", 0.0)
    t = np.full(z.shape, sm / k, np.float32); t[np.arange(len(y)), y] += 1.0 - sm
    q = cfg.get("gce_q", 0.0)
    if q > 0:   # generalized CE: (1 - p_y^q)/q ; grad wrt z = p_y^q * (p - onehot)
        py = p[np.arange(len(y)), y]
        w = (py ** q)[:, None]
        return (p - t) * w / len(y)
    return (p - t) / len(y)

def effective_wd(cfg, n):
    wd = cfg["weight_decay"]
    a = cfg.get("wd_alpha", 0.0)
    if a > 0:
        wd = wd * (cfg.get("wd_ref_n", 4000) / n) ** a
    return max(wd, cfg.get("wd_min", 0.0))

def train_one(Xs, y, steps, cfg, rng, k):
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], effective_wd(cfg, len(Xs)))
    bs = int(cfg["batch_size"]); n = len(Xs)
    ema = cfg.get("ema", 0.0); E = {a: v.copy() for a, v in P.items()} if ema else None
    ema_start = int(cfg.get("ema_start", 0.5) * steps)
    for t in range(steps):
        idx = rng.integers(0, n, bs); xb, yb = Xs[idx], y[idx]
        if cfg.get("aug_sigma", 0) > 0:
            xb = xb + cfg["aug_sigma"] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        dz = loss_grad(z, yb, k, cfg)
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
        if ema and t >= ema_start:
            if t == ema_start:
                for a in P: E[a][...] = P[a]
            else:
                for a in P: E[a] += (1 - ema) * (P[a] - E[a])
    return E if ema else P

def train(X, y, steps, cfg, seed=0):
    rng = np.random.default_rng(seed)
    k = int(y.max()) + 1
    norm = standardizer(X); Xs = norm(X)
    m = int(cfg.get("n_models", 1))
    models = [train_one(Xs, y, steps // m if cfg.get("split_steps", False) else steps, cfg, rng, k) for _ in range(m)]
    return models, norm

def predict_proba(models, norm, X):
    out = []
    for i in range(0, len(X), 4096):
        xb = norm(X[i:i+4096]); acc = 0
        for P in models:
            z, _ = forward(P, xb); z = z - z.max(1, keepdims=True); p = np.exp(z); acc = acc + p / p.sum(1, keepdims=True)
        out.append(acc / len(models))
    return np.concatenate(out)

def predict(models, norm, X):
    return predict_proba(models, norm, X).argmax(1)
