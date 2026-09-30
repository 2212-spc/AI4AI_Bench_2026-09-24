"""Experimental training core: MLP with options (noise-corrected loss, label smoothing, mixup, aug)."""
import numpy as np
import sys; sys.path.insert(0,'/app/repo')
from train import init_params, forward, backward, AdamW, lr_multiplier, standardizer

def loss_grad(z, y, k, cfg):
    """Cross-entropy on noisy labels through a symmetric-noise forward correction:
    q = (1-r) softmax(z) + r/k.  r=0 gives plain CE.  Also supports label smoothing."""
    r = float(cfg.get("noise_rate", 0.0)); s = float(cfg.get("label_smoothing", 0.0))
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    n = len(y)
    t = np.full(z.shape, s / k, np.float32); t[np.arange(n), y] += 1.0 - s
    if r <= 0:
        loss = -(t * np.log(p + 1e-12)).sum(1).mean()
        return loss, (p - t) / n
    q = (1 - r) * p + r / k
    loss = -(t * np.log(q)).sum(1).mean()
    # dL/dz = sum_j t_j * (-(1-r)/q_j) * dp_j/dz  ;  dp_j/dz_i = p_j (delta_ij - p_i)
    g = -(1 - r) * t / q                 # dL/dp
    dz = p * (g - (g * p).sum(1, keepdims=True))
    return loss, dz / n

def train(X, y, steps, cfg, seed=0, Xval=None, yval=None, log_every=0):
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    norm = standardizer(X); Xs = norm(X)
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], cfg["weight_decay"])
    bs = int(cfg["batch_size"]); n = len(Xs)
    aug = float(cfg.get("aug_sigma", 0.0)); mix = float(cfg.get("mixup", 0.0))
    for t in range(steps):
        idx = rng.integers(0, n, bs); xb, yb = Xs[idx], y[idx]
        if aug > 0: xb = xb + aug * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        loss, dz = loss_grad(z, yb, k, cfg)
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
    return P, norm

def logits(P, norm, X):
    out = []
    for i in range(0, len(X), 4096):
        out.append(forward(P, norm(X[i:i+4096]))[0])
    return np.concatenate(out)
def predict(P, norm, X): return logits(P, norm, X).argmax(1)
