"""MiniLab truth engine (host side, never shown to agents).
A numpy teacher-student regression world. Every behaviour the generator can
inject is a named flag so certificates can be computed by execution."""
import numpy as np

# ---------------- world (teacher) ----------------
def make_teacher(seed, d_in=32, t_hidden=64, noise=0.1):
    r = np.random.default_rng(seed)
    return {"W1": r.standard_normal((d_in, t_hidden)) / np.sqrt(d_in),
            "b1": 0.1 * r.standard_normal(t_hidden),
            "W2": r.standard_normal(t_hidden) / np.sqrt(t_hidden),
            "noise": noise, "d_in": d_in}

def sample(teacher, n, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, teacher["d_in"]))
    f = np.tanh(x @ teacher["W1"] + teacher["b1"]) @ teacher["W2"]
    y = f + teacher["noise"] * r.standard_normal(n)
    return x, y

# ---------------- student ----------------
def init_params(rng, d_in, hidden, w2_gain=1.0):
    W1 = rng.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt(w2_gain / hidden)
    return {"W1": W1, "b1": np.zeros(hidden), "W2": W2, "b2": np.zeros(1)}

def forward(p, x):
    h = x @ p["W1"] + p["b1"]
    a = np.maximum(h, 0.0)
    return (a @ p["W2"] + p["b2"])[:, 0], (h, a)

def loss_grads(p, x, y, denom, grad_scale=1.0):
    pred, (h, a) = forward(p, x)
    r = pred - y
    g_out = (grad_scale * r / denom)[:, None]
    g_h = (g_out @ p["W2"].T) * (h > 0)
    return {"W1": x.T @ g_h, "b1": g_h.sum(0), "W2": a.T @ g_out, "b2": g_out.sum(0)}

def mse(p, x, y):
    pred, _ = forward(p, x)
    return float(np.mean((pred - y) ** 2))

def lr_at(step, cfg):
    base, warm, total, mr = cfg["lr"], cfg["warmup_steps"], cfg["steps"], cfg["min_lr_ratio"]
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return base * (mr + (1 - mr) * 0.5 * (1 + np.cos(np.pi * t)))

DEFAULT_FLAGS = dict(bug_ga=False, bug_damp=False, lr_mult=1.0, grad_scale=1.0,
                     wd_skip_bias=False, sched_per_micro=False, w2_gain=1.0, ga_mean_after=False, lr_mult_out=1.0)

def train(cfg, xtr, ytr, seed, sep_rng=False, **flags):
    f = dict(DEFAULT_FLAGS); f.update(flags)
    rng = np.random.default_rng(seed)
    p = init_params(rng, xtr.shape[1], cfg["hidden"], f["w2_gain"])
    if sep_rng:
        rng = np.random.default_rng([seed, 7919])
    buf = {k: np.zeros_like(v) for k, v in p.items()}
    k_acc, mb, mom, wd = cfg["grad_accum_steps"], cfg["micro_batch_size"], cfg["momentum"], cfg["weight_decay"]
    damp = mom if f["bug_damp"] else 0.0
    n = xtr.shape[0]
    clip = cfg.get("clip_norm")
    micro_counter = 0
    for step in range(cfg["steps"]):
        g = {k: np.zeros_like(v) for k, v in p.items()}
        for _ in range(k_acc):
            idx = rng.integers(0, n, size=mb)
            denom = mb if f["bug_ga"] else mb * k_acc
            gi = loss_grads(p, xtr[idx], ytr[idx], denom, f["grad_scale"])
            for kk in g: g[kk] += gi[kk]
            micro_counter += 1
        if clip:
            tot = np.sqrt(sum(float(np.sum(v * v)) for v in g.values()))
            if tot > clip:
                for kk in g: g[kk] *= clip / tot
        lr = f["lr_mult"] * lr_at(micro_counter - 1 if f["sched_per_micro"] else step, cfg)
        for kk in p:
            gk = g[kk]
            if wd and not (f["wd_skip_bias"] and kk.startswith("b")):
                gk = gk + wd * p[kk]
            buf[kk] *= mom
            buf[kk] += (1.0 - damp) * gk
            p[kk] -= (lr * f["lr_mult_out"] if kk in ("W2", "b2") else lr) * buf[kk]
        if not np.isfinite(p["W2"]).all() or np.abs(p["W2"]).max() > 1e6:
            return None
    return p

def eval_runs(cfg, data, seeds, **flags):
    xtr, ytr, xte, yte = data
    out = []
    for s in seeds:
        p = train(cfg, xtr, ytr, s, **flags)
        out.append(np.inf if p is None else mse(p, xte, yte))
    return np.array(out)
