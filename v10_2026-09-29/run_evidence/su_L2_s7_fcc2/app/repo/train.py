"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).

The job trains a small ensemble (see config.json, key "members"): each member is either an MLP
(d -> h -> h -> K, ReLU) on standardised raw features or, with "hidden": 0, a multinomial logistic
regression; with "quad_features": true the input is augmented with all pairwise products of the
standardised features (so the linear member is a discriminatively-trained quadratic classifier).
Members are trained one after another with AdamW (warmup + cosine); their log-probabilities are
averaged at prediction time.
"""
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- model: MLP d -> h -> h -> K, ReLU  (h == 0: linear)
def init_params(d, h, k, rng):
    if h == 0:
        return {"W3": np.zeros((d, k), np.float32), "b3": np.zeros(k, np.float32)}
    return {
        "W1": (rng.standard_normal((d, h)) * np.sqrt(2.0 / d)).astype(np.float32), "b1": np.zeros(h, np.float32),
        "W2": (rng.standard_normal((h, h)) * np.sqrt(2.0 / h)).astype(np.float32), "b2": np.zeros(h, np.float32),
        "W3": (rng.standard_normal((h, k)) * np.sqrt(1.0 / h)).astype(np.float32), "b3": np.zeros(k, np.float32),
    }


def forward(P, x):
    if "W1" not in P:
        return x @ P["W3"] + P["b3"], (x,)
    a1 = x @ P["W1"] + P["b1"]; h1 = np.maximum(a1, 0)
    a2 = h1 @ P["W2"] + P["b2"]; h2 = np.maximum(a2, 0)
    z = h2 @ P["W3"] + P["b3"]
    return z, (x, a1, h1, a2, h2)


def backward(P, cache, dz):
    if len(cache) == 1:
        return {"W3": cache[0].T @ dz, "b3": dz.sum(0)}
    x, a1, h1, a2, h2 = cache
    G = {"W3": h2.T @ dz, "b3": dz.sum(0)}
    da2 = (dz @ P["W3"].T) * (a2 > 0)
    G["W2"] = h1.T @ da2; G["b2"] = da2.sum(0)
    da1 = (da2 @ P["W2"].T) * (a1 > 0)
    G["W1"] = x.T @ da1; G["b1"] = da1.sum(0)
    return G


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def softmax_xent_grad(z, y, k, smoothing):
    """Mean cross-entropy against label-smoothed targets; returns (loss, dL/dz)."""
    logp = log_softmax(z)
    t = np.full(z.shape, smoothing / k, np.float32)
    t[np.arange(len(y)), y] += 1.0 - smoothing
    loss = -(t * logp).sum(1).mean()
    return loss, (np.exp(logp) - t) / len(y)


# ---------------------------------------------------------------- optimiser: AdamW, warmup + cosine
class AdamW:
    def __init__(self, P, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, wd, b1, b2, eps
        self.m = {n: np.zeros_like(v) for n, v in P.items()}
        self.v = {n: np.zeros_like(v) for n, v in P.items()}
        self.t = 0

    def step(self, P, G, lr_mult):
        self.t += 1
        lr = self.lr * lr_mult
        for n in P:
            self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * G[n]
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * G[n] * G[n]
            mh = self.m[n] / (1 - self.b1 ** self.t)
            vh = self.v[n] / (1 - self.b2 ** self.t)
            upd = mh / (np.sqrt(vh) + self.eps)
            if n.startswith("W"):              # decoupled weight decay on weight matrices only
                upd = upd + self.wd * P[n]
            P[n] -= (lr * upd).astype(np.float32)


def lr_multiplier(t, steps, warmup_frac):
    w = max(1, int(warmup_frac * steps))
    if t < w:
        return (t + 1) / w
    return 0.5 * (1 + np.cos(np.pi * (t - w) / max(1, steps - w)))


# ---------------------------------------------------------------- features
def quad_features(Z, iu):
    return np.concatenate([Z, Z[:, iu[0]] * Z[:, iu[1]]], 1)


def standardizer(X, quad=False):
    """Zero-mean / unit-variance features; optionally augmented with all pairwise products (also standardised)."""
    mu = X.mean(0); sd = X.std(0) + 1e-6
    if not quad:
        return lambda A: ((A - mu) / sd).astype(np.float32)
    iu = np.triu_indices(X.shape[1])
    Q = quad_features((X - mu) / sd, iu); qmu = Q.mean(0); qsd = Q.std(0) + 1e-6
    return lambda A: ((quad_features((A - mu) / sd, iu) - qmu) / qsd).astype(np.float32)


# ---------------------------------------------------------------- training
def train_member(Xs, y, steps, cfg, k, rng, log_every=0):
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], cfg["weight_decay"])
    bs = int(cfg["batch_size"])
    n = len(Xs)
    for t in range(steps):
        idx = rng.integers(0, n, bs)
        xb, yb = Xs[idx], y[idx]
        if cfg["aug_sigma"] > 0:
            xb = xb + cfg["aug_sigma"] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        loss, dz = softmax_xent_grad(z, yb, k, cfg["label_smoothing"])
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
        if log_every and (t + 1) % log_every == 0:
            print("step %6d  loss %.4f" % (t + 1, loss), flush=True)
    return P


def member_configs(cfg):
    """Each entry of cfg["members"] overrides the top-level hyper-parameters for that ensemble member."""
    base = {n: v for n, v in cfg.items() if n != "members"}
    return [dict(base, **m) for m in cfg.get("members", [{}])]


def train(X, y, steps, cfg, seed=0, log_every=0):
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    norms = {}
    models = []
    for i, mc in enumerate(member_configs(cfg)):
        quad = bool(mc.get("quad_features", False))
        if quad not in norms:
            norms[quad] = standardizer(X, quad)
        t0 = time.time()
        P = train_member(norms[quad](X), y, steps, mc, k, rng, log_every)
        if log_every:
            print("member %d (hidden=%d, quad=%s) trained in %.1fs" % (i, mc["hidden"], quad, time.time() - t0), flush=True)
        models.append((P, quad))
    return models, norms


def predict_logp(models, norms, X):
    """Average log-probability over ensemble members (rows of X in chunks)."""
    out = []
    for i in range(0, len(X), 4096):
        chunk = X[i:i + 4096]; feats = {q: f(chunk) for q, f in norms.items()}
        lp = 0
        for P, quad in models:
            z, _ = forward(P, feats[quad])
            lp = lp + log_softmax(z)
        out.append(lp / len(models))
    return np.concatenate(out)


def predict(models, norms, X):
    return predict_logp(models, norms, X).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--predict", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    d = np.load(a.data)
    t0 = time.time()
    models, norms = train(d["X"].astype(np.float32), d["y"].astype(np.int64), a.steps, cfg, seed=a.seed, log_every=a.log_every)
    np.save(a.out, predict(models, norms, np.load(a.predict).astype(np.float32)).astype(np.int64))
    print("trained %d steps in %.1fs" % (a.steps, time.time() - t0))


if __name__ == "__main__":
    main()
