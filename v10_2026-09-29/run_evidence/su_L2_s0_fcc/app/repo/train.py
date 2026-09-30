"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).

Model: MLP d -> h -> h -> K (ReLU), AdamW with warmup + cosine schedule.  The job trains `n_models`
independently seeded copies (each for `--steps` steps) and averages their softmax outputs; a CPU-time
guard (`cpu_budget_s`) stops adding members if the machine turns out to be slower than expected.
"""
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- model: MLP d -> h -> h -> K, ReLU
def init_params(d, h, k, rng):
    return {
        "W1": (rng.standard_normal((d, h)) * np.sqrt(2.0 / d)).astype(np.float32), "b1": np.zeros(h, np.float32),
        "W2": (rng.standard_normal((h, h)) * np.sqrt(2.0 / h)).astype(np.float32), "b2": np.zeros(h, np.float32),
        "W3": (rng.standard_normal((h, k)) * np.sqrt(1.0 / h)).astype(np.float32), "b3": np.zeros(k, np.float32),
    }


def forward(P, x):
    a1 = x @ P["W1"] + P["b1"]; h1 = np.maximum(a1, 0)
    a2 = h1 @ P["W2"] + P["b2"]; h2 = np.maximum(a2, 0)
    z = h2 @ P["W3"] + P["b3"]
    return z, (x, a1, h1, a2, h2)


def backward(P, cache, dz):
    x, a1, h1, a2, h2 = cache
    G = {"W3": h2.T @ dz, "b3": dz.sum(0)}
    da2 = (dz @ P["W3"].T) * (a2 > 0)
    G["W2"] = h1.T @ da2; G["b2"] = da2.sum(0)
    da1 = (da2 @ P["W2"].T) * (a1 > 0)
    G["W1"] = x.T @ da1; G["b1"] = da1.sum(0)
    return G


def softmax_xent_grad(z, y, k, smoothing):
    """Mean cross-entropy against label-smoothed targets; returns (loss, dL/dz)."""
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
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
        c1 = 1 - self.b1 ** self.t; c2 = 1 - self.b2 ** self.t
        for n in P:
            m, v, g = self.m[n], self.v[n], G[n]
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * (g * g)
            upd = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if n.startswith("W"):              # decoupled weight decay on weight matrices only
                upd += self.wd * P[n]
            P[n] -= (lr * upd).astype(np.float32)


def lr_multiplier(t, steps, warmup_frac):
    w = max(1, int(warmup_frac * steps))
    if t < w:
        return (t + 1) / w
    return 0.5 * (1 + np.cos(np.pi * (t - w) / max(1, steps - w)))


# ---------------------------------------------------------------- training
def standardizer(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return lambda A: ((A - mu) / sd).astype(np.float32)


def train(X, y, steps, cfg, seed=0, log_every=0, norm=None):
    """Train one MLP; returns (params, standardiser)."""
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    norm = norm or standardizer(X)
    Xs = norm(X)
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
    return P, norm


def train_ensemble(X, y, steps, cfg, seed=0, log_every=0):
    """Train up to cfg['n_models'] independently seeded MLPs within the CPU budget; returns (list_of_params, norm)."""
    n_models = int(cfg.get("n_models", 1))
    budget = float(cfg.get("cpu_budget_s", 1e9))
    norm = standardizer(X)
    members, t_start = [], time.process_time()
    for i in range(n_models):
        P, _ = train(X, y, steps, cfg, seed=seed * 1000 + i, log_every=log_every, norm=norm)
        members.append(P)
        used = time.process_time() - t_start
        per_model = used / len(members)
        if log_every:
            print("model %d/%d done, cpu %.1fs" % (i + 1, n_models, used), flush=True)
        if used + per_model > budget:       # next member would not fit in the CPU budget
            break
    return members, norm


def probs(P, norm, X):
    z, _ = forward(P, norm(X))
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def predict(members, norm, X):
    """Average the members' softmax outputs; `members` may also be a single params dict."""
    if isinstance(members, dict):
        members = [members]
    out = []
    for i in range(0, len(X), 4096):
        p = sum(probs(P, norm, X[i:i + 4096]) for P in members)
        out.append(p.argmax(1))
    return np.concatenate(out)


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
    members, norm = train_ensemble(d["X"].astype(np.float32), d["y"].astype(np.int64), a.steps, cfg,
                                   seed=a.seed, log_every=a.log_every)
    np.save(a.out, predict(members, norm, np.load(a.predict).astype(np.float32)).astype(np.int64))
    print("trained %d models x %d steps in %.1fs (cpu %.1fs)" % (len(members), a.steps, time.time() - t0,
                                                                  time.process_time()))


if __name__ == "__main__":
    main()
