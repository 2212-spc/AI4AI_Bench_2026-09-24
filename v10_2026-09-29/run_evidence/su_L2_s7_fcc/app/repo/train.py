"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).

What the job does (see report.md / README.md for the reasoning):
  1. holds out a random `val_frac` of the corpus (crowd labels) as a validation split;
  2. trains one MLP per entry of `candidates` (each entry overrides the base hyper-parameters, typically the
     weight decay) for the requested number of steps on the remaining rows, stopping early through the list
     if the CPU budget `cpu_budget_s` would be exceeded;
  3. scores every candidate on the held-out split, keeps those within `ensemble_tol` (in units of the
     standard error of the validation accuracy) of the best one, and averages their class probabilities.
The loss is cross-entropy through a symmetric label-noise model (`noise_rate`), which keeps the network from
spending capacity on fitting wrong crowd labels.  `noise_rate: 0` recovers plain cross-entropy.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one core in production: avoid BLAS thread oversubscription
import argparse, json, time
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


def softmax(z):
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def noisy_xent_grad(z, y, k, noise_rate, smoothing):
    """Cross-entropy of the observed (crowd) label under a symmetric noise model:
    q = (1 - r) * softmax(z) + r / k.  Returns (mean loss, dL/dz).  r = 0 is plain cross-entropy."""
    p = softmax(z)
    n = len(y)
    t = np.full(z.shape, smoothing / k, np.float32)
    t[np.arange(n), y] += 1.0 - smoothing
    r = float(noise_rate)
    if r <= 0:
        loss = -(t * np.log(p + 1e-12)).sum(1).mean()
        return loss, (p - t) / n
    q = (1.0 - r) * p + r / k
    loss = -(t * np.log(q)).sum(1).mean()
    g = -(1.0 - r) * t / q                                  # dL/dp
    dz = p * (g - (g * p).sum(1, keepdims=True))            # through the softmax Jacobian
    return loss, dz / n


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


# ---------------------------------------------------------------- training
def standardizer(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return lambda A: ((A - mu) / sd).astype(np.float32)


def train(X, y, steps, cfg, seed=0, log_every=0, norm=None):
    """Train one MLP on (X, y) with hyper-parameters cfg.  Returns (params, standardizer)."""
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    if norm is None:
        norm = standardizer(X)
    Xs = norm(X)
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], cfg["weight_decay"])
    bs = int(cfg["batch_size"])
    n = len(Xs)
    aug = float(cfg.get("aug_sigma", 0.0))
    for t in range(steps):
        idx = rng.integers(0, n, bs)
        xb, yb = Xs[idx], y[idx]
        if aug > 0:
            xb = xb + aug * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        loss, dz = noisy_xent_grad(z, yb, k, cfg.get("noise_rate", 0.0), cfg.get("label_smoothing", 0.0))
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
        if log_every and (t + 1) % log_every == 0:
            print("step %6d  loss %.4f" % (t + 1, loss), flush=True)
    return P, norm


def logits(P, norm, X):
    out = []
    for i in range(0, len(X), 4096):
        out.append(forward(P, norm(X[i:i + 4096]))[0])
    return np.concatenate(out)


def predict(P, norm, X):
    return logits(P, norm, X).argmax(1)


def fit_predict(X, y, steps, cfg, X_test, seed=0, log=print):
    """Full job: candidate selection on a held-out crowd-label split + ensemble of the best candidates.
    Returns predicted classes for X_test."""
    cpu0 = time.process_time()
    n = len(X)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    cands = cfg.get("candidates") or [{}]
    n_val = int(round(float(cfg.get("val_frac", 0.0)) * n))
    if n_val < int(cfg.get("min_val_rows", 200)) or len(cands) == 1:
        n_val = 0                                      # too small to select on: train first candidate on all rows
    rng = np.random.default_rng(10_000 + seed)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    norm = standardizer(X)                             # feature statistics from the whole corpus (no labels)
    budget = float(cfg.get("cpu_budget_s", 1e9))

    test_probs, val_accs, costs = [], [], []
    for i, over in enumerate(cands):
        c = dict(cfg); c.update(over)
        elapsed = time.process_time() - cpu0
        if i > 0 and elapsed + 1.1 * max(costs) > budget:
            log("candidate %d %s skipped: CPU budget (%.0fs used of %.0fs)" % (i, over, elapsed, budget))
            break
        t0 = time.process_time()
        P, _ = train(Xtr, ytr, steps, c, seed=seed * 100 + i, norm=norm)
        if n_val:
            va = float((predict(P, norm, X[val_idx]) == y[val_idx]).mean())
        else:
            va = 0.0
        costs.append(time.process_time() - t0)
        val_accs.append(va)
        test_probs.append(softmax(logits(P, norm, X_test)))
        log("candidate %d %-40s val_acc(crowd) %.4f  cpu %.1fs" % (i, json.dumps(over), va, costs[-1]))
        if n_val == 0:
            break

    val_accs = np.array(val_accs)
    if n_val:
        se = np.sqrt(0.25 / n_val)
        keep = np.where(val_accs >= val_accs.max() - float(cfg.get("ensemble_tol", 1.0)) * se)[0]
    else:
        keep = np.array([0])
    log("ensembling candidates %s" % keep.tolist())
    probs = np.mean([test_probs[j] for j in keep], axis=0)
    return probs.argmax(1)


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
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    X_test = np.load(a.predict).astype(np.float32)
    pred = fit_predict(X, y, a.steps, cfg, X_test, seed=a.seed)
    np.save(a.out, pred.astype(np.int64))
    print("trained %d steps in %.1fs (cpu %.1fs)" % (a.steps, time.time() - t0, time.process_time()))


if __name__ == "__main__":
    main()
