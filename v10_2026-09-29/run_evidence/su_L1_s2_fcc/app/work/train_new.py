"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).

Weight decay is the one hyper-parameter whose optimum moves a lot with corpus size (heavy decay
was right for the 4k-row prototype, far too heavy for the full corpus).  If config.json lists
`weight_decay_candidates`, the job holds out `holdout_frac` of the corpus, trains one shorter model
per candidate, picks the candidate with the best accuracy against the held-out crowd labels
(under roughly symmetric label noise that is monotone in gold accuracy), then retrains on all rows.
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


def train(X, y, steps, cfg, seed=0, log_every=0):
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    norm = standardizer(X)
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


def predict(P, norm, X):
    out = []
    for i in range(0, len(X), 4096):
        z, _ = forward(P, norm(X[i:i + 4096]))
        out.append(z.argmax(1))
    return np.concatenate(out)


def select_weight_decay(X, y, steps, cfg, seed, log=print):
    """Pick weight decay from cfg['weight_decay_candidates'] by accuracy on a held-out crowd split."""
    cands = list(cfg["weight_decay_candidates"])
    if len(cands) == 1:
        return cands[0]
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(X))
    n_hold = int(cfg.get("holdout_frac", 0.1) * len(X))
    hold, tr = perm[:n_hold], perm[n_hold:]
    # same number of passes over the (smaller) training split as the final run
    sel_steps = int(round(steps * len(tr) / len(X) * cfg.get("select_steps_frac", 1.0)))
    best, best_acc = None, -1.0
    for wd in cands:
        c = dict(cfg); c["weight_decay"] = wd
        P, norm = train(X[tr], y[tr], sel_steps, c, seed=seed)
        acc = (predict(P, norm, X[hold]) == y[hold]).mean()
        log("  candidate weight_decay=%-5g  holdout crowd-acc %.4f" % (wd, acc))
        if acc > best_acc:
            best, best_acc = wd, acc
    return best


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
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    t0 = time.time()
    if cfg.get("weight_decay_candidates"):
        cfg["weight_decay"] = select_weight_decay(X, y, a.steps, cfg, a.seed)
        print("selected weight_decay=%g (%.1fs)" % (cfg["weight_decay"], time.time() - t0), flush=True)
    P, norm = train(X, y, a.steps, cfg, seed=a.seed, log_every=a.log_every)
    np.save(a.out, predict(P, norm, np.load(a.predict).astype(np.float32)).astype(np.int64))
    print("trained %d steps in %.1fs" % (a.steps, time.time() - t0))


if __name__ == "__main__":
    main()
