"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).
"""
import argparse, json, os, time

# Production has one CPU core. Set these before loading numpy/BLAS.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
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


def softmax_xent_grad(z, y, k, smoothing, noise_rate=0.0):
    """Likelihood of crowd labels under a uniform label-corruption model.

    The network predicts clean-label probabilities p; observed labels have
    probabilities (1 - noise_rate) * p + noise_rate / k. This gives unlikely
    crowd labels bounded influence instead of forcing their memorization.
    noise_rate=0 recovers ordinary (optionally smoothed) cross entropy.
    """
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(logp)
    t = np.full(z.shape, smoothing / k, np.float32)
    t[np.arange(len(y)), y] += 1.0 - smoothing
    if noise_rate == 0:
        return -(t * logp).sum(1).mean(), (p - t) / len(y)
    observed = (1.0 - noise_rate) * p + noise_rate / k
    ratio = t / observed
    dz = (1.0 - noise_rate) * p * ((ratio * p).sum(1, keepdims=True) - ratio)
    return -(t * np.log(observed)).sum(1).mean(), dz / len(y)


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
    # The sample's regularization strength is a reference, not a constant
    # carried over to a dataset twenty times larger. Scale by corpus row count.
    reference_rows = float(cfg.get("decay_reference_rows", len(Xs)))
    wd = cfg["weight_decay"] * (reference_rows / len(Xs)) ** cfg.get("decay_power", 0.0)
    opt = AdamW(P, cfg["lr"], wd)
    bs = int(cfg["batch_size"])
    n = len(Xs)
    for t in range(steps):
        idx = rng.integers(0, n, bs)
        xb, yb = Xs[idx], y[idx]
        if cfg["aug_sigma"] > 0:
            xb = xb + cfg["aug_sigma"] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        loss, dz = softmax_xent_grad(z, yb, k, cfg["label_smoothing"], cfg.get("noise_rate", 0.0))
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
        if log_every and (t + 1) % log_every == 0:
            print("step %6d  loss %.4f" % (t + 1, loss), flush=True)
    return P, norm


def predict_proba(P, norm, X):
    out = np.empty((len(X), len(P["b3"])), dtype=np.float32)
    for i in range(0, len(X), 4096):
        z, _ = forward(P, norm(X[i:i + 4096]))
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        out[i:i + len(z)] = p / p.sum(1, keepdims=True)
    return out


def predict(P, norm, X):
    return predict_proba(P, norm, X).argmax(1)


def train_predict(X, y, steps, cfg, test_X, seed=0, log_every=0):
    """Fit independent members from the supplied corpus, then average p(y|x)."""
    if len(X) == 0 or steps <= 0:
        raise ValueError("Training requires nonempty data and positive steps")
    members = cfg.get("members", [{}])
    if not members:
        raise ValueError("At least one ensemble member is required")
    probabilities = None
    for i, overrides in enumerate(members):
        member_cfg = dict(cfg)
        member_cfg.update(overrides)
        if not 0 <= member_cfg.get("noise_rate", 0.0) < 1:
            raise ValueError("noise_rate must be in [0, 1)")
        start = time.process_time()
        P, norm = train(X, y, steps, member_cfg, seed=seed + 1009 * i,
                        log_every=log_every)
        p = predict_proba(P, norm, test_X)
        if probabilities is None:
            probabilities = p
        else:
            probabilities += p
        print("member %d/%d: %.1f CPU seconds" %
              (i + 1, len(members), time.process_time() - start), flush=True)
    return probabilities.argmax(1).astype(np.int64)


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
    with open(a.config) as f:
        cfg = json.load(f)
    with np.load(a.data, allow_pickle=False) as d:
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.int64)
    test_X = np.load(a.predict, allow_pickle=False).astype(np.float32)
    t0 = time.process_time()
    predictions = train_predict(X, y, a.steps, cfg, test_X,
                                seed=a.seed, log_every=a.log_every)
    np.save(a.out, predictions)
    print("trained %d steps per member in %.1f CPU seconds" %
          (a.steps, time.process_time() - t0))


if __name__ == "__main__":
    main()
