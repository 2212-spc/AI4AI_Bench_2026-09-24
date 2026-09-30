"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).
"""
import argparse, json, os, time
# Enforce the production CPU contract before numpy loads a BLAS runtime.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
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


def train(X, y, steps, cfg, seed=0, log_every=0, deadline=None):
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    norm = standardizer(X)
    Xs = norm(X)
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    # Reference regularization is selected on 4,000 independent crowd rows.
    # Do not treat repeating a small sample as additional independent evidence.
    reference = float(cfg.get("reference_rows", len(X)))
    exponent = float(cfg.get("decay_exponent", 0.0))
    wd = float(cfg["weight_decay"]) * min(1.0, reference / len(X)) ** exponent
    opt = AdamW(P, cfg["lr"], wd)
    bs = int(cfg["batch_size"])
    n = len(Xs)
    for t in range(steps):
        if deadline is not None and t % 64 == 0 and time.process_time() >= deadline:
            print("CPU guard: stopped member at step %d/%d" % (t, steps), flush=True)
            break
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
    started_cpu = time.process_time()
    t0 = time.time()
    if a.steps <= 0:
        ap.error("--steps must be positive")
    with open(a.config) as f:
        cfg = json.load(f)
    with np.load(a.data, allow_pickle=False) as d:
        X = np.asarray(d["X"], dtype=np.float32)
        y = np.asarray(d["y"], dtype=np.int64)
    V = np.asarray(np.load(a.predict, allow_pickle=False), dtype=np.float32)
    if X.ndim != 2 or y.shape != (len(X),) or len(X) == 0:
        raise ValueError("Training inputs must be a nonempty feature matrix and label vector")
    if V.ndim != 2 or V.shape[1] != X.shape[1]:
        raise ValueError("Prediction feature dimension must match training features")
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    if y.min() < 0 or y.max() >= k:
        raise ValueError("Labels must be integer class IDs in range")
    if not np.isfinite(X).all() or not np.isfinite(V).all():
        raise ValueError("Features must be finite")
    if not len(V):
        np.save(a.out, np.empty(0, dtype=np.int64))
        return
    # Exponents bracket uncertainty in transferring regularization to 20x data.
    # Each member is fitted from scratch solely on the command-line corpus.
    exponents = cfg.get("ensemble_exponents", [cfg.get("decay_exponent", 0.0)])
    if not exponents:
        raise ValueError("At least one ensemble member is required")
    scores = np.zeros((len(V), k), np.float64)
    members = 0
    budget = min(105.0, float(cfg.get("training_cpu_budget", 100.0)))
    deadline = started_cpu + budget
    for member, exponent in enumerate(exponents):
        remaining = deadline - time.process_time()
        if members and remaining < 5.0:
            break
        local = dict(cfg, num_classes=k, decay_exponent=float(exponent))
        # Fair allocation bounds training cost even on slow numpy builds.
        member_deadline = time.process_time() + max(0.0, remaining) / (len(exponents) - member)
        wd = float(local["weight_decay"]) * min(1.0, float(local.get("reference_rows",len(X))) / len(X)) ** float(exponent)
        print("member %d: seed=%d weight_decay=%.6g" %
              (member + 1, a.seed + 1009 * member, wd), flush=True)
        P, norm = train(X, y, a.steps, local, seed=a.seed + 1009 * member,
                        log_every=a.log_every, deadline=member_deadline)
        for i in range(0, len(V), 2048):
            z, _ = forward(P, norm(V[i:i+2048]))
            z -= z.max(axis=1, keepdims=True)
            probabilities = np.exp(z)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("Nonfinite model predictions")
            scores[i:i+2048] += probabilities
        members += 1
    if not members:
        raise RuntimeError("No trained ensemble members")
    np.save(a.out, scores.argmax(axis=1).astype(np.int64))
    print("trained %d members; %.2fs CPU, %.2fs wall" %
          (members, time.process_time() - started_cpu, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
