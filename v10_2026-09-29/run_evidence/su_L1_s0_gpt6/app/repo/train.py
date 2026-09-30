"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).
"""
import argparse, json, os, time
# The production allocation is one CPU; set these before importing NumPy.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
    """Fit on training features; return a transform reusable for prediction."""
    mu = X.mean(0, dtype=np.float64)
    sd = X.std(0, dtype=np.float64) + 1e-6
    xs = ((X - mu) / sd).astype(np.float32)
    cov = (xs.T @ xs).astype(np.float64) / len(xs)
    eigenvalues, vectors = np.linalg.eigh(cov)
    # Relative floor also handles constant or redundant input columns.
    floor = max(1e-5, float(eigenvalues.max()) * 1e-5)
    whitening = (vectors * np.maximum(eigenvalues, floor) ** -0.5) @ vectors.T
    transform = (whitening / sd[:, None]).astype(np.float32)
    mu = mu.astype(np.float32)
    return lambda A: ((A - mu) @ transform).astype(np.float32)


def noise_grad(z, y, noise):
    """Likelihood for a clean classifier followed by uniform label corruption.

    q(y|x) = (1-noise)*softmax(z)[y] + noise/K. The responsibility
    factor reduces the gradient from labels inconsistent with the model.
    """
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    rows = np.arange(len(y))
    py = p[rows, y].copy()
    qy = (1.0 - noise) * py + noise / z.shape[1]
    responsibility = (1.0 - noise) * py / np.maximum(qy, 1e-20)
    p[rows, y] -= 1.0
    p *= (responsibility / len(y))[:, None]
    return -np.log(np.maximum(qy, 1e-20)).mean(), p


def train_member(Xs, y, steps, cfg, seed, decay, log_every=0):
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], decay)
    bs = int(cfg["batch_size"])
    averaged = {name: np.zeros_like(value) for name, value in P.items()}
    count = 0
    average_start = int(cfg.get("average_start", 0.7) * steps)
    average_interval = max(1, steps // 100)
    for t in range(steps):
        idx = rng.integers(0, len(Xs), bs)
        xb, yb = Xs[idx], y[idx]
        if cfg.get("aug_sigma", 0) > 0:
            xb = xb + cfg["aug_sigma"] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        # First learn the broad signal before downweighting inconsistent labels.
        noise = cfg.get("label_noise", 0.0) * min(1.0, t / max(1, 0.2 * steps))
        loss, dz = noise_grad(z, yb, noise)
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
        if t >= average_start and ((t + 1) % average_interval == 0 or t + 1 == steps):
            count += 1
            for name in P:
                averaged[name] += (P[name] - averaged[name]) / count
        if log_every and (t + 1) % log_every == 0:
            print("step %6d  loss %.4f" % (t + 1, loss), flush=True)
    return averaged if count else P


def train(X, y, steps, cfg, seed=0, log_every=0):
    if steps < 1 or len(X) == 0:
        raise ValueError("Training requires positive steps and a nonempty dataset")
    norm = standardizer(X)
    Xs = norm(X)
    # Small-data regularization does not transfer unchanged to the full corpus.
    ratio = len(X) / float(cfg.get("reference_rows", 4000))
    models = []
    for member, power in enumerate(cfg.get("decay_powers", [0.75])):
        decay = cfg["weight_decay"] * ratio ** (-power)
        print("member %d: rows=%d, weight_decay=%.5f" %
              (member + 1, len(X), decay), flush=True)
        models.append(train_member(Xs, y, steps, cfg, seed + 1009 * member,
                                   decay, log_every))
    return models, norm


def predict(models, norm, X):
    out = []
    for i in range(0, len(X), 2048):
        xb = norm(X[i:i + 2048])
        total = None
        for P in models:
            z, _ = forward(P, xb)
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            total = p if total is None else total + p
        out.append(total.argmax(1))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


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
    cpu0 = time.process_time()
    P, norm = train(d["X"].astype(np.float32), d["y"].astype(np.int64), a.steps, cfg, seed=a.seed, log_every=a.log_every)
    np.save(a.out, predict(P, norm, np.load(a.predict).astype(np.float32)).astype(np.int64))
    print("trained %d steps per member in %.1fs wall / %.1fs CPU" %
          (a.steps, time.time() - t0, time.process_time() - cpu0))


if __name__ == "__main__":
    main()
