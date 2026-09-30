"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).

Training recipe (see report.md):
  1. Hold out `holdout_frac` of the corpus (crowd labels) as a validation split.
  2. For every weight decay in `weight_decay_grid`, train on the rest for `select_steps_frac * steps`
     steps and score on the holdout.  Crowd accuracy is a monotone proxy for gold accuracy, so the
     best crowd score picks the regularisation strength that suits *this* corpus size.
  3. Train `ensemble` models (different seeds) on the full corpus for the full `steps` with the chosen
     weight decay; average their logits for prediction.
  A CPU-time guard (`cpu_budget_sec`) trims the ensemble if the machine is slower than expected.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # single core in production; avoids BLAS oversubscription elsewhere
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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


def fit(Xs, y, steps, cfg, k, seed, wd, log_every=0):
    """Train one MLP on already-standardised features; returns parameters."""
    rng = np.random.default_rng(seed)
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], wd)
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


def logits(models, Xs):
    out = np.zeros((len(Xs), models[0]["W3"].shape[1]), np.float32)
    for i in range(0, len(Xs), 4096):
        for P in models:
            out[i:i + 4096] += forward(P, Xs[i:i + 4096])[0]
    return out / len(models)


def train(X, y, steps, cfg, seed=0, log_every=0, verbose=True):
    """Full recipe: holdout selection of weight decay, then an ensemble on all rows.
    Returns (list of parameter dicts, standardiser)."""
    t0 = time.process_time()
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    norm = standardizer(X)
    Xs = norm(X)
    n = len(Xs)
    budget = float(cfg.get("cpu_budget_sec", 90.0))
    grid = list(cfg.get("weight_decay_grid", [cfg["weight_decay"]]))
    wd = cfg["weight_decay"]

    # ---- 1+2. holdout model selection of the weight decay
    if len(grid) > 1 and n >= int(cfg.get("min_rows_for_selection", 1000)):
        rng = np.random.default_rng(10_000 + seed)
        perm = rng.permutation(n)
        nh = int(round(cfg["holdout_frac"] * n))
        hold, rest = perm[:nh], perm[nh:]
        sel_steps = max(1, int(cfg["select_steps_frac"] * steps))
        scores = []
        for i, w in enumerate(grid):
            ts = time.process_time()
            P = fit(Xs[rest], y[rest], sel_steps, cfg, k, seed * 100 + i, w)
            acc = float((logits([P], Xs[hold]).argmax(1) == y[hold]).mean())
            scores.append(acc)
            if verbose:
                print("select wd=%-6g holdout crowd acc %.4f  (%.1fs)" % (w, acc, time.process_time() - ts), flush=True)
            # guard: if selection alone threatens the budget, stop scanning the grid
            per = (time.process_time() - t0) / (i + 1)
            if per * (len(grid) - i - 1) + time.process_time() - t0 > 0.5 * budget and i + 1 < len(grid):
                if verbose: print("selection cut short to respect CPU budget", flush=True)
                break
        wd = grid[int(np.argmax(scores))]
        if verbose: print("chosen weight_decay = %g" % wd, flush=True)

    # ---- 3. ensemble on the full corpus
    models = []
    n_ens = max(1, int(cfg.get("ensemble", 1)))
    for j in range(n_ens):
        ts = time.process_time()
        models.append(fit(Xs, y, steps, cfg, k, seed * 1000 + j, wd, log_every))
        dt = time.process_time() - ts
        if verbose: print("ensemble member %d trained (%.1fs)" % (j + 1, dt), flush=True)
        if j + 1 < n_ens and time.process_time() - t0 + dt > budget:
            if verbose: print("ensemble trimmed to %d to respect CPU budget" % (j + 1), flush=True)
            break
    return models, norm


def predict(models, norm, X):
    if isinstance(models, dict):
        models = [models]
    return logits(models, norm(X)).argmax(1)


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
    models, norm = train(d["X"].astype(np.float32), d["y"].astype(np.int64), a.steps, cfg, seed=a.seed, log_every=a.log_every)
    np.save(a.out, predict(models, norm, np.load(a.predict).astype(np.float32)).astype(np.int64))
    print("trained %d steps (%d models) in %.1fs wall, %.1fs cpu" % (a.steps, len(models), time.time() - t0, time.process_time()))


if __name__ == "__main__":
    main()
