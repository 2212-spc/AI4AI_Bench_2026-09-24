"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).
"""
import argparse, json, os, time
# Limit BLAS threads before importing numpy: production has one CPU core.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
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
def standardizer(X, whitening=0.0):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    if whitening > 0:
        xs = ((X - mu) / sd).astype(np.float64)
        covariance = xs.T @ xs / len(xs)
        eigval, eigvec = np.linalg.eigh(covariance)
        scales = np.maximum(eigval, 1e-4) ** (-0.5 * whitening)
        transform = ((eigvec * scales) @ eigvec.T).astype(np.float32)
        return lambda A: (((A - mu) / sd) @ transform).astype(np.float32)
    return lambda A: ((A - mu) / sd).astype(np.float32)


def noise_corrected_grad(z, y, k, noise_rate):
    """Forward likelihood for uniform random label replacement."""
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    rows = np.arange(len(y))
    py = p[rows, y].copy()
    observed = (1.0 - noise_rate) * py + noise_rate / k
    scale = (1.0 - noise_rate) * py / np.maximum(observed, 1e-12)
    p[rows, y] -= 1.0
    return -np.log(np.maximum(observed, 1e-12)).mean(), p * (scale / len(y))[:, None]


def train(X, y, steps, cfg, seed=0, log_every=0):
    rng = np.random.default_rng(seed)
    k = int(cfg.get("num_classes", int(y.max()) + 1))
    norm = standardizer(X, float(cfg.get('whitening', 0.0)))
    Xs = norm(X)
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    wd = cfg['weight_decay'] * (float(cfg.get('reference_rows', 4000)) / len(X)) ** float(cfg.get('decay_power', 0.0))
    opt = AdamW(P, cfg['lr'], wd)
    bs = int(cfg["batch_size"])
    n = len(Xs)
    average = None
    average_count = 0
    average_start = float(cfg.get('average_start', 1.0))
    average_every = max(1, steps // 100)
    for t in range(steps):
        idx = rng.integers(0, n, bs)
        xb, yb = Xs[idx], y[idx]
        if cfg["aug_sigma"] > 0:
            xb = xb + cfg["aug_sigma"] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        if cfg.get('noise_rate', 0.0) > 0:
            loss, dz = noise_corrected_grad(z, yb, k, cfg['noise_rate'])
        else:
            loss, dz = softmax_xent_grad(z, yb, k, cfg['label_smoothing'])
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
        if t >= int(steps * average_start) and (t + 1) % average_every == 0:
            average_count += 1
            if average is None:
                average = {name: value.copy() for name, value in P.items()}
            else:
                for name in P:
                    average[name] += (P[name] - average[name]) / average_count
        if log_every and (t + 1) % log_every == 0:
            print("step %6d  loss %.4f" % (t + 1, loss), flush=True)
    return (average if average is not None else P), norm


def predict(P, norm, X):
    out = []
    for i in range(0, len(X), 4096):
        z, _ = forward(P, norm(X[i:i + 4096]))
        out.append(z.argmax(1))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def probabilities(P, norm, X):
    out = []
    for i in range(0, len(X), 2048):
        z, _ = forward(P, norm(X[i:i + 2048]))
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        out.append(p)
    return np.concatenate(out) if out else np.empty((0, len(P['b3'])), np.float32)


def fit_job(X, y, steps, cfg, seed=0, log_every=0):
    """Tune only on input crowd labels, then refit on ALL input rows.

    A fresh random holdout becomes useful with the full corpus. Its accuracy
    is an unbiased affine proxy for clean accuracy under symmetric noise.
    The small prototype still uses its supplied hyperparameters directly.
    """
    started = time.process_time()
    n = len(y)
    cfg = dict(cfg)
    cfg['num_classes'] = int(cfg.get('num_classes', int(y.max()) + 1))
    full_wd = cfg['weight_decay'] * (float(cfg.get('reference_rows', 4000)) / n) ** float(cfg.get('decay_power', 0.0))
    final_cfg = {**cfg, 'weight_decay': full_wd, 'decay_power': 0.0}
    peer = None
    if n >= int(cfg.get('tune_min_rows', 12000)) and cfg.get('tune_decay', False):
        rng = np.random.default_rng(seed + 314159)
        order = rng.permutation(n)
        nv = min(int(cfg.get('validation_max_rows', 12000)), max(1000, int(n * cfg.get('validation_fraction', 0.15))))
        vi, ti = order[:nv], order[nv:]
        tx, ty, vx, vy = X[ti], y[ti], X[vi], y[vi]
        pilot_steps = max(1, int(steps * len(ti) / n))
        best = None
        durations = []
        for multiplier in cfg.get('decay_candidates', [1.0, 0.3333333333, 3.0]):
            # Budget additional pilots while reserving time for a complete refit.
            if durations:
                estimate = max(durations)
                remaining = estimate * (1.0 + n / len(ti))
                if time.process_time() - started + 1.25 * remaining > float(cfg.get('cpu_budget', 105.0)):
                    break
            target_wd = full_wd * float(multiplier)
            pilot_cfg = {**final_cfg, 'weight_decay': target_wd * (n / len(ti)) ** float(cfg.get('decay_power', 0.0))}
            t0 = time.process_time()
            P, norm = train(tx, ty, pilot_steps, pilot_cfg, seed + 10007)
            p = probabilities(P, norm, vx)
            accuracy = float(np.mean(p.argmax(1) == vy))
            rho = float(cfg.get('noise_rate', 0.0))
            py = (1.0 - rho) * p[np.arange(nv), vy] + rho / p.shape[1]
            loss = float(-np.log(np.maximum(py, 1e-12)).mean())
            durations.append(time.process_time() - t0)
            print('validation decay=%.6g crowd_accuracy=%.4f crowd_loss=%.4f' % (target_wd, accuracy, loss), flush=True)
            key = (accuracy, -loss)
            if best is None or key > best:
                best = key
                final_cfg['weight_decay'] = target_wd
                peer = (P, norm)
        del tx, ty, vx, vy
    print('full-corpus refit rows=%d steps=%d decay=%.6g' % (n, steps, final_cfg['weight_decay']), flush=True)
    P, norm = train(X, y, steps, final_cfg, seed=seed, log_every=log_every)
    models = [(P, norm)]
    if peer is not None and cfg.get('ensemble_validation_model', True):
        models.append(peer)
    return models


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
    if a.steps <= 0:
        ap.error('--steps must be positive')
    d = np.load(a.data, allow_pickle=False)
    t0 = time.time()
    X, y = d['X'].astype(np.float32), d['y'].astype(np.int64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError('Expected a nonempty feature matrix and matching label vector')
    if not np.isfinite(X).all() or np.any(y < 0):
        raise ValueError('Features must be finite and labels nonnegative')
    models = fit_job(X, y, a.steps, cfg, seed=a.seed, log_every=a.log_every)
    test_X = np.load(a.predict, allow_pickle=False).astype(np.float32)
    if test_X.ndim != 2 or test_X.shape[1] != X.shape[1] or not np.isfinite(test_X).all():
        raise ValueError('Prediction features have the wrong shape or nonfinite values')
    pred = np.empty(len(test_X), dtype=np.int64)
    for i in range(0, len(test_X), 2048):
        batch = test_X[i:i + 2048]
        scores = sum(probabilities(P, norm, batch) for P, norm in models)
        pred[i:i + len(batch)] = scores.argmax(1)
    np.save(a.out, pred)
    print("trained %d steps in %.1fs" % (a.steps, time.time() - t0))


if __name__ == "__main__":
    main()
