"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).

Pipeline (see report.md for the reasoning behind each step):
  1. exact duplicate rows are collapsed (the corpus contains many; duplicates share their crowd label);
  2. a slice of the unique rows is held out as a crowd-labelled validation split;
  3. several MLP candidates that differ only in weight decay are trained (generalised cross-entropy loss,
     which is robust to the ~30% wrong crowd labels) in an order given by a data-size prior, each for
     `epochs_per_candidate` passes over the unique training rows;
  4. the validation split picks the weight decay and greedily assembles an ensemble (candidates plus a
     cheap LDA model, kept only if they improve validation accuracy);
  5. extra models at the selected weight decay are trained on all unique rows and added to the ensemble.
  All stages are CPU-time aware (`cpu_budget_s`) so the job stays well inside the production limit.
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


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def loss_grad(z, y, k, cfg):
    """Mean loss over the batch and dL/dz.

    loss = "ce":  cross-entropy with optional label smoothing.
    loss = "gce": generalised cross-entropy (Zhang & Sabuncu 2018), L = (1 - p_y^q) / q, q in (0, 1].
                  q -> 0 recovers cross-entropy, q = 1 is the (noise-robust) MAE; the gradient weight
                  p_y^q down-weights examples the model finds implausible, i.e. mostly the wrong labels.
    """
    n = len(y)
    zc = z - z.max(1, keepdims=True)
    logp = zc - np.log(np.exp(zc).sum(1, keepdims=True))
    p = np.exp(logp)
    rows = np.arange(n)
    if cfg.get("loss", "gce") == "ce":
        s = float(cfg.get("label_smoothing", 0.0))
        t = np.full(z.shape, s / k, np.float32)
        t[rows, y] += 1.0 - s
        return -(t * logp).sum(1).mean(), (p - t) / n
    q = float(cfg.get("gce_q", 0.5))
    py = np.maximum(p[rows, y], 1e-6)
    loss = ((1.0 - py ** q) / q).mean()
    w = py ** q                                   # dL/dz = w * (p - onehot) for GCE
    dz = p * w[:, None]
    dz[rows, y] -= w
    return loss, (dz / n).astype(np.float32)


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


# ---------------------------------------------------------------- helpers
def standardizer(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return lambda A: ((A - mu) / sd).astype(np.float32)


def dedupe(X, y, k):
    """Collapse exact duplicate feature rows; a group that disagrees on its label takes the majority."""
    Xu, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    votes = np.zeros((len(Xu), k), np.int64)
    np.add.at(votes, (inv, y), 1)
    return Xu, votes.argmax(1), cnt


def prior_weight_decay(n_train, cfg):
    """Data-size prior for the best weight decay (it shrinks as the number of unique rows grows).
    Only used to order the candidates so that the most plausible ones are trained first."""
    return float(cfg["prior_wd_at_4000"]) * (4000.0 / max(n_train, 1)) ** float(cfg["prior_wd_exponent"])


def candidate_steps(n_train, steps, cfg):
    """Mini-batch steps for one model: `epochs_per_candidate` passes over the unique training rows,
    at least `min_steps`, never more than the production step budget."""
    s = int(round(float(cfg["epochs_per_candidate"]) * n_train / int(cfg["batch_size"])))
    return int(max(int(cfg["min_steps"]), min(s, steps)))


class MLPModel:
    def __init__(self, P): self.P = P
    def proba(self, Xs):
        out = []
        for i in range(0, len(Xs), 4096):
            z, _ = forward(self.P, Xs[i:i + 4096])
            out.append(softmax(z))
        return np.concatenate(out)


class LDAModel:
    """Linear discriminant analysis with a lightly shrunk shared covariance: cheap, low variance."""
    def __init__(self, Xs, y, k, shrink):
        self.means = np.array([Xs[y == c].mean(0) if (y == c).any() else Xs.mean(0) for c in range(k)])
        self.logpri = np.log((np.bincount(y, minlength=k) + 1.0) / (len(y) + k))
        R = Xs - self.means[y]
        S = R.T @ R / len(Xs)
        S = (1 - shrink) * S + shrink * np.eye(Xs.shape[1]) * np.trace(S) / Xs.shape[1]
        self.Si = np.linalg.inv(S)
    def proba(self, Xs):
        sc = np.stack([-0.5 * np.einsum("ij,jk,ik->i", Xs - m, self.Si, Xs - m) + lp
                       for m, lp in zip(self.means, self.logpri)], 1)
        return softmax(sc)


def train_one(Xs, y, steps, cfg, weight_decay, seed=0, log_every=0):
    """Train a single MLP on standardised features for `steps` mini-batch steps."""
    rng = np.random.default_rng(seed)
    k = int(cfg["num_classes"])
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], weight_decay)
    bs = int(cfg["batch_size"])
    n = len(Xs)
    for t in range(steps):
        idx = rng.integers(0, n, bs)
        xb, yb = Xs[idx], y[idx]
        if cfg.get("aug_sigma", 0.0) > 0:
            xb = xb + cfg["aug_sigma"] * rng.standard_normal(xb.shape).astype(np.float32)
        z, cache = forward(P, xb)
        loss, dz = loss_grad(z, yb, k, cfg)
        G = backward(P, cache, dz)
        opt.step(P, G, lr_multiplier(t, steps, cfg["warmup_frac"]))
        if log_every and (t + 1) % log_every == 0:
            print("  step %6d  loss %.4f" % (t + 1, loss), flush=True)
    return MLPModel(P)


# ---------------------------------------------------------------- full pipeline
def train(X, y, steps, cfg, seed=0, log_every=0, verbose=True):
    """Returns (list_of_models, standardiser, info).  Prediction = argmax of the mean class probabilities."""
    t_start = time.process_time()
    elapsed = lambda: time.process_time() - t_start
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    rng = np.random.default_rng(seed + 12345)
    cfg = dict(cfg)
    cfg.setdefault("num_classes", int(y.max()) + 1)
    k = int(cfg["num_classes"])
    budget = float(cfg["cpu_budget_s"])

    Xu, yu, cnt = dedupe(X, y, k)
    n = len(Xu)
    norm = standardizer(Xu)
    Xs = norm(Xu)

    # validation split on unique rows (so no twin of a validation row is ever trained on)
    n_val = int(min(cfg["val_max"], round(cfg["val_frac"] * n)))
    n_val = max(0, min(n_val, n - 1))
    if n_val < int(cfg["val_min"]):
        n_val = 0
    perm = rng.permutation(n)
    va, tr = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = Xs[tr], yu[tr], Xs[va], yu[va]
    val_acc = (lambda p: float((p.argmax(1) == yva).mean())) if n_val > 0 else (lambda p: 0.0)

    wds = [float(w) for w in cfg["wd_candidates"]]
    prior = prior_weight_decay(len(tr), cfg)
    wds.sort(key=lambda w: abs(np.log(w) - np.log(prior)))
    steps_c = candidate_steps(len(tr), steps, cfg)
    say("rows=%d unique=%d (max multiplicity %d)  train=%d val=%d  steps/candidate=%d  prior wd=%.2f  order=%s"
        % (len(X), n, int(cnt.max()), len(tr), len(va), steps_c, prior, wds))

    # --- stage 1: candidates
    cands, per_model = [], 0.0
    for i, wd in enumerate(wds):
        if i > 0 and elapsed() + 2.2 * per_model > budget:      # keep room for one all-data model
            say("skipping remaining candidates (cpu used %.1fs, per model %.1fs, budget %.0fs)" % (elapsed(), per_model, budget))
            break
        t0 = time.process_time()
        m = train_one(Xtr, ytr, steps_c, cfg, wd, seed=seed + i, log_every=log_every)
        per_model = max(per_model, time.process_time() - t0)
        pv = m.proba(Xva) if n_val > 0 else None
        cands.append({"name": "mlp wd=%g" % wd, "wd": wd, "model": m, "val_proba": pv, "val_acc": val_acc(pv)})
        say("candidate wd=%-5g  val(crowd) acc %.4f  train time %.1fs" % (wd, cands[-1]["val_acc"], time.process_time() - t0))
        if n_val == 0:
            break
    cands.sort(key=lambda c: -c["val_acc"])
    best_wd = cands[0]["wd"]

    # --- stage 2: greedy ensemble on validation (candidates within tolerance of the best, plus LDA)
    keep = [cands[0]]
    if n_val > 0:
        extra = [c for c in cands[1:] if c["val_acc"] >= cands[0]["val_acc"] - float(cfg["ensemble_tol"])]
        if cfg.get("lda_shrink", -1) >= 0:
            lda = LDAModel(Xtr, ytr, k, float(cfg["lda_shrink"]))
            pv = lda.proba(Xva)
            extra.append({"name": "lda", "wd": None, "model": lda, "val_proba": pv, "val_acc": val_acc(pv)})
            say("lda                val(crowd) acc %.4f" % extra[-1]["val_acc"])
        acc_ens = keep[0]["val_acc"]
        for c in extra:
            a = val_acc(np.mean([e["val_proba"] for e in keep + [c]], 0))
            if a >= acc_ens:
                keep.append(c); acc_ens = a
        say("selected wd=%g; ensemble on validation: %s -> val acc %.4f" % (best_wd, [c["name"] for c in keep], acc_ens))
    models = [c["model"] for c in keep]
    use_lda = any(c["name"] == "lda" for c in keep)

    # --- stage 3: models at the selected weight decay on all unique rows
    if n_val > 0:
        steps_f = candidate_steps(n, steps, cfg)
        for j in range(int(cfg["final_models"])):
            if elapsed() + 1.25 * per_model * steps_f / max(steps_c, 1) > budget:
                say("stopping final models (cpu used %.1fs, budget %.0fs)" % (elapsed(), budget))
                break
            models.append(train_one(Xs, yu, steps_f, cfg, best_wd, seed=seed + 1000 + j, log_every=log_every))
            say("final all-data model %d wd=%g (%d steps)" % (j + 1, best_wd, steps_f))
        if use_lda:
            models.append(LDAModel(Xs, yu, k, float(cfg["lda_shrink"])))
    say("ensemble of %d models, total cpu %.1fs" % (len(models), elapsed()))
    return models, norm, {"cands": cands, "kept": [c["name"] for c in keep], "best_wd": best_wd}


def predict(models, norm, X):
    Xs = norm(X)
    return np.mean([m.proba(Xs) for m in models], 0).argmax(1)


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
    models, norm, _ = train(d["X"].astype(np.float32), d["y"].astype(np.int64), a.steps, cfg, seed=a.seed,
                            log_every=a.log_every)
    np.save(a.out, predict(models, norm, np.load(a.predict).astype(np.float32)).astype(np.int64))
    print("trained %d steps in %.1fs wall, %.1fs cpu" % (a.steps, time.time() - t0, time.process_time()))


if __name__ == "__main__":
    main()
