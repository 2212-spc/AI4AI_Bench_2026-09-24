"""Nightly classifier training job.

Production runs exactly this command (single CPU core, numpy only):

    python train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy>

<corpus.npz> holds `X` (float32, n x d raw features) and `y` (int64 crowd labels in 0..K-1).
<test_X.npy> holds the features of the evaluation set; the job writes one predicted class per row to --out.
Hyper-parameters come from config.json next to this file (override with --config).

How training works (see report.md for the reasoning):
  * `--steps` x `ref_batch_size` fixes the *sample budget* (32 passes over the data in production);
    every candidate model below is trained for that many samples.
  * A fraction of the corpus (crowd labels) is held out as a validation split.
  * Several candidates that differ only in weight decay are trained in an order given by a data-size
    prior (optimal decay shrinks as the corpus grows); training stops early if the CPU budget is at risk.
  * All candidates whose validation accuracy is within `ensemble_tol` of the best are ensembled
    (averaged softmax probabilities) to produce the predictions.
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


def train_one(Xs, y, steps, cfg, weight_decay, seed=0, log_every=0):
    """Train a single MLP on already-standardised features for `steps` mini-batch steps."""
    rng = np.random.default_rng(seed)
    k = int(cfg["num_classes"])
    P = init_params(Xs.shape[1], int(cfg["hidden"]), k, rng)
    opt = AdamW(P, cfg["lr"], weight_decay)
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
            print("  step %6d  loss %.4f" % (t + 1, loss), flush=True)
    return P


def predict_proba(P, Xs):
    out = []
    for i in range(0, len(Xs), 4096):
        z, _ = forward(P, Xs[i:i + 4096])
        out.append(softmax(z))
    return np.concatenate(out)


def prior_weight_decay(n_train, cfg):
    """Data-size prior for the best weight decay.  In the sample study the optimum shrank with the
    number of *unique* training rows (~7 at 900 rows, ~3 at 1800, ~2.5 at 3700), roughly like n^-0.5.
    Only used to decide the order in which candidates are trained."""
    return float(cfg["prior_wd_at_4000"]) * (4000.0 / max(n_train, 1)) ** float(cfg["prior_wd_exponent"])


def dedupe(X, y, k):
    """Collapse exact duplicate feature rows (the corpus contains many; duplicates share their crowd
    label).  If a duplicate group ever disagrees, take the majority label."""
    Xu, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    votes = np.zeros((len(Xu), k), np.int64)
    np.add.at(votes, (inv, y), 1)
    return Xu, votes.argmax(1), cnt


def candidate_steps(n_train, steps, cfg):
    """Mini-batch steps for one candidate: `epochs_per_candidate` passes over the unique training rows,
    never more than the production step budget itself."""
    s = int(round(float(cfg["epochs_per_candidate"]) * n_train / int(cfg["batch_size"])))
    return int(max(int(cfg["min_steps"]), min(s, steps)))


def train(X, y, steps, cfg, seed=0, log_every=0, verbose=True):
    """Full pipeline; returns (list_of_params, standardiser, info)."""
    t_start = time.process_time()
    rng = np.random.default_rng(seed + 12345)
    cfg = dict(cfg)
    cfg.setdefault("num_classes", int(y.max()) + 1)
    k = int(cfg["num_classes"])

    Xu, yu, cnt = dedupe(X, y, k)
    n = len(Xu)
    norm = standardizer(Xu)
    Xs = norm(Xu)

    # held-out crowd-labelled validation split (on unique rows, so no twin of a validation row is trained on)
    n_val = int(min(cfg["val_max"], round(cfg["val_frac"] * n)))
    n_val = max(0, min(n_val, n - 1))
    if n_val < int(cfg["val_min"]):
        n_val = 0
    perm = rng.permutation(n)
    va, tr = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = Xs[tr], yu[tr], Xs[va], yu[va]

    wds = [float(w) for w in cfg["wd_candidates"]]
    prior = prior_weight_decay(len(tr), cfg)
    wds.sort(key=lambda w: abs(np.log(w) - np.log(prior)))
    steps_c = candidate_steps(len(tr), steps, cfg)
    if verbose:
        print("rows=%d unique=%d (max multiplicity %d)  train=%d  val=%d  steps/candidate=%d  prior wd=%.2f  order=%s"
              % (len(X), n, int(cnt.max()), len(tr), len(va), steps_c, prior, wds), flush=True)

    budget = float(cfg["cpu_budget_s"])
    models, per_model = [], None
    for i, wd in enumerate(wds):
        used = time.process_time() - t_start
        # keep room for the final all-data model (~1.1x a candidate, it sees slightly more rows)
        if i > 0 and used + 2.3 * per_model > budget:
            if verbose:
                print("skipping remaining candidates (cpu used %.1fs, per model %.1fs, budget %.0fs)"
                      % (used, per_model, budget), flush=True)
            break
        t0 = time.process_time()
        P = train_one(Xtr, ytr, steps_c, cfg, wd, seed=seed + i, log_every=log_every)
        dt = time.process_time() - t0
        per_model = dt if per_model is None else max(per_model, dt)
        pv = predict_proba(P, Xva) if n_val > 0 else None
        acc = float((pv.argmax(1) == yva).mean()) if n_val > 0 else 0.0
        models.append({"wd": wd, "P": P, "val_acc": acc, "val_proba": pv})
        if verbose:
            print("candidate wd=%-5g  val(crowd) acc %.4f  train time %.1fs" % (wd, acc, dt), flush=True)
        if n_val == 0:
            break

    models.sort(key=lambda m: -m["val_acc"])
    best_wd = models[0]["wd"]
    keep = [models[0]]
    if n_val > 0:
        # greedily add further candidates (within tolerance of the best) while the validation accuracy
        # of the averaged ensemble does not drop
        acc_ens = models[0]["val_acc"]
        for m in models[1:]:
            if m["val_acc"] < models[0]["val_acc"] - float(cfg["ensemble_tol"]):
                break
            pv = np.mean([e["val_proba"] for e in keep + [m]], 0)
            a = float((pv.argmax(1) == yva).mean())
            if a >= acc_ens:
                keep.append(m); acc_ens = a
    Ps = [m["P"] for m in keep]

    # final model at the selected weight decay on all unique rows (validation rows included)
    used = time.process_time() - t_start
    if cfg.get("final_full_model", True) and n_val > 0 and used + 1.2 * per_model <= budget:
        steps_f = candidate_steps(n, steps, cfg)
        Pf = train_one(Xs, yu, steps_f, cfg, best_wd, seed=seed + 1000, log_every=log_every)
        Ps.append(Pf)
        if verbose:
            print("final all-data model wd=%g (%d steps)" % (best_wd, steps_f), flush=True)
    if verbose:
        print("ensemble: candidates wd=%s + final  total cpu %.1fs" % ([m["wd"] for m in keep], time.process_time() - t_start), flush=True)
    return Ps, norm, {"models": models, "kept": [m["wd"] for m in keep], "best_wd": best_wd}


def predict(Ps, norm, X):
    Xs = norm(X)
    p = np.mean([predict_proba(P, Xs) for P in Ps], 0)
    return p.argmax(1)


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
    Ps, norm, _ = train(d["X"].astype(np.float32), d["y"].astype(np.int64), a.steps, cfg, seed=a.seed,
                        log_every=a.log_every)
    np.save(a.out, predict(Ps, norm, np.load(a.predict).astype(np.float32)).astype(np.int64))
    print("trained %d steps in %.1fs wall, %.1fs cpu" % (a.steps, time.time() - t0, time.process_time()))


if __name__ == "__main__":
    main()
