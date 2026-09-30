"""Hidden world for dedup_cv.

Rows come from *entities* (patients / documents / devices). Each entity contributes 1..m rows that are
near-copies (small jitter) and share ONE noisy label draw (label noise is at the entity level, e.g. an
annotator labelled the entity once). No entity id is given to the agent.  Test = brand-new entities.
Random K-fold CV therefore rewards memorising the entity (k=1 NN, tiny leaves); on new entities that is
exactly wrong when entity-level noise is large.  Settings differ in how complex f is and how noisy
labels are, so no fixed hyper-parameter is right everywhere."""
import numpy as np


def f_true(X, kind, W):
    if kind == "smooth":
        return np.tanh(X @ W[:, 0]) + 0.5 * np.sin(X @ W[:, 1])
    if kind == "wiggly":
        return np.sin(2.5 * X @ W[:, 0]) * np.cos(1.5 * X @ W[:, 1]) + 0.6 * np.sin(3 * X @ W[:, 2])
    if kind == "sparse":
        return 1.5 * np.tanh(2 * X[:, 0]) + np.sign(X[:, 1]) * 0.8 + 0.4 * X[:, 2] * X[:, 3]
    raise ValueError(kind)


def sample(seed, n_ent, d, kind, noise, jitter, max_rep, rep_skew, W=None, n_test_ent=1500):
    rng = np.random.default_rng(seed)
    if W is None:
        W = rng.normal(0, 1 / np.sqrt(d), (d, 3))

    def draw(n, reps):
        C = rng.normal(0, 1, (n, d))
        yc = f_true(C, kind, W) + rng.normal(0, noise, n)          # entity-level label noise
        idx = np.repeat(np.arange(n), reps)
        X = C[idx] + rng.normal(0, jitter, (len(idx), d))
        return X, yc[idx], idx, C

    reps = np.minimum(1 + rng.geometric(1 / rep_skew, n_ent) - 1, max_rep)
    Xtr, ytr, gid, _ = draw(n_ent, reps)
    perm = rng.permutation(len(ytr)); Xtr, ytr, gid = Xtr[perm], ytr[perm], gid[perm]
    # test: new entities, one row each, scored against the NOISE-FREE target
    Ct = rng.normal(0, 1, (n_test_ent, d))
    Xte = Ct + rng.normal(0, jitter, Ct.shape)
    yte = f_true(Ct, kind, W)
    return dict(X=Xtr, y=ytr, groups=gid, Xte=Xte, yte=yte)


# name, seed, n_entities, d, kind, entity_noise, jitter, max_rep, mean_rep, hidden
SETTINGS = [
    ("dev_a", 21, 500, 8, "smooth", 0.8, 0.03, 12, 5.0, False),
    ("dev_b", 22, 900, 6, "wiggly", 0.25, 0.03, 8, 3.0, False),
    ("hid_a", 201, 400, 10, "smooth", 1.1, 0.05, 20, 7.0, True),
    ("hid_b", 202, 1200, 5, "wiggly", 0.15, 0.02, 6, 2.5, True),
    ("hid_c", 203, 700, 8, "sparse", 0.6, 0.04, 15, 4.0, True),
    ("hid_d", 204, 600, 12, "wiggly", 0.5, 0.06, 10, 4.0, True),
]


def build_setting(row):
    name, seed, n_ent, d, kind, noise, jit, mr, rs, hidden = row
    s = sample(seed, n_ent, d, kind, noise, jit, mr, rs)
    s.update(name=name, hidden=hidden)
    return s
