"""Noisy budgeted policy simulation for E8 worlds against an execution truth table V[256, S].
Each lab run = (cell, seed) -> V[cell, seed]; agents pick seeds (common random numbers allowed).
usage: e8_sim.py <truth.npz> [budget] [n_sim]"""
import sys, json, itertools
sys.path.insert(0, "/tmp/bench/gen")
import numpy as np
import e8

K, NC = e8.K, e8.NC
PRED_TOL = 0.06


def feats(masks, order=2):
    X = np.array([[1.0 if m >> i & 1 else -1.0 for i in range(K)] for m in masks])
    cols = [np.ones(len(masks))] + [X[:, i] for i in range(K)]
    if order >= 2: cols += [X[:, i] * X[:, j] for i in range(K) for j in range(i + 1, K)]
    return np.stack(cols, 1)

ALLF = feats(range(NC))


def ridge_predict(obs, lam=1e-3):
    ms = list(obs); y = np.array([np.mean(obs[m]) for m in ms]); w = np.array([len(obs[m]) for m in ms], float)
    X = feats(ms); Wd = np.sqrt(w)[:, None]
    A = (X * Wd).T @ (X * Wd) + lam * np.eye(X.shape[1]) * len(ms); b = (X * Wd).T @ (y * np.sqrt(w))
    return ALLF @ np.linalg.solve(A, b)


# 2^(8-3) resolution IV: F=ABC, G=ABD, H=BCDE  (factors A..H = c1..c8)
def frac_design():
    out = []
    for a, b, c, d, e in itertools.product([0, 1], repeat=5):
        s = lambda *v: sum(v) % 2  # parity in {0,1} coding == product in +-1 coding up to sign (any sign choice is a valid fraction)
        f, g, h = s(a, b, c), s(a, b, d), s(b, c, d, e)
        bits = [a, b, c, d, e, f, g, h]
        out.append(sum(bit << i for i, bit in enumerate(bits)))
    return out

FRAC = frac_design()


class Lab:
    def __init__(self, V, rng, budget):
        self.V, self.rng, self.B, self.used = V, rng, budget, 0
        self.perm = rng.permutation(V.shape[1])
    def run(self, m, k):  # k-th seed of this episode's seed order (common random numbers)
        if self.used >= self.B: raise StopIteration
        self.used += 1; return float(self.V[m, self.perm[k]])


def P_expert(V, rng, B, n_final=3, n_rep=4, local=False):
    """blind adaptive expert: res-IV screen (32) -> model-guided exploitation -> replicate the finalists."""
    L = Lab(V, rng, B); obs = {}
    for m in FRAC: obs.setdefault(m, []).append(L.run(m, 0))
    reserve = n_final * n_rep; turn = 0
    while L.used < B - reserve:
        pred = ridge_predict(obs)
        if local and turn % 2 == 1:  # check the one-flip neighbourhood of the current best observed cell
            best_obs = min(obs, key=lambda m: np.mean(obs[m]))
            cand = [m for m in e8.singles(best_obs) if m not in obs]
            nxt = min(cand, key=lambda m: pred[m]) if cand else None
        else:
            nxt = next((int(m) for m in np.argsort(pred) if m not in obs), None)
        turn += 1
        if nxt is None:
            if turn < 4 * B: continue
            break
        obs[nxt] = [L.run(nxt, 0)]
    pred = ridge_predict(obs)
    score = {m: 0.5 * np.mean(v) + 0.5 * pred[m] for m, v in obs.items()}
    fin = sorted(score, key=score.get)[:n_final]
    for m in fin:
        for k in range(1, 1 + n_rep):
            if L.used < B: obs[m].append(L.run(m, k))
    best = min(fin, key=lambda m: np.mean(obs[m]))
    return best, float(np.mean(obs[best])), L.used


def _eval(L, cache, m, r):
    if m not in cache: cache[m] = np.mean([L.run(m, k) for k in range(r)])
    return cache[m]


def _hill(V, rng, B, start_cands, moves, r):
    L = Lab(V, rng, B); cache = {}
    try:
        cur = min(start_cands, key=lambda m: _eval(L, cache, m, r))
        while True:
            cands = moves(cur)
            vals = {c: _eval(L, cache, c, r) for c in cands}
            c = min(vals, key=vals.get)
            if vals[c] >= cache[cur]: break
            cur = c
    except StopIteration:
        cur = min(cache, key=cache.get)
    return cur, float(cache[cur]), L.used


def P_ofat(V, rng, B, r=None):
    r = r or B // (K + 1); L = Lab(V, rng, B)
    base = np.mean([L.run(0, k) for k in range(r)]); m = 0; pred = base
    for i in range(K):
        d = np.mean([L.run(1 << i, k) for k in range(r)]) - base
        if d < 0: m |= 1 << i; pred += d
    return m, float(pred), L.used


POL = {
    "ofat": P_ofat,
    "fwd": lambda V, rng, B: _hill(V, rng, B, [0], e8.adds, 2),
    "bwd": lambda V, rng, B: _hill(V, rng, B, [NC - 1], e8.rems, 2),
    "textbook": lambda V, rng, B: _hill(V, rng, B, [1 << i for i in range(K)] + [3, 7], e8.singles, 1),
    "pairhill": lambda V, rng, B: _hill(V, rng, B, [1 << i for i in range(K)] + e8.PAIRS, e8.singles, 1),
    "bwd2": lambda V, rng, B: _hill(V, rng, B, [NC - 1], e8.rems2, 1),
    "fwd2": lambda V, rng, B: _hill(V, rng, B, [0], e8.adds2, 1),
    "flip_from_all": lambda V, rng, B: _hill(V, rng, B, [NC - 1], e8.singles, 2),
    "EXPERT": P_expert,
}


def simulate(V, B=64, n_sim=300, delta=None, seed=0):
    mu = V.mean(1); opt = int(np.argmin(mu)); reg = mu / mu[opt] - 1
    if delta is None: delta = 0.025
    rng = np.random.default_rng(seed); out = {}
    for name, f in POL.items():
        nd = nb = 0; ends = {}
        for _ in range(n_sim):
            m, pred, used = f(V, rng, B); assert used <= B
            okd = reg[m] <= delta; okp = abs(pred - mu[m]) / mu[m] <= PRED_TOL
            nd += okd; nb += okd and okp; k = e8.lab(m); ends[k] = ends.get(k, 0) + 1
        out[name] = {"pass_dec": round(nd / n_sim, 3), "pass": round(nb / n_sim, 3),
                     "top_end": sorted(ends.items(), key=lambda t: -t[1])[:2]}
    return out


if __name__ == "__main__":
    z = np.load(sys.argv[1]); V = z["V"]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 64; n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    for k, v in simulate(V, B, n).items(): print(k, v)
