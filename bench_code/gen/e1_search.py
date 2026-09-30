"""Adversarial world search for E1 (model-mimicking baseline as a generation-time constraint).

Observation from the first 10 agent runs: frontier agents do not search blindly; they follow a *textbook path*
(test the teammate plan -> see c1+c2 overshoot -> try the textbook rescue c1+c2+c3 -> greedily add/remove the rest).
So a world where the textbook path is right is solved by priors, not by experimentation.
This script samples many worlds, computes a coarse 64-cell table (few seeds) and keeps worlds where
  - the optimum is well separated,
  - OFAT-combine, greedy forward, greedy backward AND the textbook path all end outside the accept set,
  - (blind factorial solvability is certified later with the full truth table in gen_e1.py).
usage: e1_search.py <start_idx> <n_worlds> <n_seeds> <out.jsonl>"""
import sys, json, time
sys.path.insert(0, "/tmp/bench/gen"); sys.path.insert(0, "/tmp/bench/lab")
import numpy as np
from multiprocessing import Pool
import e1_world as W


def sample_params(i):
    r = np.random.default_rng([4242, i])
    ch = lambda xs: xs[int(r.integers(len(xs)))]
    return dict(lr=ch([0.015, 0.02, 0.03, 0.04]), lr_f=ch([1.5, 2.0, 2.5, 3.0]), mom=0.9, mom2=ch([0.95, 0.97]),
                warm=10, clip=ch([0.3, 0.5, 1.0, 2.0]), hid2=ch([48, 160, 256]), wd2=ch([1e-4, 3e-4, 1e-3, 3e-3]),
                tseed=int(r.integers(1, 10**6)), dseed=int(r.integers(1, 10**6)))


def lab(m): return "+".join(W.names_of(m)) or "-"


def greedy(mu, start, add=True):
    cur = start
    while True:
        cands = [cur | 1 << i for i in range(6) if not cur >> i & 1] if add else [cur & ~(1 << i) for i in range(6) if cur >> i & 1]
        if not cands: return cur
        c = min(cands, key=lambda x: mu[x])
        if mu[c] >= mu[cur]: return cur
        cur = c


def textbook(mu):
    """mimic of observed agent behaviour (noise-free): start from the best of {singles, c1+c2, c1+c2+c3},
    then alternate greedy add / greedy remove until no single edit improves."""
    starts = [1 << i for i in range(6)] + [3, 7]
    cur = min(starts, key=lambda x: mu[x])
    while True:
        nb = [cur ^ (1 << i) for i in range(6)]
        c = min(nb, key=lambda x: mu[x])
        if mu[c] >= mu[cur]: return cur
        cur = c


def analyse(mu):
    opt = int(np.argmin(mu)); reg = mu / mu[opt] - 1
    # accept set = cells within 2.5% of optimum; require a gap to the next cell
    srt = np.sort(reg)
    acc = set(int(m) for m in np.where(reg <= 0.025)[0])
    gap = float(min([x for x in srt if x > 0.025], default=1) - max([x for x in srt if x <= 0.025]))
    ofat = sum(1 << i for i in range(6) if mu[1 << i] < mu[0])
    res = {"opt": lab(opt), "opt_gain": float(1 - mu[opt] / mu[0]), "gap": gap, "acc": [lab(m) for m in sorted(acc)],
           "ofat": lab(ofat), "fwd": lab(greedy(mu, 0)), "bwd": lab(greedy(mu, 63, add=False)), "textbook": lab(textbook(mu)),
           "local_optima": int(sum(all(mu[m] <= mu[m ^ (1 << i)] for i in range(6)) for m in range(64)))}
    res["traps"] = {k: W.mask_of(res[k].split("+")) not in acc if res[k] != "-" else 0 not in acc for k in ["ofat", "fwd", "bwd", "textbook"]}
    res["n_traps"] = sum(res["traps"].values())
    return res


WORLDS = {}


def job(a):
    i, m, s = a
    if i not in WORLDS: WORLDS[i] = W.build_world(sample_params(i))
    return i, m, s, W.run_cell(WORLDS[i], m, 2000 + s)[0]


if __name__ == "__main__":
    s0, n, S, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    t = time.time()
    with Pool(4) as pool:
        res = pool.map(job, [(i, m, s) for i in range(s0, s0 + n) for m in range(64) for s in range(S)], chunksize=32)
    for i in range(s0, s0 + n):
        V = np.zeros((64, S))
        for j, m, s, v in res:
            if j == i: V[m, s] = v
        mu = V.mean(1); a = analyse(mu); a["i"] = i; a["params"] = sample_params(i); a["mu"] = mu.round(6).tolist()
        open(out, "a").write(json.dumps(a) + "\n")
        print(i, a["n_traps"], a["traps"], "opt", a["opt"], "gain %.2f gap %.3f" % (a["opt_gain"], a["gap"]), "tb->", a["textbook"], "lo", a["local_optima"])
    print("time %.0f" % (time.time() - t))
