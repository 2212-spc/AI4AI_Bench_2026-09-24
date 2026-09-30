import sys, json
sys.path.insert(0, "/tmp/bench/gen")
import numpy as np, e8, e8_sim as S
EXPERT_KW = dict(local=True, n_final=3, n_rep=8)   # frozen after tuning on world 1 only
B = 96

def pick_delta(V):
    mu = V.mean(1); opt = int(np.argmin(mu)); reg = mu / mu[opt] - 1
    se = np.array([np.std(V[m] - V[opt], ddof=1) / np.sqrt(V.shape[1]) for m in range(e8.NC)]) / mu[opt]
    best = None
    for d in np.arange(0.020, 0.0451, 0.0005):
        c = np.min(np.abs(reg - d) / np.maximum(se, 1e-9))
        if best is None or c > best[1]: best = (float(d), float(c))
    return opt, reg, best

def run(i, n=300):
    V = np.load(f"/tmp/bench/e8/truth{i}.npz")["V"]; mu = V.mean(1)
    opt, reg, (delta, clear) = pick_delta(V)
    a = e8.analyse(mu, delta)
    out = {"world": i, "seeds": V.shape[1], "opt": e8.lab(opt), "gain": round(float(1 - mu[opt] / mu[0]), 3), "delta": delta, "clear_pairedSE": round(clear, 2),
           "n_acc": a["n_acc"], "per_run_sd": round(float(np.median(V.std(1, ddof=1) / mu)), 3),
           "singles": {e8.lab(1 << k): round(float(mu[1 << k] / mu[0] - 1), 3) for k in range(e8.K)},
           "noise_free_ends": a["ends"], "noise_free_traps": a["n_traps"]}
    rng = np.random.default_rng(7); sim = {}
    for name, f in S.POL.items():
        nd = nb = 0; ends = {}
        for _ in range(n):
            m, p, u = (S.P_expert(V, rng, B, **EXPERT_KW) if name == "EXPERT" else f(V, rng, B)); assert u <= B
            ok = reg[m] <= delta; nd += ok; nb += ok and abs(p - mu[m]) / mu[m] <= S.PRED_TOL
            ends[e8.lab(m)] = ends.get(e8.lab(m), 0) + 1
        sim[name] = (round(nd / n, 3), round(nb / n, 3), max(ends, key=ends.get))
    out["sim_B96(pass_decision,pass_both,modal_end)"] = sim
    return out

if __name__ == "__main__":
    for i in map(int, sys.argv[1:]):
        print(json.dumps(run(i)))
