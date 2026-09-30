"""E8 probe: 8-change worlds (E1's six + c7 longer warm-up [substitute for clipping] + c8 output-layer lr multiplier
[per-layer lr, couples with width/lr]). Question: does a richer mechanism library let adversarial search close the
backward-elimination escape hatch while an adaptive blind expert still solves the world within budget?"""
import sys, json, time
sys.path.insert(0, "/tmp/bench/gen"); sys.path.insert(0, "/tmp/bench/lab")
import numpy as np
import e1_world as W
import engine as E

K = 8; NC = 1 << K
NAMES = [f"c{i+1}" for i in range(K)]


def sample_params(i):
    r = np.random.default_rng([8484, i])
    ch = lambda xs: xs[int(r.integers(len(xs)))]
    return dict(lr=ch([0.015, 0.02, 0.03, 0.04]), lr_f=ch([1.5, 2.0, 2.5, 3.0]), mom=0.9, mom2=ch([0.95, 0.97]),
                warm=10, clip=ch([0.3, 0.5, 1.0, 2.0]), hid2=ch([48, 160, 256]), wd2=ch([1e-4, 3e-4, 1e-3, 3e-3]),
                warm2=ch([40, 80, 120]), out_f=ch([0.25, 0.5, 2.0]),
                tseed=int(r.integers(1, 10**6)), dseed=int(r.integers(1, 10**6)))


def sample_params_bal(i):
    """position-balanced prior: every proposal can point the wrong way (real 'bad ideas' are common), so the optimum's
    size is not concentrated near the full set (removes the base-rate advantage of starting from all-on)."""
    r = np.random.default_rng([8585, i])
    ch = lambda xs: xs[int(r.integers(len(xs)))]
    return dict(lr=ch([0.015, 0.02, 0.03, 0.04]), lr_f=ch([0.5, 1.5, 2.0, 3.0]), mom=0.9, mom2=ch([0.8, 0.95, 0.97, 0.99]),
                warm=10, clip=ch([0.1, 0.3, 1.0, 2.0]), hid2=ch([32, 48, 160, 256]), wd2=ch([1e-4, 1e-3, 3e-3, 1e-2]),
                warm2=ch([1, 40, 120, 200]), out_f=ch([0.25, 0.5, 2.0, 4.0]),
                tseed=int(r.integers(1, 10**6)), dseed=int(r.integers(1, 10**6)))

import os as _os
if _os.environ.get("E8_PRIOR") == "bal":
    sample_params = sample_params_bal


def build(params):
    base = dict(hidden=96, lr=params["lr"], momentum=params["mom"], weight_decay=0.0, warmup_steps=params["warm"], steps=400,
                min_lr_ratio=0.05, grad_accum_steps=1, micro_batch_size=32, clip_norm=None)
    deltas = {"c1": {"lr": params["lr"] * params["lr_f"]}, "c2": {"momentum": params["mom2"]}, "c3": {"clip_norm": params["clip"]},
              "c4": {"hidden": params["hid2"]}, "c5": {"weight_decay": params["wd2"]}, "c6": {},
              "c7": {"warmup_steps": params["warm2"]}, "c8": {"flag_lr_mult_out": params["out_f"]}}
    w = W.make_world(params["tseed"], base, deltas, dict(d_in=32, t_hidden=64, noise=0.1), params["dseed"])
    return w


def run_cell(world, mask, seed):
    cfg = dict(world["base"]); sep = False
    for i, nm in enumerate(NAMES):
        if mask >> i & 1:
            if nm == "c6": sep = True
            else: cfg.update(world["deltas"][nm])
    xtr, ytr, xva, yva = world["data"]
    cfg, flags = W._split(cfg)
    p = E.train(cfg, xtr, ytr, seed, sep_rng=sep, **flags)
    vt = float(np.var(yva))
    if p is None: return vt
    v = E.mse(p, xva, yva)
    return vt if (not np.isfinite(v) or v > vt) else v


def lab(m): return "+".join(n for i, n in enumerate(NAMES) if m >> i & 1) or "base"
def mask_of(s): return 0 if s == "base" else sum(1 << NAMES.index(n) for n in s.split("+"))


# ---------------- noise-free mimics on a mean table mu[256] ----------------
def hill(mu, cur, moves):
    while True:
        cands = moves(cur)
        if not cands: return cur
        c = min(cands, key=lambda x: mu[x])
        if mu[c] >= mu[cur]: return cur
        cur = c

singles = lambda m: [m ^ (1 << i) for i in range(K)]
adds = lambda m: [m | 1 << i for i in range(K) if not m >> i & 1]
rems = lambda m: [m & ~(1 << i) for i in range(K) if m >> i & 1]
PAIRS = [(1 << i) | (1 << j) for i in range(K) for j in range(i + 1, K)]
adds2 = lambda m: adds(m) + [m | p for p in PAIRS if not m & p]
rems2 = lambda m: rems(m) + [m & ~p for p in PAIRS if m & p == p]


def mimics(mu):
    full = NC - 1
    ofat = sum(1 << i for i in range(K) if mu[1 << i] < mu[0])
    tb_start = min([1 << i for i in range(K)] + [3, 7], key=lambda x: mu[x])
    ph_start = min([1 << i for i in range(K)] + PAIRS, key=lambda x: mu[x])
    return {"ofat": ofat, "fwd": hill(mu, 0, adds), "bwd": hill(mu, full, rems), "textbook": hill(mu, tb_start, singles),
            "pairhill": hill(mu, ph_start, singles), "fwd2": hill(mu, 0, adds2), "bwd2": hill(mu, full, rems2),
            "flip_from_all": hill(mu, full, singles)}


def analyse(mu, delta=0.025):
    opt = int(np.argmin(mu)); reg = mu / mu[opt] - 1
    acc = set(int(m) for m in np.where(reg <= delta)[0])
    mm = mimics(mu)
    traps = {k: (v not in acc) for k, v in mm.items()}
    lo = int(sum(all(mu[m] <= mu[x] for x in singles(m)) for m in range(NC)))
    return {"opt": lab(opt), "opt_gain": float(1 - mu[opt] / mu[0]), "n_acc": len(acc),
            "ends": {k: lab(v) for k, v in mm.items()}, "end_regret": {k: round(float(reg[v]), 4) for k, v in mm.items()},
            "traps": traps, "n_traps": int(sum(traps.values())), "local_optima": lo}
