"""ScaleLab world function.  One vectorised mean function covers every card; a task's world is a
parameter dict (inactive cards at neutral values, see cards.py).  Noise is a deterministic function of
(instance salt, config, seed, metric) so re-running an identical config+seed returns the identical
number - exactly like a deterministic training stack - and averaging requires new seeds.

Config keys (arrays or scalars, broadcast together):
  N  non-embedding parameters          D   training tokens            B  batch size in tokens
  lr peak learning rate                wd  AdamW weight decay         wu warmup fraction of steps
  qk 0/1 qk-layernorm                  q   fraction of the corpus removed by the quality filter
  pool unique tokens sampled from the (filtered) corpus               b  fraction of 'books' in the mixture
  sub  raw-corpus subsample (unique tokens) taken before the quality filter
  f  checkpoint position as a fraction of the run (1 = final)         sched 0 cosine, 1 wsd, 2 constant
"""
import hashlib, json
import numpy as np

B_REF = 5.0e5          # reference batch (tokens) for the lr-batch law
INF = float("inf")

BASE = {  # every parameter the world function reads, at neutral / default values
    "E": 1.69, "A": 406.4, "alpha": 0.34, "Bc": 410.7, "beta": 0.28,
    "eta0": 3e-3, "gN": 0.0, "gD": 0.0, "gB": 0.0, "k_lo": 0.0, "k_hi": 0.0,
    "bc0": INF, "psi": 0.0,
    "h0": INF, "delta": 0.0, "w0": 0.02, "omega": 0.0, "Q": 1.0, "c_wu": 0.0,
    "tau0": 0.2, "chi": 0.0, "k_tau": 0.0,
    "Rs": INF, "U0": INF, "mu": 0.0, "nu": 1.0,
    "et": 0.02, "ct": 0.9,
    "phi": 0.0, "km": 0.0, "xi": 0.0, "gb": 0.0, "hb": 0.0, "L50": 2.9, "sa": 0.12, "vb": 0.0,
    "sigma0": 0.01, "rho": 0.3,
    "ca": 0.0, "zeta": 1.0, "rmin": 0.1, "cd": 0.2,
}
DEFAULT_CFG = {"B": B_REF, "lr": None, "wd": 0.1, "wu": 0.01, "qk": 0, "q": 0.0, "pool": INF, "b": 0.0,
               "f": 1.0, "sched": 0}


def full(p):
    out = dict(BASE); out.update(p); return out


def _a(x):
    return np.asarray(x, dtype=float)


def cfg_arrays(cfg):
    c = dict(DEFAULT_CFG); c.update({k: v for k, v in cfg.items() if v is not None or k not in c})
    return c


def eta_star(p, N, D, B=B_REF):
    return p["eta0"] * (_a(N) / 1e8) ** (-p["gN"]) * (_a(D) / 2e9) ** (-p["gD"]) * (_a(B) / B_REF) ** p["gB"]


def eta_max(p, N, wu, qk):
    return p["h0"] * (_a(N) / 1e8) ** (-p["delta"]) * (1 + _a(wu) / p["w0"]) ** p["omega"] * np.where(_a(qk) > 0, p["Q"], 1.0)


def tau_star(p, N, D):
    return p["tau0"] * ((_a(D) / _a(N)) / 20.0) ** p["chi"]


def b_crit(p, D):
    return p["bc0"] * (_a(D) / 2e9) ** p["psi"]


def effective_tokens(p, D, q=0.0, pool=INF, B=B_REF, D_horizon=None, sub=INF):
    """Deff after repetition (C6), quality (C7) and batch (C3).  D may be a checkpoint position;
    D_horizon (full run length) sets Bcrit, which Zhang et al. tie to the data horizon.
    Unique tokens U = min(pool, U0*(1-q), sub*(1-q)): `pool` caps the filtered pool directly, `sub` is a
    raw-corpus subsample taken *before* the quality filter (an ablation subset), U0 the full raw corpus."""
    D = _a(D); q = _a(q)
    U = np.minimum(np.minimum(_a(pool), p["U0"] * (1 - q)), _a(sub) * (1 - q))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        R = np.maximum(D / U - 1.0, 0.0)
        Rs = p["Rs"]
        if np.isinf(Rs):
            Dp = D
        else:
            Dp = np.where(D > U, U * (1.0 + Rs * (1.0 - np.exp(-R / Rs))), D)
    m = 1.0 + p["mu"] * q ** p["nu"]
    Dh = D if D_horizon is None else _a(D_horizon)
    return m * Dp / (1.0 + _a(B) / b_crit(p, Dh))


def core_loss(p, c):
    """Noise-free final validation loss (nats/token) before schedule, contamination or metric effects."""
    N = _a(c["N"]); Dfull = _a(c["D"]); f = _a(c.get("f", 1.0)); D = Dfull * f
    B = _a(c.get("B", B_REF))
    Deff = effective_tokens(p, D, c.get("q", 0.0), c.get("pool", INF), B, D_horizon=Dfull, sub=c.get("sub", INF))
    L = p["E"] + p["A"] * N ** (-p["alpha"]) + p["Bc"] * Deff ** (-p["beta"])
    lr = c.get("lr")
    if lr is not None and (p["k_lo"] > 0 or p["k_hi"] > 0):
        u = np.log(_a(lr) / eta_star(p, N, Dfull, B))
        L = L + np.where(u < 0, p["k_lo"], p["k_hi"]) * u ** 2
    if lr is not None and p["k_tau"] > 0:
        tau = B / (_a(lr) * _a(c.get("wd", 0.1)) * Dfull)
        v = np.clip(np.log(tau / tau_star(p, N, Dfull)), -3, 3)
        L = L + p["k_tau"] * v ** 2
    if p["c_wu"] > 0:
        L = L + p["c_wu"] * _a(c.get("wu", 0.01))
    return L


def sched_penalty(p, c):
    if p["ca"] <= 0:
        return 0.0
    f = _a(c.get("f", 1.0)); s = _a(c.get("sched", 0))
    cos = p["rmin"] + (1 - p["rmin"]) * 0.5 * (1 + np.cos(np.pi * f))
    wsd = np.where(f <= 1 - p["cd"], 1.0, (1 - f) / p["cd"])
    r = np.where(s == 0, cos, np.where(s == 1, wsd, 1.0))
    return p["ca"] * r ** p["zeta"]


def val_loss(p, c):
    """Web validation loss in nats/token (what `lab run` reports as `loss`)."""
    return core_loss(p, c) + sched_penalty(p, c) + p["vb"] * _a(c.get("b", 0.0)) ** 2


def qa_acc(p, c):
    """(clean accuracy, aggregate accuracy on the public QA benchmark)."""
    b = _a(c.get("b", 0.0)); L = core_loss(p, c)
    Lk = L - p["gb"] * b + p["hb"] * b ** 2
    a = 1.0 / (1.0 + np.exp(-(p["L50"] - Lk) / p["sa"]))
    expo = p["km"] * (_a(c["N"]) / 1e8) ** p["xi"] * b * _a(c["D"]) / 1e9
    mem = 1.0 - np.exp(-expo)
    agg = (1 - p["phi"]) * a + p["phi"] * (a + (1 - a) * mem)
    return a, agg


def answer_token_loss(p, c):
    return p["et"] + p["ct"] * (core_loss(p, c) - p["E"])


def exact_match(p, c, weights):
    """weights: {k: w_k} answer-length mixture."""
    pt = np.exp(-answer_token_loss(p, c))
    return sum(w * pt ** int(k) for k, w in weights.items())


def sigma(p, N):
    return p["sigma0"] * (_a(N) / 1e8) ** (-p["rho"])


def zdraw(salt, cfg, seed, metric):
    """Deterministic standard normal for (instance, config, seed, metric)."""
    key = json.dumps({"s": salt, "c": {k: (round(float(v), 12) if isinstance(v, (int, float, np.floating)) else v)
                                       for k, v in sorted(cfg.items())}, "seed": int(seed), "m": metric},
                     sort_keys=True)
    h = hashlib.sha256(key.encode()).digest()
    u1 = (int.from_bytes(h[:8], "big") + 0.5) / 2 ** 64
    u2 = (int.from_bytes(h[8:16], "big") + 0.5) / 2 ** 64
    return float(np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2))

