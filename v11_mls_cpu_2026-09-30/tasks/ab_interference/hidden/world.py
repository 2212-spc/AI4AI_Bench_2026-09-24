"""Hidden world for ab_interference: a two-sided marketplace experiment with a market-level
saturation (two-stage randomised) design.  The agent never sees this file.

Market m has n_m sellers, a demand level L_m (persistent, visible through the pre-period) and a
post-period shock g_m (not predictable from the pre-period).  Seller i has a persistent quality a_i.
The treatment multiplies a seller's attractiveness by e^beta.  Buyers split the market's demand
across sellers in proportion to attractiveness ("share stealing", strength s in [0,1]) and the
market's total demand responds to the treated fraction p through an expansion term
E_m(p) = gamma_m * p + kappa * p * (1 - p)   (E_m(0) = 0, E_m(1) = gamma_m).

    Y_i = L_m g_m n_m (a_i / sum_j a_j) * e^{beta T_i} / ((1 - s) + s r_m) * (1 + E_m(p_m)) * noise_i
    r_m = sum_j a_j e^{beta T_j} / sum_j a_j        (realised mean attractiveness multiplier)

s = 0 and gamma = kappa = 0 is the no-interference world (naive diff-in-means is right).
gamma_m may depend on market size (gamma_m = gamma0 + gamma_size * log(n_m / median n)).

Estimand (finite population, per seller):
    GTE = (1/N) sum_m L_m g_m n_m [ e^beta / ((1 - s) + s e^beta) * (1 + gamma_m) - 1 ]
Scale: B = (1/N) sum_m L_m g_m n_m  (mean seller outcome with nobody treated).
"""
import numpy as np
import pandas as pd

GRID = np.round(np.arange(1, 10) / 10, 1)
DESIGNS = {                      # probability of each saturation level in GRID
    "uniform": np.ones(9),
    "cautious": (1.1 - GRID) ** 1.5,          # most markets get low treated fractions
    "coarse": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], float),
    "bold": GRID ** 1.5,                    # most markets get high treated fractions
}


def make_experiment(P, seed):
    rng = np.random.default_rng(seed)
    M = P["M"]
    n = np.clip(np.round(np.exp(rng.normal(np.log(P["n_med"]), P["n_sig"], M))), 15, 2000).astype(int)
    L = 100.0 * np.exp(rng.normal(0, P["L_sig"], M))                  # demand level per seller
    g = np.exp(rng.normal(0, P["g_sig"], M) - P["g_sig"] ** 2 / 2)    # post-period market shock
    w = DESIGNS[P["design"]]; w = w / w.sum()
    p = rng.choice(GRID, size=M, p=w)
    gam = P["gamma0"] + P.get("gamma_size", 0.0) * np.log(n / np.median(n))
    beta, s, kap = P["beta"], P["s"], P["kappa"]
    eb = np.exp(beta)
    rows, gte_num, base_num = [], 0.0, 0.0
    for m in range(M):
        nm = n[m]
        a = np.exp(rng.normal(0, P["a_sig"], nm))
        share = a / a.sum()
        T = np.zeros(nm, int); T[rng.permutation(nm)[: int(round(p[m] * nm))]] = 1
        r = np.sum(share * np.exp(beta * T))
        E = gam[m] * p[m] + kap * p[m] * (1 - p[m])
        mu = L[m] * g[m] * nm * share * np.exp(beta * T) / ((1 - s) + s * r) * (1 + E)
        y = mu * np.exp(rng.normal(0, P["eps"], nm) - P["eps"] ** 2 / 2)
        pre = L[m] * nm * share * np.exp(rng.normal(0, P["eps_pre"], nm) - P["eps_pre"] ** 2 / 2)
        rows.append(pd.DataFrame(dict(market=m, p_market=p[m], treated=T, pre_outcome=pre, outcome=y)))
        gte_num += L[m] * g[m] * nm * (eb / ((1 - s) + s * eb) * (1 + gam[m]) - 1.0)
        base_num += L[m] * g[m] * nm
    df = pd.concat(rows, ignore_index=True)
    perm = rng.permutation(M)                          # market ids carry no information
    df["market"] = perm[df["market"].values]
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    df["pre_outcome"] = df["pre_outcome"].round(3); df["outcome"] = df["outcome"].round(3)
    N = int(n.sum())
    return df, gte_num / N, base_num / N


BASE = dict(M=100, n_med=150, n_sig=0.4, L_sig=0.6, g_sig=0.012, a_sig=0.5, eps=0.12, eps_pre=0.12,
            beta=0.2, s=0.9, gamma0=0.0, gamma_size=0.0, kappa=0.0, design="uniform")


def P_(**kw):
    d = dict(BASE); d.update(kw); return d


# name, params, base seed, hidden
SETTINGS = [
    # dev: strong share stealing, total demand slightly shrinks -> naive has the wrong sign
    ("dev_steal", P_(beta=0.20, s=0.9, gamma0=-0.06, kappa=0.10), 1000, False),
    # dev: positive spillover (market grows, concave in p), mild stealing
    ("dev_grow", P_(M=80, beta=0.10, s=0.6, gamma0=0.12, kappa=0.30, g_sig=0.015), 2000, False),
    # hidden: no interference at all -> naive is right; estimator must not over-correct
    ("hid_none", P_(M=70, beta=0.12, s=0.0, gamma0=0.0, kappa=0.0, g_sig=0.015, L_sig=0.8), 3000, True),
    # hidden: pure zero-sum stealing, congestion worse in big markets, very unequal market sizes
    ("hid_zero_sum", P_(M=150, n_sig=0.9, n_med=70, beta=0.30, s=1.0, gamma0=-0.02, gamma_size=-0.09,
                         kappa=0.0), 4000, True),
    # hidden: cautious rollout design (mostly low p), strong concave network effect
    ("hid_cautious", P_(M=100, beta=0.15, s=0.5, gamma0=0.20, kappa=0.40, design="cautious"), 5000, True),
    # hidden: few markets, coarse design, noisier markets, stealing + mild growth
    ("hid_small", P_(M=50, n_med=120, beta=0.25, s=0.8, gamma0=0.05, kappa=-0.15, g_sig=0.012,
                      L_sig=0.9, design="coarse"), 6000, True),
]
N_REP = 8


def build_setting(row):
    name, P, seed, hidden = row
    reps = [make_experiment(P, seed + k) for k in range(N_REP)]
    return dict(name=name, hidden=hidden, units=[r[0] for r in reps], gte=[r[1] for r in reps],
                base=[r[2] for r in reps])
