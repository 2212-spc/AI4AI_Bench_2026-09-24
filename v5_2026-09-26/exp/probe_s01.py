"""Numeric probe for the s01 (mix-p99 Jensen) blueprint design, run before the blueprint is written.

Checks, on many candidate draws:
  1. the constructive draw closes (rates, pool, menu, SLO windows all satisfiable),
  2. q1's truth / naive / M-M-1 / wrong-batch values and their separation,
  3. q2's interval width in tolerance units (G3 wants >= 4T),
  4. q3's option losses, the feasibility screen and the regret gap (G8 wants > 0),
  5. the tolerance T that `calibrate` will derive, by simulating the reference estimator,
  6. the discrimination B = min over rivals of |rival - truth| / T,
  7. rho: what the *free* notebook evidence alone buys, against T.

    python3 exp/probe_s01.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalelab.common import draw_card
from scalelab.labs import servelab as SV

BITS = 16
RHO_T = (0.25, 0.40, 0.55, 0.70, 0.80, 0.85)     # the six tenants' utilisations at the production batch
JIT = 0.010
GROWTH = 1.03                                    # disclosed next-quarter growth on every tenant
TOP_HI = 1.08                                    # the top tenant's committed peak, upper end
LN100 = math.log(100.0)
LN2 = math.log(2.0)

OR_IDX, OR_REPS, OR_DUR = (3, 4, 5), 4, 30.0     # oracle: the three highest-rho buckets, bought long
MENU_DUR = 5.0
NB_REPS, NB_DUR = 3, 5.0                         # the notebook's own load probe


def es(pf, seq, batch):
    return SV.service_time(pf, BITS, seq, batch)


def factor(k, rho):
    if rho >= 1.0:
        return None
    return 1.0 + k * rho / (1.0 - rho)


def mix_p99(pf, seq, batch, rates, k=None):
    k = 0.5 * (1.0 + pf["cs2"]) if k is None else k
    S = es(pf, seq, batch)
    acc = 0.0
    for r in rates:
        f = factor(k, r * S)
        if f is None:
            return None
        acc += f
    return 1e3 * LN100 * S * acc / len(rates)


def naive_mean_rate(pf, seq, batch, rates, k=None):
    k = 0.5 * (1.0 + pf["cs2"]) if k is None else k
    S = es(pf, seq, batch)
    f = factor(k, float(np.mean(rates)) * S)
    return None if f is None else 1e3 * LN100 * S * f


def b_fit(pf, seq, pool):
    return int(pool // (SV.kv_per_token(pf) * float(seq)))


FAIL_MS = 10000.0


def draw(rng, tries=400):
    why = "no draw attempted"
    for _ in range(tries):
        N = float(np.exp(rng.uniform(math.log(6.5e9), math.log(1.4e10))))
        p = dict(N=N, L=int(rng.integers(28, 49)), H_kv=8, d_head=128, M_gpu=80.0e9,
                 act_bytes=float(rng.uniform(1.4e9, 2.6e9)),
                 P_peak=float(rng.uniform(3.4e14, 4.6e14)), BW=float(rng.uniform(2.8e12, 4.2e12)))
        p.update(draw_card(rng, "S1", {"eff": (0.55, 0.78)}))
        p.update(draw_card(rng, "S5", {"mem_util": (0.84, 0.92), "kv_bytes": (1.9, 2.0)}))
        p.update(draw_card(rng, "S4", {"cs2": (1.6, 2.9)}))
        pf = SV.full(p)
        B0 = int(rng.choice([24, 32, 40]))
        menu = (B0 // 2, B0, B0 + 16, 3 * B0)
        seq = 128 * int(rng.integers(12, 21))
        if SV.batch_max(pf, BITS, seq) < menu[3] + 8:
            why = "dev box cannot run the largest menu batch at seq %d" % seq
            continue
        S0 = es(pf, seq, B0)
        rates = [round((t + float(rng.uniform(-JIT, JIT))) / S0, 2) for t in RHO_T]
        if len(set(rates)) < len(rates) or min(rates) <= 0:
            why = "tenant rates collide after rounding"
            continue
        rho = [r * S0 for r in rates]
        if not all(x < y for x, y in zip(rho, rho[1:])):
            why = "rounding broke the tenant ordering"
            continue
        if max(rho) >= 0.88 or min(rho) <= 0.20:
            why = "tenant utilisation window broken (max %.3f)" % max(rho)
            continue
        if es(pf, seq, menu[0]) * max(rates) <= 1.02:
            why = "batch %d is still stable for the top tenant" % menu[0]
            continue
        if es(pf, seq, menu[2]) * max(rates) >= 0.88:
            why = "batch %d does not relieve the top tenant enough" % menu[2]
            continue
        if TOP_HI * rates[-1] * S0 >= 0.93:
            why = "next quarter saturates the top tenant (rho %.3f)" % (TOP_HI * rates[-1] * S0)
            continue
        per = SV.kv_per_token(pf) * float(seq)
        lo_fit, hi_fit = menu[2] + 2, menu[3] - 6
        if lo_fit >= hi_fit:
            why = "no pool window"
            continue
        fit = int(rng.integers(lo_fit, hi_fit + 1))
        pool = per * (fit + float(rng.uniform(0.15, 0.85)))
        if b_fit(pf, seq, pool) != fit:
            why = "pool rounding"
            continue
        p.update(prod_seq=seq, prod_batch=B0, nb_batch=2 * B0, kv_pool=pool,
                 tenant_rates=rates, menu=menu,
                 top_rate_next=float(0.5 * (GROWTH + TOP_HI) * rates[-1]))
        pf = SV.full(p)
        truth = mix_p99(pf, seq, B0, rates)
        naive = naive_mean_rate(pf, seq, B0, rates)
        if truth is None or naive is None or truth / naive < 1.20:
            why = "Jensen gap too small (%.3f)" % (truth / naive if naive else float("nan"))
            continue
        best = mix_p99(pf, seq, menu[2], rates)
        slo = 50.0 * round(math.sqrt(truth * naive) / 50.0)
        if not (naive < 0.96 * slo and truth > 1.06 * slo and best < 0.90 * slo):
            why = "no SLO between the naive reading and the truth"
            continue
        p["slo_ms"] = slo
        return p, "ok"
    return None, why


def keys(p):
    pf = SV.full(p)
    seq, B0, rates, menu = p["prod_seq"], p["prod_batch"], p["tenant_rates"], p["menu"]
    out = {"q1": mix_p99(pf, seq, B0, rates)}
    nxt = [GROWTH * r for r in rates[:-1]]
    out["q2_lo"] = mix_p99(pf, seq, B0, nxt + [GROWTH * rates[-1]])
    out["q2_hi"] = mix_p99(pf, seq, B0, nxt + [TOP_HI * rates[-1]])
    loss, fit = {}, b_fit(pf, seq, p["kv_pool"])
    for b in menu:
        v = mix_p99(pf, seq, b, rates)
        loss["B%d" % b] = FAIL_MS if (v is None or b > fit) else v
    out["q3"] = loss
    out["_k"], out["_fit"] = 0.5 * (1.0 + pf["cs2"]), fit
    return out


def rivals(p):
    pf = SV.full(p)
    seq, B0, rates = p["prod_seq"], p["prod_batch"], p["tenant_rates"]
    nb = p["nb_batch"]
    out = {}
    out["skip:service"] = mix_p99(pf, seq, nb, [r * es(pf, seq, B0) / es(pf, seq, nb) for r in rates])
    out["skip:service2"] = mix_p99(pf, seq, nb, rates)
    out["skip:queue"] = mix_p99(pf, seq, B0, rates, k=1.0)
    out["skip:wait"] = 1e3 * LN100 * es(pf, seq, B0)
    out["skip:aggregate"] = naive_mean_rate(pf, seq, B0, rates)
    out["B_prior"] = naive_mean_rate(pf, seq, nb, rates)
    out["skip:quantile"] = mix_p99(pf, seq, B0, rates) * LN2 / LN100
    return {a: b for a, b in out.items() if b is not None}


# --------------------------------------------------------------------- the reference estimator (kappa = 1)
def est_k(pf, seq, batch, rates, idx, reps, dur, rng):
    """Inverse-variance average of per-row k = (E[T]/E[S] - 1) / (rho/(1-rho)).

    util is exact, so E[S] and every rho come for free; p50 and p99 are independent draws on the same row,
    so their geometric mean carries sig_lat/sqrt(2).  k enters E[T] linearly: the estimator is an average.
    """
    S = es(pf, seq, batch)
    sig = pf["sig_lat"] / math.sqrt(dur / SV.DUR_REF) / math.sqrt(2.0)
    k = 0.5 * (1.0 + pf["cs2"])
    num = den = 0.0
    for i in idx:
        rho = rates[i] * S
        x = rho / (1.0 - rho)
        f = 1.0 + k * x
        for _ in range(reps):
            k_row = (f * math.exp(sig * float(rng.normal())) - 1.0) / x
            var = (sig * f / x) ** 2
            num += k_row / var
            den += 1.0 / var
    return num / den


def tol_of(p, n=400, seed=7):
    rng = np.random.default_rng(seed)
    pf = SV.full(p)
    seq, B0, rates = p["prod_seq"], p["prod_batch"], p["tenant_rates"]
    truth = mix_p99(pf, seq, B0, rates)
    err = [abs(mix_p99(pf, seq, B0, rates, k=est_k(pf, seq, B0, rates, OR_IDX, OR_REPS, OR_DUR, rng)) - truth)
           for _ in range(n)]
    return 2.25 * float(np.percentile(err, 90)), truth


def free_err(p, n=400, seed=9):
    """What the notebook's own three load rows buy, if the agent derives everything else correctly."""
    rng = np.random.default_rng(seed)
    pf = SV.full(p)
    seq, B0, rates, nb = p["prod_seq"], p["prod_batch"], p["tenant_rates"], p["nb_batch"]
    truth = mix_p99(pf, seq, B0, rates)
    mean_r = [float(np.mean(rates))]
    err = [abs(mix_p99(pf, seq, B0, rates,
                       k=est_k(pf, seq, nb, mean_r, (0,), NB_REPS, NB_DUR, rng)) - truth)
           for _ in range(n)]
    return 2.25 * float(np.percentile(err, 90))


def main():
    rows = []
    n_ok = 0
    for ws in range(12):
        rng = np.random.default_rng(1000 + ws)
        p, why = draw(rng)
        if p is None:
            print("ws=%-2d  DRAW FAILED: %s" % (ws, why))
            continue
        n_ok += 1
        pf = SV.full(p)
        K, R = keys(p), rivals(p)
        T1, truth = tol_of(p)
        T2 = T1 * K["q2_lo"] / truth
        width = K["q2_hi"] - K["q2_lo"]
        regs = sorted(v - min(K["q3"].values()) for v in K["q3"].values())
        B = min(abs(v - truth) / T1 for v in R.values())
        nearest = min(R, key=lambda a: abs(R[a] - truth))
        fe = free_err(p)
        rows.append((ws, B, width / T2, regs[1], fe / T1))
        print("ws=%-2d N=%.2fe9 L=%d seq=%d B0=%d menu=%s cs2=%.3f k=%.3f fit=%d"
              % (ws, p["N"] / 1e9, p["L"], p["prod_seq"], p["prod_batch"], p["menu"],
                 pf["cs2"], K["_k"], K["_fit"]))
        print("      E[S] ms by menu batch: %s"
              % ["%.2f" % (1e3 * es(pf, p["prod_seq"], b)) for b in p["menu"]])
        print("      rho at production: %s  (next-quarter top %.3f)"
              % (["%.3f" % (r * es(pf, p["prod_seq"], p["prod_batch"])) for r in p["tenant_rates"]],
                 TOP_HI * p["tenant_rates"][-1] * es(pf, p["prod_seq"], p["prod_batch"])))
        print("      q1 truth %.1f ms   T %.2f ms (%.2f%%)   SLO %.0f   B = %.1f  (nearest %s = %.1f)"
              % (truth, T1, 100 * T1 / truth, p["slo_ms"], B, nearest, R[nearest]))
        print("      q2 [%.1f, %.1f]  width %.1f ms = %.1f T" % (K["q2_lo"], K["q2_hi"], width, width / T2))
        print("      q3 losses %s  ->  best %s, gap %.1f ms"
              % ({b: ("FAIL" if v >= FAIL_MS else round(v, 1)) for b, v in K["q3"].items()},
                 min(K["q3"], key=K["q3"].get), regs[1]))
        print("      free notebook route: err %.2f%% of truth = %.2f T   (>1 means runs are needed)"
              % (100 * fe / truth, fe / T1))
        print("      over_T: %s" % {a: round(abs(v - truth) / T1, 1) for a, v in sorted(R.items())})
    if rows:
        print("\nsummary over %d draws: B min %.1f median %.1f | q2 width min %.1f T | q3 gap min %.0f ms"
              " | free-route min %.2f T" % (len(rows), min(r[1] for r in rows),
                                            float(np.median([r[1] for r in rows])),
                                            min(r[2] for r in rows), min(r[3] for r in rows),
                                            min(r[4] for r in rows)))
    print("draws ok: %d/12" % n_ok)


if __name__ == "__main__":
    main()
