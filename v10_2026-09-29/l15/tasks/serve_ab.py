"""SERVE-AB (level L1): roll out a slower, better model onto a shared, autoscaled GPU pool.

The situation.  Chat traffic is served by ONE pool of GPU replicas behind ONE FIFO queue.  Model A serves
everything today.  Model B (a reasoning checkpoint) wins the offline eval and the product team wants it
launched.  B generates longer answers: every B request holds a replica about kappa ~ 2-2.6x as long.  The
pool is autoscaled per hour to a utilisation target, within a fixed GPU quota.  Users who wait too long in
the queue abandon.  The deliverable is a launch SCHEDULE (fraction of traffic routed to B for each hour of
the day) plus two numeric answers.

Difficulty mechanism (the family's signature: standard experimental hygiene yields a complete,
self-consistent, WRONG launch that the agent's own checks confirm):

  P1  THE A/B CANNOT SEE THE COST.  Both arms share the queue, so every per-arm difference (rating, score,
      latency) nets out the queueing cost that B imposes on everybody: the randomised comparison estimates
      the quality effect only.  (Interference / SUTVA violation through a shared resource.)
  P2  THE GUARDRAILED TEST CANNOT SEE IT EITHER.  Experiments may route at most 30% of peak-hour traffic to an
      unlaunched model.  At 30% the autoscaler still has quota headroom and simply adds replicas: queue wait
      and abandonment stay flat, so a switchback / global-holdout comparison (the textbook fix for
      interference) also reads "no cost".  At 100% the peak hours need more replicas than the quota: the
      pool saturates and the queue overflows (utilisation > 1 at the quota -> abandonment).
  P3  OVERLOAD IS PRICED BY PATIENCE, WHICH NORMAL TRAFFIC NEVER REVEALS.  How much an overloaded hour costs
      depends on how long users wait before abandoning.  In history and in any guardrailed test the pool is
      never saturated, so patience is unidentified there.  The lab offers a capacity drill (cap the replica
      count during off-peak hours) - an experiment the agent has to think of and design.
      CORRECTION (post-run, 2026-09-29; documentation only, no behaviour change): "unidentified" is WRONG.
      In M/M/c+M abandonment acts at rate theta on waiting requests, so abandon_rate = theta * E[wait]
      holds exactly in unsaturated hours too (Little's law for abandonment); theta is identifiable from
      history whenever abandonment and wait are both nonzero.  The drill is a higher-SNR path, not the only
      one.  GPT-6 ran 0 drills in 7/8 serve_ab runs and its Q1 interval covered the truth in 8/8.
      See V10_REPORT.md section 6.5.
  P4  THE LAUNCH FRACTION AT PEAK IS A TRADE-OFF, NOT A RULE.  "Stay under the autoscaler target at the
      quota" is too conservative, "fill to the quota" too aggressive; the optimum balances the quality gain
      against Erlang-A queueing at the quota and moves with the instance.

Ground truth is exact: per hour the pool is an M/M/c+M queue (Erlang-A) at its stationary distribution,
computed by a truncated birth-death chain (tail mass < 1e-12), with c set by the autoscaler rule
c = clip(ceil(offered_load / U_TGT), CMIN, quota).  Day-to-day load varies (multiplicative Gaussian noise,
integrated with Gauss-Hermite quadrature); the schedule value is the traffic-weighted mean change in the
per-request score, exactly separable by hour, maximised per hour on a fine grid.

Per-request score (the product metric, stated in the docs): rating in [0,1] if served, 0 if abandoned,
minus ETA per second spent in the queue.
"""
import csv, io, json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

LEVEL = "L1"
BUDGET = 24.0              # experiment-days
U_TGT = 0.75               # autoscaler utilisation target (public)
CMIN = 4                   # autoscaler floor (public)
GUARD_LO, GUARD_HI = 8, 21 # guardrail window, hours 08:00-21:59 (public)
GUARD_CAP = 0.30           # max fraction of traffic an experiment may send to B inside the window (public)
DRILL_LO, DRILL_HI = 1, 5  # capacity drills only 01:00-05:59 (public)
MAX_DAYS = 14
HIST_DAYS = 14
RATING_SD = 0.35           # per-request rating sd (served requests)
QNOISE = 0.12              # hour-level multiplicative noise of queue quantities (bursty arrivals)
EPS_R = 0.0010             # R1: schedule regret tolerance (score units)
W1_CAP = 0.0060            # R2: width cap, full-rollout interval
W2_CAP = 0.0040            # R3: width cap, own-schedule interval
K_DECOY = 2.5
BIZ_LO, BIZ_HI = 9, 17     # business hours (L2 heterogeneity lives here; unused at L1)
H = np.arange(24)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(9)
_GH_W = _GH_W / _GH_W.sum()


def _sig(x, k=4):
    return float("%.*g" % (k, x))


# ------------------------------------------------------------------ queueing core
def erlang_a(lam, s, c, theta):
    """M/M/c+M stationary: (P_abandon, E[queue wait] over all arrivals, P_wait, tail_mass)."""
    mu = 1.0 / s
    a = lam * s
    over = max(0.0, 1.0 - c / a) if a > 0 else 0.0
    nq = 3.0 * (lam / theta) * max(0.05, over) + 12.0 * math.sqrt(lam / theta) + 60
    nmax = int(c + nq)
    n = np.arange(nmax + 1, dtype=float)
    death = np.minimum(n, c) * mu + np.maximum(n - c, 0.0) * theta
    lr = np.zeros(nmax + 1)
    lr[1:] = np.cumsum(math.log(lam) - np.log(death[1:]))
    lr -= lr.max()
    pi = np.exp(lr)
    pi /= pi.sum()
    EQ = float((np.maximum(n - c, 0.0) * pi).sum())
    return theta * EQ / lam, EQ / lam, float(pi[c:].sum()), float(pi[-1])


def replicas(lam, s, Q, cap=None):
    need = max(CMIN, int(math.ceil(lam * s / U_TGT - 1e-9)))
    c = min(Q, need)
    if cap is not None:
        c = min(c, int(cap))
    return c


def kap(p, h):
    if "kap_h" in p:                      # per-hour form (used by strategies' fitted models)
        return p["kap_h"][h]
    return p["kap_biz"] if BIZ_LO <= h <= BIZ_HI else p["kap_off"]


def dq(p, h):
    if "dq_h" in p:
        return p["dq_h"][h]
    return p["dq_biz"] if BIZ_LO <= h <= BIZ_HI else p["dq_off"]


def sbar(p, h, x):
    return p["sA"] * (1.0 + x * (kap(p, h) - 1.0))


def hour_state(p, h, x, lam, cap=None):
    s = sbar(p, h, x)
    c = replicas(lam, s, p["Q"], cap)
    Pab, EW, Pw, _ = erlang_a(lam, s, c, p["theta"])
    return {"c": c, "s": s, "Pab": Pab, "EW": EW, "Pw": Pw, "util": lam * (1 - Pab) * s / c}


def hour_score(p, h, x, lam, cap=None):
    st = hour_state(p, h, x, lam, cap)
    return (1.0 - st["Pab"]) * (p["qA"] + x * dq(p, h)) - p["eta"] * st["EW"]


def day_types(p):
    """(weight, multiplier) of the weekly day types; multiplicative day noise is integrated separately."""
    if p.get("w_we", 1.0) == 1.0:
        return [(1.0, 1.0)]
    return [(5.0 / 7, 1.0), (2.0 / 7, p["w_we"])]


def _nodes(p):
    out = []
    for wt, m in day_types(p):
        for xg, wg in zip(_GH_X, _GH_W):
            out.append((wt * wg, m * (1.0 + p["sd_day"] * xg)))
    return out


_CACHE = {}


def _key(p):
    return json.dumps(p, sort_keys=True, default=float)


def hour_value(p, h, x):
    """(E[lam * (score(x) - score(0))], E[lam]) for hour h, over day types and day noise."""
    num_, den = 0.0, 0.0
    for wt, m in _nodes(p):
        lam = p["lam"][h] * m
        num_ += wt * lam * (hour_score(p, h, x, lam) - hour_score(p, h, 0.0, lam))
        den += wt * lam
    return num_, den


def daily_delta(p, xs):
    nums, dens = 0.0, 0.0
    for h in H:
        a, b = hour_value(p, int(h), float(xs[h]))
        nums += a
        dens += b
    return nums / dens


def optimum(p):
    k = _key(p)
    if k in _CACHE:
        return _CACHE[k]
    xs, vals = [], []
    den = sum(hour_value(p, int(h), 0.0)[1] for h in H)
    for h in H:
        h = int(h)
        g1 = np.linspace(0, 1, 51)
        v1 = [hour_value(p, h, float(x))[0] for x in g1]
        i = int(np.argmax(v1))
        lo, hi = max(0.0, g1[i] - 0.03), min(1.0, g1[i] + 0.03)
        g2 = np.linspace(lo, hi, 61)
        v2 = [hour_value(p, h, float(x))[0] for x in g2]
        j = int(np.argmax(v2))
        if v2[j] >= v1[i]:
            xs.append(float(g2[j])); vals.append(v2[j])
        else:
            xs.append(float(g1[i])); vals.append(v1[i])
    res = (np.array(xs), float(sum(vals) / den))
    _CACHE[k] = res
    return res


# ------------------------------------------------------------------ the schedules a decoy would ship
def _x_max(p, h, pred, lam=None):
    """largest x in [0,1] (grid 0.002) with pred(x) true at the weekday-mean load of hour h."""
    lam = p["lam"][h] if lam is None else lam
    best = 0.0
    for x in np.linspace(0, 1, 501):
        if pred(float(x), lam):
            best = float(x)
    return best


def sched_keep_target(p):
    """P4 decoy: never let the autoscaler's replica demand exceed the quota (stay at target utilisation)."""
    return np.array([_x_max(p, h, lambda x, lam, h=h: lam * sbar(p, h, x) / U_TGT <= p["Q"]) for h in H])


def sched_rho(p, r):
    """rule of thumb: utilisation at the quota <= r."""
    return np.array([_x_max(p, h, lambda x, lam, h=h: lam * sbar(p, h, x) <= r * p["Q"]
                            or lam * sbar(p, h, x) / U_TGT <= p["Q"]) for h in H])


def sched_guard(p):
    return np.array([GUARD_CAP if GUARD_LO <= h <= GUARD_HI else 1.0 for h in H])


def naive_delta(p):
    """What the per-arm A/B comparison measures, scaled to 100%: traffic-weighted quality gain."""
    w = np.array(p["lam"])
    return float(sum(w[h] * dq(p, h) for h in H) / w.sum())


def fluid_delta(p, xs):
    """P3 decoy: Erlang-C below saturation, fluid loss (1 - 1/rho) with zero queue cost above it."""
    nums, dens = 0.0, 0.0
    for h in H:
        h = int(h)
        for wt, m in _nodes(p):
            lam = p["lam"][h] * m
            vals = []
            for x in (float(xs[h]), 0.0):
                s = sbar(p, h, x)
                c = replicas(lam, s, p["Q"])
                a = lam * s
                if a < c:
                    B = 1.0
                    for k in range(1, c + 1):
                        B = a * B / (k + a * B)
                    C = c * B / (c - a * (1 - B))
                    EW, Pab = C * s / (c - a), 0.0
                else:
                    EW, Pab = 0.0, 1.0 - c / a
                vals.append((1 - Pab) * (p["qA"] + x * dq(p, h)) - p["eta"] * EW)
            nums += wt * lam * (vals[0] - vals[1])
            dens += wt * lam
    return nums / dens


def erlang_b(c, a):
    B = 1.0
    for k in range(1, int(c) + 1):
        B = a * B / (k + a * B)
    return B


def loss_delta(p, xs):
    """P3 decoy model: no patience data, so a request that finds no free replica is treated as lost at once
    (Erlang-B loss system, the zero-patience limit): no queue, no waiting cost.  Same autoscaler, same day
    noise, same everything else."""
    nums, dens = 0.0, 0.0
    for h in H:
        h = int(h)
        for wt, m in _nodes(p):
            lam = p["lam"][h] * m
            vals = []
            for x in (float(xs[h]), 0.0):
                s = sbar(p, h, x)
                c = replicas(lam, s, p["Q"])
                vals.append((1 - erlang_b(c, lam * s)) * (p["qA"] + x * dq(p, h)))
            nums += wt * lam * (vals[0] - vals[1])
            dens += wt * lam
    return nums / dens


def truth(p):
    k = "T" + _key(p)
    if k in _CACHE:
        return _CACHE[k]
    xs, Ds = optimum(p)
    ones = np.ones(24)
    Dfull = daily_delta(p, ones)
    reg = {"ship_all": Ds - Dfull,
           "guard_safe": Ds - daily_delta(p, sched_guard(p)),
           "keep_target": Ds - daily_delta(p, sched_keep_target(p)),
           "no_ship": Ds}
    for r in (0.85, 0.90, 0.95):
        reg["rho_%.2f" % r] = Ds - daily_delta(p, sched_rho(p, r))
    pk = int(np.argmax(p["lam"]))
    stpk = hour_state(p, pk, 1.0, p["lam"][pk])
    t = {"level": p["level"], "D_star": round(Ds, 6), "x_star": [round(float(x), 3) for x in xs],
         "D_full": round(Dfull, 6), "naive_ab_delta": round(naive_delta(p), 6),
         "fluid_full": round(fluid_delta(p, ones), 6), "loss_full": round(loss_delta(p, ones), 6),
         "decoy_regret": {k_: round(v, 6) for k_, v in reg.items()},
         "peak_hour": pk, "peak_full_util_at_quota": round(p["lam"][pk] * sbar(p, pk, 1.0) / p["Q"], 3),
         "peak_full_abandon": round(stpk["Pab"], 4), "peak_full_wait_s": round(stpk["EW"], 2)}
    _CACHE[k] = t
    return t


# ------------------------------------------------------------------ instance construction
def _profile(g):
    amp = g.uniform(0.40, 0.55)
    hpk = g.uniform(13.0, 15.5)
    pr = 1 + amp * np.cos(2 * np.pi * (H + 0.5 - hpk) / 24) + 0.10 * np.cos(4 * np.pi * (H + 0.5 - hpk - 1.5) / 24)
    return pr / pr.max(), round(float(hpk), 2)


def _draw(g, level):
    pr, hpk = _profile(g)
    p = {"level": level, "Q": int(g.choice([16, 20, 24, 32])), "sA": round(float(g.uniform(1.6, 3.6)), 3),
         "qA": round(float(g.uniform(0.62, 0.74)), 4), "eta": round(float(g.uniform(0.006, 0.012)), 4),
         "theta": round(1.0 / float(g.uniform(20.0, 90.0)), 5), "sd_day": 0.03, "w_we": 1.0,
         "dow0": int(g.integers(7)), "hist_seed": int(g.integers(1 << 30)), "hpk": hpk}
    k = round(float(g.uniform(2.0, 2.6)), 3)
    d = round(float(g.uniform(0.020, 0.050)), 4)
    p.update({"kap_off": k, "kap_biz": k, "dq_off": d, "dq_biz": d})
    if level == "L2":
        p["kap_off"] = round(float(g.uniform(1.6, 2.0)), 3)
        p["kap_biz"] = round(p["kap_off"] + float(g.uniform(0.45, 0.7)), 3)
        p["dq_off"] = round(float(g.uniform(0.015, 0.030)), 4)
        p["dq_biz"] = round(p["dq_off"] + float(g.uniform(0.010, 0.025)), 4)
        p["w_we"] = round(float(g.uniform(0.72, 0.84)), 3)
        p["sd_day"] = 0.05
    # peak load: offered load of 100% B at the peak hour, as a multiple of the quota
    kpk = max(kap(p, h) for h in range(GUARD_LO, GUARD_HI + 1))
    if level == "L0":
        X = float(g.uniform(0.50, 0.64))
    else:
        # masking needs X * (1 + GUARD_CAP (kappa - 1)) / kappa * (1 + sd) * week <= U_TGT
        wk = 1.0
        kg = min(kap(p, h) for h in range(GUARD_LO, GUARD_HI + 1))
        xmax = U_TGT * kpk / ((1 + GUARD_CAP * (kpk - 1)) * (1 + 1.2 * p["sd_day"]) * wk)
        X = float(g.uniform(1.06, max(1.07, min(1.28, 0.99 * xmax))))
    lam_pk = X * p["Q"] / (p["sA"] * kpk)
    p["lam"] = [round(float(lam_pk * v), 5) for v in pr]
    return p


def _conditions(p):
    """Closed-form instance conditions (no lab calls).  Returns (list of bools, info)."""
    t = truth(p)
    info = {}
    lam = np.array(p["lam"])
    Q = p["Q"]
    conds = {}
    hi_load = lambda h, x: lam[h] * (1 + 2.5 * p["sd_day"]) * sbar(p, h, x)
    if p["level"] == "L0":
        # nothing saturates even at 100% B on a +2.5 sd day: the plain A/B answer is right
        conds["no_binding"] = all(hi_load(h, 1.0) / U_TGT <= Q for h in H)
        conds["ship_all_ok"] = t["decoy_regret"]["ship_all"] <= EPS_R / 3
        conds["naive_q1_ok"] = abs(t["naive_ab_delta"] - t["D_full"]) <= W1_CAP / 4
    else:
        wk = max(m for _, m in day_types(p))
        # masking: at the guardrail fraction the autoscaler still has headroom on a +1 sd weekday
        conds["masked_at_guard"] = all(lam[h] * wk * (1 + p["sd_day"]) * sbar(p, h, GUARD_CAP) / U_TGT <= Q
                                       for h in range(GUARD_LO, GUARD_HI + 1))
        # all hours that can saturate at 100% B lie inside the guardrail window (nothing to see off-peak)
        conds["saturation_inside_window"] = all(hi_load(h, 1.0) * wk / U_TGT <= Q for h in H
                                                if not (GUARD_LO <= h <= GUARD_HI))
        # the drill can reach overload: at 01-05 there is a cap k >= 2 with offered load / k in [1.15, 1.6]
        ok_drill = False
        for h in range(DRILL_LO, DRILL_HI + 1):
            a = lam[h] * p["sA"]
            ok_drill = ok_drill or any(1.15 <= a / kk <= 1.6 for kk in range(2, Q + 1))
        conds["drill_feasible"] = ok_drill
        thr = K_DECOY * EPS_R
        for nm in ("ship_all", "guard_safe", "no_ship", "rho_0.95"):
            conds["decoy_" + nm] = t["decoy_regret"][nm] >= thr
        conds["q1_naive_far"] = abs(t["naive_ab_delta"] - t["D_full"]) >= K_DECOY * W1_CAP
        conds["q1_loss_far"] = abs(t["loss_full"] - t["D_full"]) >= 0.5 * K_DECOY * W1_CAP
        conds["peak_interior"] = 0.25 < min(t["x_star"][GUARD_LO:GUARD_HI + 1]) < 0.97
        conds["positive_launch"] = t["D_star"] >= 4 * EPS_R
    info["conds"] = {k_: bool(v) for k_, v in conds.items()}
    return list(conds.values()), info


def _sample(seed, level):
    g = np.random.default_rng(73000 + seed)
    p = None
    for _ in range(60):
        p = _draw(g, level)
        if all(_conditions(p)[0]):
            break
    return p


def sample_params(seed):
    return _sample(seed, LEVEL)


def instance_gate(p):
    c, info = _conditions(p)
    t = dict(truth(p))
    t.update(info)
    return bool(all(c)), t


def pool_gate(seeds):
    return _pool([sample_params(s) for s in seeds])


def _pool(ps):
    n = len(ps)
    if n < 4:
        return True, {"n": n}
    qs = {}
    for p in ps:
        qs[p["Q"]] = qs.get(p["Q"], 0) + 1
    mc = max(qs.values())
    return bool(mc <= math.ceil(n / 2.0) and len(qs) >= 2), {"n": n, "quota_counts": qs}


# ------------------------------------------------------------------ observation model (what the lab reports)
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HIST_COLS = ["day", "dow", "hour", "requests", "replicas", "utilization", "abandon_rate", "mean_queue_wait_s",
             "mean_rating", "mean_gen_time_s", "mean_score"]
EXP_COLS = ["day", "dow", "hour", "fraction_B", "replicas", "utilization", "abandon_rate", "mean_queue_wait_s",
            "requests_A", "requests_B", "rating_A", "rating_B", "gen_time_A_s", "gen_time_B_s", "score_A", "score_B"]


def _day_mult(p, g, dow):
    wk = p.get("w_we", 1.0) if dow >= 5 else 1.0
    return wk * (1.0 + p["sd_day"] * float(g.normal()))


def _obs_cell(p, h, x, lam, cap, g):
    """One (day, hour) cell of telemetry.  Expectations are exact (Erlang-A); noise is unbiased."""
    st = hour_state(p, h, x, lam, cap)
    z = float(g.normal())
    Pab = max(0.0, st["Pab"] * (1.0 + QNOISE * z + 0.03 * float(g.normal())))
    EW = max(0.0, st["EW"] * (1.0 + QNOISE * z + 0.03 * float(g.normal())))
    util = min(1.0, st["util"] * (1.0 + 0.004 * float(g.normal())))
    out = {"replicas": st["c"], "utilization": round(util, 4), "abandon_rate": round(Pab, 5),
           "mean_queue_wait_s": round(EW, 3)}
    arms = {}
    for arm, frac, kk, qq in (("A", 1.0 - x, 1.0, p["qA"]), ("B", x, kap(p, h), p["qA"] + dq(p, h))):
        n = int(g.poisson(lam * 3600.0 * frac)) if frac > 0 else 0
        if n == 0:
            arms[arm] = (0, None, None, None)
            continue
        ns = max(1, int(round(n * (1.0 - min(Pab, 1.0)))))
        r = qq + RATING_SD * float(g.normal()) / math.sqrt(ns)
        s = p["sA"] * kk * (1.0 + float(g.normal()) / math.sqrt(ns))
        sc = r * (1.0 - Pab) - p["eta"] * EW
        arms[arm] = (n, round(r, 5), round(s, 4), round(sc, 5))
    return out, arms


def _fmt(v):
    return "" if v is None else str(v)


def _hist_mults(p):
    """History day multipliers, CONSTRUCTED so that within each day type the 14-day sample has mean exactly 1
    (x the day-type factor) and sample sd exactly sd_day: the history is representative by construction, so
    the population load an agent estimates from it is the load the truth uses (no unlucky-history bias)."""
    g = np.random.default_rng(p["hist_seed"])
    z = g.normal(size=HIST_DAYS)
    dows = [(p["dow0"] + d) % 7 for d in range(HIST_DAYS)]
    groups = [[d for d in range(HIST_DAYS) if dows[d] < 5], [d for d in range(HIST_DAYS) if dows[d] >= 5]]
    if p.get("w_we", 1.0) == 1.0:
        groups = [list(range(HIST_DAYS))]
    for gr in groups:
        v = z[gr]
        z[gr] = (v - v.mean()) / v.std(ddof=1)
    return [(p.get("w_we", 1.0) if dows[d] >= 5 else 1.0) * (1.0 + p["sd_day"] * float(z[d]))
            for d in range(HIST_DAYS)], dows, g


def history_rows(p):
    mults, dows, g = _hist_mults(p)
    rows = []
    for d in range(HIST_DAYS):
        dow = dows[d]
        m = mults[d]
        for h in H:
            h = int(h)
            out, arms = _obs_cell(p, h, 0.0, p["lam"][h] * m, None, g)
            n, r, s, sc = arms["A"]
            rows.append([d + 1, DOW[dow], h, n, out["replicas"], out["utilization"], out["abandon_rate"],
                         out["mean_queue_wait_s"], r, s, sc])
    return rows


def _csv(cols, rows):
    b = io.StringIO()
    w = csv.writer(b, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([_fmt(v) for v in r])
    return b.getvalue()


# ------------------------------------------------------------------ ops
def _hours(a):
    v = a.get("hours", "all")
    if v in ("all", None, ""):
        return list(range(24))
    out = []
    try:
        if isinstance(v, (int, float)):
            out = [int(v)]
        elif isinstance(v, list):
            out = [int(num(x, "hours[]", 0, 23, integer=True)) for x in v]
        else:
            for part in str(v).split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = part.split("-")
                    lo, hi = int(lo), int(hi)
                    if not (0 <= lo <= hi <= 23):
                        raise ValueError
                    out += list(range(lo, hi + 1))
                elif part:
                    out.append(int(part))
    except LabError:
        raise
    except Exception:
        raise LabError("hours must be 'all', a list of hours 0-23, or ranges like '0-7,22-23'; got %r" % (v,))
    out = sorted(set(out))
    if not out or any(h < 0 or h > 23 for h in out):
        raise LabError("hours must be within 0-23; got %r" % (v,))
    return out


def _parse_test(w, a):
    hours = _hours(a)
    if "schedule" in a:
        sch = a["schedule"]
        if not isinstance(sch, list) or len(sch) != 24:
            raise LabError("schedule must be a list of 24 fractions (hour 0 .. hour 23)")
        fr = {h: num(sch[h], "schedule[%d]" % h, 0.0, 1.0) for h in hours}
    else:
        f = num(a.get("fraction"), "fraction", 0.0, 1.0)
        fr = {h: f for h in hours}
    days = int(num(a.get("days", 1), "days", 1, MAX_DAYS, integer=True))
    cap = a.get("max_replicas")
    if cap is not None:
        cap = int(num(cap, "max_replicas", 1, w.p["Q"], integer=True))
        bad = [h for h in hours if not (DRILL_LO <= h <= DRILL_HI)]
        if bad:
            raise LabError("max_replicas (capacity drill) is only allowed when every tested hour is in the "
                           "maintenance window %02d:00-%02d:59; offending hours %s" % (DRILL_LO, DRILL_HI, bad))
    over = [h for h in hours if GUARD_LO <= h <= GUARD_HI and fr[h] > GUARD_CAP + 1e-12]
    if over:
        raise LabError("guardrail: between %02d:00 and %02d:59 an experiment may route at most %d%% of traffic "
                       "to an unlaunched model; hours %s exceed it" % (GUARD_LO, GUARD_HI, round(100 * GUARD_CAP),
                                                                        over))
    return hours, fr, days, cap


def _cost_test(w, a):
    return float(_parse_test(w, a)[2])


def _run_test(w, a, ctx):
    p = w.p
    hours, fr, days, cap = _parse_test(w, a)
    d0 = int(sum(r["cost"] for r in ctx["session"].records if r["op"] == "abtest"))  # calendar days used so far
    rows = []
    tot = {"A": [0, 0.0], "B": [0, 0.0]}
    for j in range(days):
        dayno = d0 + j
        dow = (p["dow0"] + HIST_DAYS + dayno) % 7
        g = rng_for(w.salt, "day", dayno)
        m = _day_mult(p, g, dow)
        gc = rng_for(w.salt, "cells", ctx["i"], j)
        for h in hours:
            out, arms = _obs_cell(p, h, fr[h], p["lam"][h] * m, cap, gc)
            for arm in "AB":
                n, r, s, sc = arms[arm]
                if n:
                    tot[arm][0] += n
                    tot[arm][1] += n * sc
            rows.append([HIST_DAYS + dayno + 1, DOW[dow], h, fr[h], out["replicas"], out["utilization"],
                         out["abandon_rate"], out["mean_queue_wait_s"], arms["A"][0], arms["B"][0],
                         arms["A"][1], arms["B"][1], arms["A"][2], arms["B"][2], arms["A"][3], arms["B"][3]])
    fn = "data/abtest_%03d.csv" % ctx["i"]
    if ctx.get("app_dir"):
        fp = os.path.join(ctx["app_dir"], fn)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, "w").write(_csv(EXP_COLS, rows))
    summ = {arm: {"requests": tot[arm][0],
                  "mean_score": round(tot[arm][1] / tot[arm][0], 5) if tot[arm][0] else None} for arm in "AB"}
    return {"file": "/app/" + fn, "calendar_days": [HIST_DAYS + d0 + 1, HIST_DAYS + d0 + days],
            "hours": hours, "max_replicas": cap, "rows": len(rows), "summary": summ}


def _read_csv(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        out.append({k: (float(v) if v not in ("",) and k != "dow" else (v if k == "dow" else None))
                    for k, v in r.items()})
    return out


# ------------------------------------------------------------------ deliverables
def _parse_rollout(txt):
    xs = {}
    for ln, line in enumerate(txt.splitlines(), 1):
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        parts = s.replace("=", " ").replace(":", " ").split()
        if len(parts) != 2:
            raise ValueError("line %d: expected '<hour> <fraction>'" % ln)
        h = int(parts[0])
        f = float(parts[1])
        if not (0 <= h <= 23):
            raise ValueError("line %d: hour %d out of range" % (ln, h))
        if not (math.isfinite(f) and 0.0 <= f <= 1.0):
            raise ValueError("line %d: fraction must be in [0, 1]" % ln)
        if h in xs:
            raise ValueError("line %d: hour %d listed twice" % (ln, h))
        xs[h] = f
    if sorted(xs) != list(range(24)):
        raise ValueError("every hour 0-23 must appear exactly once (missing %s)" % sorted(set(range(24)) - set(xs)))
    return np.array([xs[h] for h in range(24)])


_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def _parse_answer(txt, key):
    import re
    # Markdown decoration is accepted: the task text itself shows the form as a bullet in backticks
    # ("* `Q1: [lo, hi]`"), so a memo that copies that layout must parse (2026-09-29: two Fable runs were
    # failed on R0_format for exactly this; grader bug, fixed before any result was reported).
    m = re.findall(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?[*_`]*%s[*_`]*\s*[:=]\s*[*_`]*\[?\s*%s\s*,\s*%s\s*\]?"
                   % (key, _NUM, _NUM), txt, re.M)
    if len(m) != 1:
        raise ValueError("report.md must contain exactly one line '%s: [lo, hi]' (found %d)" % (key, len(m)))
    lo, hi = float(m[0][0]), float(m[0][1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError("%s: need finite lo <= hi" % key)
    return lo, hi


class World(_W):
    NAME = "serve_ab"
    ARTIFACTS = ["rollout.conf", "report.md"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "experiment-days"
    OPS = {"abtest": (_cost_test, _run_test,
                      "run a live experiment for N days: route a fraction of traffic to model B in the given hours")}

    def public_spec(self):
        p = self.p
        return {"ops": {"abtest": {
            "args": {"fraction": "share of the tested hours' traffic routed to B (0..1), or use schedule",
                     "schedule": "alternative to fraction: list of 24 per-hour fractions (only listed hours run)",
                     "hours": "'all' (default), a list, or ranges like '0-7,22-23' - hours not listed run "
                              "production (all A) and are not reported",
                     "days": "1..%d consecutive calendar days" % MAX_DAYS,
                     "max_replicas": "optional capacity drill: cap the autoscaler at this many replicas "
                                     "(allowed only if every tested hour is within %02d:00-%02d:59)" % (DRILL_LO,
                                                                                                         DRILL_HI)},
            "guardrail": "hours %02d-%02d: fraction <= %.2f" % (GUARD_LO, GUARD_HI, GUARD_CAP),
            "cost": "1 experiment-day per calendar day, whatever the hours or fraction",
            "returns": "per (day, hour) telemetry CSV written under /app/data/ + a per-arm summary"}},
            "quota_replicas": p["Q"], "autoscaler": {"target_utilization": U_TGT, "min_replicas": CMIN},
            "eta_per_queue_second": p["eta"],
            "deliverables": ["/app/rollout.conf", "/app/report.md"]}

    def grade(self, art_dir, ledger=None):
        p = self.p
        t = truth(p)
        items = {}
        why = []
        xs = None
        q1 = q2 = None
        try:
            xs = _parse_rollout(open(os.path.join(art_dir, "rollout.conf")).read())
        except Exception as ex:
            why.append("rollout.conf: %s" % (ex if isinstance(ex, ValueError) else type(ex).__name__))
        try:
            rep = open(os.path.join(art_dir, "report.md")).read()
            q1 = _parse_answer(rep, "Q1")
            q2 = _parse_answer(rep, "Q2")
        except Exception as ex:
            why.append("report.md: %s" % (ex if isinstance(ex, ValueError) else type(ex).__name__))
        items["R0_format"] = (not why, "; ".join(why) or "rollout.conf and report.md are well-formed")
        if why:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()}}
        Ds = t["D_star"]
        Dx = daily_delta(p, xs)
        reg = Ds - Dx
        items["R1_schedule"] = (reg <= EPS_R + 1e-12,
                                "expected daily score change of your schedule %+.5f vs the best achievable %+.5f: "
                                "regret %.5f (allowed %.4f)" % (Dx, Ds, reg, EPS_R))
        Df = t["D_full"]
        cov1, wid1 = q1[0] <= Df <= q1[1], q1[1] - q1[0]
        items["R2_full_launch"] = (bool(cov1 and wid1 <= W1_CAP + 1e-6),
                                   "Q1 truth %+.5f; you said [%+.5f, %+.5f] (%s, width %.5f, cap %.4f)"
                                   % (Df, q1[0], q1[1], "covers" if cov1 else "misses", wid1, W1_CAP))
        cov2, wid2 = q2[0] <= Dx <= q2[1], q2[1] - q2[0]
        items["R3_own_schedule"] = (bool(cov2 and wid2 <= W2_CAP + 1e-6),
                                    "Q2 truth (your schedule) %+.5f; you said [%+.5f, %+.5f] (%s, width %.5f, cap "
                                    "%.4f)" % (Dx, q2[0], q2[1], "covers" if cov2 else "misses", wid2, W2_CAP))
        cand = {"ship_all": np.ones(24), "guard_safe": sched_guard(p), "no_ship": np.zeros(24),
                "keep_target": sched_keep_target(p), "rho_0.90": sched_rho(p, 0.90), "optimum": optimum(p)[0]}
        dist = {k: round(float(np.abs(xs - v).mean()), 4) for k, v in cand.items()}
        closest = min(dist, key=dist.get)
        pk = t["peak_hour"]
        diag = {"regret": round(reg, 6), "D_submitted": round(Dx, 6), "closest_schedule": closest,
                "closest_dist": dist[closest], "peak_fraction": round(float(xs[pk]), 3),
                "q1_centre_minus_truth": round((q1[0] + q1[1]) / 2 - Df, 6),
                "q1_centre_minus_naive": round((q1[0] + q1[1]) / 2 - t["naive_ab_delta"], 6),
                "q1_centre_minus_fluid": round((q1[0] + q1[1]) / 2 - t["fluid_full"], 6),
                "q1_centre_minus_lossmodel": round((q1[0] + q1[1]) / 2 - t["loss_full"], 6),
                "q2_centre_minus_truth": round((q2[0] + q2[1]) / 2 - Dx, 6)}
        if ledger is not None:
            tests = [r for r in ledger if r["op"] == "abtest"]
            drills = [r for r in tests if r["args"].get("max_replicas") is not None]
            peakfr = []
            for r in tests:
                try:
                    hs, fr, _, _ = _parse_test(self, r["args"])
                    peakfr += [fr[h] for h in hs if GUARD_LO <= h <= GUARD_HI]
                except Exception:
                    pass
            diag.update({"n_tests": len(tests), "n_drills": len(drills),
                         "drill_days": int(sum(r["cost"] for r in drills)),
                         "max_peak_fraction_tested": max(peakfr) if peakfr else None,
                         "spent": round(sum(r["cost"] for r in ledger), 2)})
        n_ok = sum(1 for v_ in items.values() if v_[0])
        return {"pass": bool(all(v_[0] for v_ in items.values())), "score": round(n_ok / 4.0, 4),
                "items": {k: {"ok": bool(v_[0]), "detail": v_[1]} for k, v_ in items.items()},
                "diag": diag, "ref": t}


# ------------------------------------------------------------------ strategies (public facts only)
def _hist(sess):
    p = sess.w.p
    return [{"day": r[0], "dow": r[1], "hour": r[2], "requests": r[3], "replicas": r[4], "utilization": r[5],
             "abandon_rate": r[6], "mean_queue_wait_s": r[7], "mean_rating": r[8], "mean_gen_time_s": r[9],
             "mean_score": r[10]} for r in history_rows(p)]


def _test(sess, **a):
    res = sess.call("abtest", a)
    fp = os.path.join(sess.app_dir, res["file"][len("/app/"):])
    return _read_csv(fp)


def _fit_history(sess):
    """load profile (weekday level), weekend factor, day-to-day sd, sA, qA from the 14-day history."""
    hist = _hist(sess)
    wkend = lambda d: d in ("Sat", "Sun")
    days = sorted(set(r["day"] for r in hist))
    tot = {d: sum(r["requests"] for r in hist if r["day"] == d) for d in days}
    dw = {d: next(r["dow"] for r in hist if r["day"] == d) for d in days}
    wd = [d for d in days if not wkend(dw[d])]
    we = [d for d in days if wkend(dw[d])]
    mwd = np.mean([tot[d] for d in wd])
    w_we = float(np.mean([tot[d] for d in we]) / mwd) if we else 1.0
    if abs(w_we - 1.0) < 0.03:
        w_we, wd = 1.0, days
        mwd = np.mean([tot[d] for d in days])
    sd = float(np.std([tot[d] / mwd for d in wd], ddof=1))
    lam = [float(np.mean([r["requests"] for r in hist if r["hour"] == h and r["day"] in wd])) / 3600.0 for h in H]
    n = sum(r["requests"] for r in hist)
    sA = sum(r["requests"] * r["mean_gen_time_s"] for r in hist) / n
    qA = sum(r["requests"] * r["mean_rating"] for r in hist) / n
    return {"lam": lam, "w_we": round(w_we, 4), "sd_day": max(sd, 1e-4), "sA": sA, "qA": qA,
            "Q": sess.w.p["Q"], "eta": sess.w.p["eta"]}


def _fit_arms(rows, base):
    """per-hour kappa (gen-time ratio) and quality lift (rating_B - rating_A) from A/B rows."""
    kh, dh = [], []
    for h in H:
        rr = [r for r in rows if r["hour"] == h and r["requests_B"] and r["requests_A"]]
        if not rr:
            kh.append(None); dh.append(None); continue
        wB = np.array([r["requests_B"] for r in rr]); wA = np.array([r["requests_A"] for r in rr])
        gB = np.sum(wB * [r["gen_time_B_s"] for r in rr]) / wB.sum()
        gA = np.sum(wA * [r["gen_time_A_s"] for r in rr]) / wA.sum()
        rB = np.sum(wB * [r["rating_B"] for r in rr]) / wB.sum()
        rA = np.sum(wA * [r["rating_A"] for r in rr]) / wA.sum()
        kh.append(float(gB / gA)); dh.append(float(rB - rA))
    return kh, dh


def _theta_from(rows):
    num_ = sum(r["requests_A"] * r["abandon_rate"] + (r["requests_B"] or 0) * r["abandon_rate"] for r in rows)
    den = sum(r["requests_A"] * r["mean_queue_wait_s"] + (r["requests_B"] or 0) * r["mean_queue_wait_s"]
              for r in rows)
    return num_ / den


def _write(art_dir, xs, q1, q2, note=""):
    with open(os.path.join(art_dir, "rollout.conf"), "w") as f:
        f.write("# hour fraction_B\n")
        for h in H:
            f.write("%02d %.4f\n" % (h, float(xs[h])))
    with open(os.path.join(art_dir, "report.md"), "w") as f:
        f.write("# Launch report\n\n%s\n\nQ1: [%.7f, %.7f]\nQ2: [%.7f, %.7f]\n" % (note, q1[0], q1[1], q2[0], q2[1]))


def _iv(c, w):
    return (c - 0.45 * w, c + 0.45 * w)


def _guard_ab(sess, days=7):
    """the standard, guardrail-compliant A/B: 30% everywhere for a week."""
    return _test(sess, fraction=GUARD_CAP, hours="all", days=days)


def _model(sess, ab_rows, theta):
    ph = _fit_history(sess)
    kh, dh = _fit_arms(ab_rows, ph)
    ph.update({"kap_h": kh, "dq_h": dh, "theta": theta})
    return ph


def strat_oracle(sess, art_dir, rng):
    """history -> load/sA/qA; guardrailed A/B (30%, 7 days) -> per-hour kappa and quality lift; capacity drill
    at 01-05 with the replica cap set so offered load / cap ~ 1.3 -> patience rate theta = abandon / wait
    (Little's law for memoryless abandonment); Erlang-A model with the autoscaler rule and day-to-day noise ->
    per-hour optimal fraction; both answers from the same model."""
    ab = _guard_ab(sess)
    ph = _fit_history(sess)
    a_night = np.median([ph["lam"][h] * ph["sA"] for h in range(DRILL_LO, DRILL_HI + 1)])
    k = max(1, int(round(a_night / 1.3)))
    dr = _test(sess, fraction=0.0, hours="%d-%d" % (DRILL_LO, DRILL_HI), days=2, max_replicas=k)
    ph = _model(sess, ab, _theta_from(dr))
    xs, D = optimum(ph)
    Df = daily_delta(ph, np.ones(24))
    _write(art_dir, xs, _iv(Df, W1_CAP), _iv(D, W2_CAP))


def strat_ship_all(sess, art_dir, rng):
    """P1 ABLATION: trust the randomised A/B (B wins on score) and ship 100%; answers = the A/B lift."""
    ab = _guard_ab(sess)
    nA = sum(r["requests_A"] for r in ab); nB = sum(r["requests_B"] for r in ab)
    lift = (sum(r["requests_B"] * r["score_B"] for r in ab) / nB - sum(r["requests_A"] * r["score_A"] for r in ab) / nA)
    _write(art_dir, np.ones(24), _iv(lift, W1_CAP), _iv(lift, W2_CAP))


def strat_guard_safe(sess, art_dir, rng):
    """P2 ABLATION: knows the A/B nets out shared-queue costs, so looks at the pool itself during the
    guardrailed test (queue wait / abandonment vs the history) - sees no change - and ships what was
    validated: 30% at peak hours, 100% elsewhere.  Answers from the measured lift, with the pool's own
    measured score change."""
    ab = _guard_ab(sess)
    ph = _fit_history(sess)
    kh, dh = _fit_arms(ab, ph)
    xs = sched_guard(sess.w.p)
    lam = np.array(ph["lam"])
    lift = float(np.sum(lam * np.array(dh)) / lam.sum())
    q2 = float(np.sum(lam * np.array(dh) * xs) / lam.sum())
    _write(art_dir, xs, _iv(lift, W1_CAP), _iv(q2, W2_CAP))


def strat_loss_no_drill(sess, art_dir, rng):
    """P3 ABLATION: does the capacity arithmetic right (autoscaler, quota, per-hour kappa, day noise) but has
    no measurement of patience, so it prices overflow with the zero-patience loss model (Erlang-B: a request
    that finds no free replica is lost, nobody waits).  Schedule optimised under that model; answers from it."""
    ab = _guard_ab(sess)
    ph = _model(sess, ab, 1.0)
    grid = np.linspace(0, 1, 51)
    best = []
    for h in H:
        h = int(h)
        vals = []
        for x in grid:
            xs = np.zeros(24); xs[h] = x
            vals.append(loss_delta(ph, xs))
        best.append(float(grid[int(np.argmax(vals))]))
    xs = np.array(best)
    _write(art_dir, xs, _iv(loss_delta(ph, np.ones(24)), W1_CAP), _iv(loss_delta(ph, xs), W2_CAP))


def strat_no_ship(sess, art_dir, rng):
    """P4 ABLATION: sees the capacity risk and does not launch; Q1 from the oracle's model (right), Q2 = 0."""
    ab = _guard_ab(sess)
    ph = _fit_history(sess)
    a_night = np.median([ph["lam"][h] * ph["sA"] for h in range(DRILL_LO, DRILL_HI + 1)])
    k = max(1, int(round(a_night / 1.3)))
    dr = _test(sess, fraction=0.0, hours="%d-%d" % (DRILL_LO, DRILL_HI), days=2, max_replicas=k)
    ph = _model(sess, ab, _theta_from(dr))
    _write(art_dir, np.zeros(24), _iv(daily_delta(ph, np.ones(24)), W1_CAP), (-0.4 * W2_CAP, 0.4 * W2_CAP))


def strat_sweep_guard(sess, art_dir, rng):
    """SEARCH: pool-level dose-response inside the guardrail.  Days at 0%, 10%, 20%, 30% everywhere; per hour,
    regress the pool's mean score (all requests) on the fraction and extrapolate the line; ship the per-hour
    argmax of the fitted line (0 or 1), answers from the same line."""
    fits = {h: [] for h in H}
    for f in (0.0, 0.1, 0.2, 0.3):
        rows = _test(sess, fraction=f, hours="all", days=3)
        for r in rows:
            n = (r["requests_A"] or 0) + (r["requests_B"] or 0)
            sc = ((r["requests_A"] or 0) * (r["score_A"] or 0) + (r["requests_B"] or 0) * (r["score_B"] or 0)) / n
            fits[int(r["hour"])].append((f, sc, n))
    ph = _fit_history(sess)
    lam = np.array(ph["lam"])
    slope = []
    for h in H:
        x = np.array([a for a, _, _ in fits[h]]); y = np.array([b for _, b, _ in fits[h]])
        slope.append(float(np.polyfit(x, y, 1)[0]))
    slope = np.array(slope)
    xs = (slope > 0).astype(float)
    D1 = float(np.sum(lam * slope) / lam.sum())
    Dx = float(np.sum(lam * slope * xs) / lam.sum())
    _write(art_dir, xs, _iv(D1, W1_CAP), _iv(Dx, W2_CAP))


def strat_offpeak_full(sess, art_dir, rng):
    """SEARCH: the guardrail only binds 08-21, so measure 100% B where it is allowed (00-07, 22-23) against
    history, fit score change vs offered load across those hours, extrapolate to the peak hours."""
    rows = _test(sess, fraction=1.0, hours="0-7,22-23", days=7)
    ph = _fit_history(sess)
    lam = np.array(ph["lam"])
    hist = _hist(sess)
    base = {h: np.mean([r["mean_score"] for r in hist if r["hour"] == h]) for h in H}
    pts = []
    for h in sorted(set(int(r["hour"]) for r in rows)):
        sc = np.mean([r["score_B"] for r in rows if int(r["hour"]) == h])
        pts.append((lam[h], sc - base[h]))
    x = np.array([a for a, _ in pts]); y = np.array([b for _, b in pts])
    c1 = np.polyfit(x, y, 1)
    dh = np.polyval(c1, lam)
    xs = (dh > 0).astype(float)
    _write(art_dir, xs, _iv(float(np.sum(lam * dh) / lam.sum()), W1_CAP),
           _iv(float(np.sum(lam * dh * xs) / lam.sum()), W2_CAP))


def strat_wide(sess, art_dir, rng):
    t = truth(sess.w.p)
    _write(art_dir, optimum(sess.w.p)[0], (t["D_full"] - 0.02, t["D_full"] + 0.02),
           (t["D_star"] - 0.02, t["D_star"] + 0.02))


def strat_nop(sess, art_dir, rng):
    pass


def strat_bad_format(sess, art_dir, rng):
    open(os.path.join(art_dir, "rollout.conf"), "w").write("peak: 0.5\n")
    open(os.path.join(art_dir, "report.md"), "w").write("Q1 is about 0.01\n")


STRATEGIES = {"oracle": (strat_oracle, "pass"),
              "ship_all": (strat_ship_all, "fail"),
              "guard_safe": (strat_guard_safe, "fail"),
              "loss_no_drill": (strat_loss_no_drill, "fail"),
              "no_ship": (strat_no_ship, "fail"),
              "sweep_guard": (strat_sweep_guard, "fail"),
              "offpeak_full": (strat_offpeak_full, "fail"),
              "wide": (strat_wide, "fail"),
              "nop": (strat_nop, "fail"),
              "bad_format": (strat_bad_format, "fail")}
NOISY_FAIL = ()
SEARCH = ["sweep_guard", "offpeak_full"]
PRINCIPLES = {"P1_ab_nets_out_shared_cost": ("ship_all", ["R1_schedule", "R2_full_launch"]),
              "P2_guardrail_test_is_masked": ("guard_safe", ["R1_schedule"]),
              "P3_overload_priced_by_patience": ("loss_no_drill", ["R2_full_launch"]),
              "P4_partial_launch_beats_none": ("no_ship", ["R1_schedule"])}


# ------------------------------------------------------------------ grader falsification
def _put(art_dir, xs, q1, q2):
    _write(art_dir, xs, q1, q2)


def _mut_sched(p, t, art_dir, leg):
    xs, Ds = optimum(p)
    if leg == "small":
        ys = np.clip(xs + 0.01, 0, 1)
        Dy = daily_delta(p, ys)
        _put(art_dir, ys, _iv(t["D_full"], W1_CAP), _iv(Dy, W2_CAP))
        return None
    # the worse of the two plausible wrong launches (100% everywhere; guardrail fraction at the peak)
    ys = max((np.ones(24), sched_guard(p)), key=lambda z: Ds - daily_delta(p, z))
    _put(art_dir, ys, _iv(t["D_full"], W1_CAP), _iv(daily_delta(p, ys), W2_CAP))
    return "R1_schedule"


def _mut_q1(p, t, art_dir, leg):
    xs, Ds = optimum(p)
    d = 0.1 * W1_CAP if leg == "small" else 0.6 * W1_CAP
    _put(art_dir, xs, (t["D_full"] + d - 0.45 * W1_CAP, t["D_full"] + d + 0.45 * W1_CAP), _iv(Ds, W2_CAP))
    return None if leg == "small" else "R2_full_launch"


def _mut_q1_width(p, t, art_dir, leg):
    xs, Ds = optimum(p)
    w = W1_CAP if leg == "small" else 1.2 * W1_CAP
    _put(art_dir, xs, (t["D_full"] - w / 2, t["D_full"] + w / 2), _iv(Ds, W2_CAP))
    return None if leg == "small" else "R2_full_launch"


def _mut_q2(p, t, art_dir, leg):
    xs, Ds = optimum(p)
    d = 0.1 * W2_CAP if leg == "small" else 0.6 * W2_CAP
    _put(art_dir, xs, _iv(t["D_full"], W1_CAP), (Ds + d - 0.45 * W2_CAP, Ds + d + 0.45 * W2_CAP))
    return None if leg == "small" else "R3_own_schedule"


def _mut_format(p, t, art_dir, leg):
    xs, Ds = optimum(p)
    _put(art_dir, xs, _iv(t["D_full"], W1_CAP), _iv(Ds, W2_CAP))
    if leg == "small":
        txt = open(os.path.join(art_dir, "report.md")).read()
        open(os.path.join(art_dir, "report.md"), "w").write(
            "Some prose with numbers 0.1, 0.2 and a table.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n" + txt)
        return None
    lines = open(os.path.join(art_dir, "rollout.conf")).read().splitlines()
    open(os.path.join(art_dir, "rollout.conf"), "w").write("\n".join(lines[:-1]) + "\n")   # hour 23 missing
    return "R0_format"


MUTATE = [("schedule", _mut_sched), ("q1_coverage", _mut_q1), ("q1_width", _mut_q1_width),
          ("q2_coverage", _mut_q2), ("format", _mut_format)]


# ------------------------------------------------------------------ task text: operational facts only
def instruction(p, t):
    return """# Launch model B on the shared serving pool

Our chat assistant is served from **one pool of GPU replicas** behind **one FIFO request queue**.  Model A
serves all traffic today.  Model B won the offline evaluation and the product team wants it launched.  You
decide the launch: **the fraction of each hour's traffic routed to B**, for every hour of the day (the same
schedule every day of the week).

## How serving works (from the platform team)

* Requests arrive as a Poisson stream; the rate depends on the hour of day and varies from day to day.
  `/app/data/history.csv` has the last %d days of production telemetry (all traffic on A).
* A replica generates one response at a time.  Generation times are exponentially distributed; each model
  has its own mean.  Routing is per request, at random, with the hour's fraction.
* The autoscaler sets the replica count for each hour to `ceil(offered_load / %.2f)`, at least %d and at
  most the GPU quota of **%d replicas**.  `offered_load` = that hour's actual arrival rate x the mean
  generation time of the traffic mix actually running in that hour.  The quota cannot be raised.
* A request that finds every replica busy waits in the queue (first come, first served, both models share
  it).  A waiting user leaves (abandons) at a constant hazard - how long they have already waited does not
  change the chance of leaving in the next second - and nobody has measured their mean patience.  An
  abandoned request gets no answer.  Treat each hour as being in steady state.

## The metric

Per-request score = the user's rating of the answer (0..1) if it is served, **0 if abandoned**, minus
**%.4f per second spent waiting in the queue** (abandoned requests count the time they waited).

The **value** of a launch schedule = the long-run mean per-request score under it (total score over all
requests, over many days, divided by the number of requests) minus the same quantity for all-A.

## Experiments

You have **%g experiment-days**.

* `lab abtest fraction=<f> hours=<which> days=<n> [max_replicas=<k>]` (or `schedule=[24 fractions]` instead
  of `fraction`) - runs live for n consecutive calendar days; each tested hour routes that fraction of its
  traffic to B; hours not listed run production (all A) and are not reported.  Writes per-(day, hour)
  telemetry to a CSV under `/app/data/` and prints a per-arm summary.  Cost: 1 experiment-day per day.
* **Guardrail:** between %02d:00 and %02d:59 an experiment may route at most **%d%%** of traffic to B.
* `max_replicas` caps the autoscaler for the tested hours; SRE only allows it inside the nightly
  maintenance window %02d:00-%02d:59 (every tested hour must be in that window).
* Column definitions: `/app/docs/telemetry.md`.  `lab spec`, `lab status`, `lab history` are free.

## Deliverables

1. **`/app/rollout.conf`** - the launch schedule, one line per hour `HH fraction` (24 lines, hours 00-23,
   fraction of that hour's traffic on B, 0..1).  Graded: the value of your schedule must be within
   **%.4f** of the best value any schedule achieves.
2. **`/app/report.md`** - your launch memo, free-form, for the product and platform teams.  It must contain
   exactly one line of each of these two forms (values are score units per request, e.g. `+0.0123`):
   * `Q1: [lo, hi]` - the value of launching B to **100%% of traffic in every hour**.  Width <= %.4f.
   * `Q2: [lo, hi]` - the value of **your** schedule in `rollout.conf`.  Width <= %.4f.

Graded against the exact expected values; there is no partial credit inside an item.
""" % (HIST_DAYS, U_TGT, CMIN, p["Q"], p["eta"], BUDGET, GUARD_LO, GUARD_HI, round(100 * GUARD_CAP), DRILL_LO,
       DRILL_HI, EPS_R, W1_CAP, W2_CAP)


def docs(p):
    tel = """# Telemetry columns

`/app/data/history.csv` (production, all traffic on A) - one row per (day, hour):

| column | meaning |
|---|---|
| day, dow, hour | calendar day number, weekday, hour of day (00-23) |
| requests | requests that arrived in that hour |
| replicas | replica count the autoscaler ran in that hour |
| utilization | mean fraction of replicas busy |
| abandon_rate | fraction of arriving requests that abandoned in the queue |
| mean_queue_wait_s | mean time in queue over all arriving requests (abandoned ones count the time they waited; served ones that never queued count 0) |
| mean_rating | mean rating of served answers |
| mean_gen_time_s | mean generation time of served answers |
| mean_score | mean per-request score (the metric) over all arriving requests |

Experiment CSVs (`/app/data/abtest_NNN.csv`) - same (day, hour) grain; calendar days continue after the
history (experiments run one after another):

| column | meaning |
|---|---|
| fraction_B | fraction routed to B in that hour |
| replicas, utilization, abandon_rate, mean_queue_wait_s | pool-level, as above |
| requests_A / requests_B | arrivals per arm |
| rating_A / rating_B | mean rating of served answers per arm |
| gen_time_A_s / gen_time_B_s | mean generation time of served answers per arm |
| score_A / score_B | mean per-request score per arm (all arriving requests of that arm) |

Hour-level queue measurements (abandon_rate, mean_queue_wait_s) carry hour-to-hour measurement noise of
roughly 10%%; it is unbiased.  Ratings are noisy per request.
"""
    return {"docs/telemetry.md": tel.replace("%%", "%"),
            "data/history.csv": _csv(HIST_COLS, history_rows(p))}


def hints(p):
    return {1: """
Hint 1: in a shared queue, both arms of an A/B test wait in the same line.
""", 2: """
Hint 2: what does the pool look like at 100% B in the busiest hour, and what would you need to know to price it?
"""}
