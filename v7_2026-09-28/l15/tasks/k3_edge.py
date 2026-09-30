"""K3 lr-edge: pick (width, learning rate) for a single fixed-FLOP pre-training run.

Mechanism (hidden; transplanted from Wortsman et al. 2309.14322 "small-scale proxies for large-scale Transformer
instability" + Chinchilla-style capacity/optimisation decomposition, with counterfactual constants):
  * a run at width n is *unstable* above a critical learning rate eta_c(n) = C * n^-q; q is NOT 1 and must be
    measured (half the instances have q < 1, half q > 1);
  * below the edge, loss decomposes as L = L0 + Ac*n^-ac (capacity) + Ad*(eta*S)^-ad (optimisation progress),
    so a *larger* learning rate is better - but approaching the edge costs a pre-divergence penalty and,
    stochastically, the whole run diverges (p_div is a logistic in rho = eta/eta_c with width w);
  * the FLOP budget ties width and steps: n^2 * S = BUD, so width is bought with steps.
The trap: proxies are affordable only at small width, where eta_c is much higher.  Transplanting the
proxy-optimal eta diverges the target run; scaling it by 1/n is only right when q = 1, which it never is here.
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, choice, rng_for, read_json_artifact

BUD = 4096.0 ** 2 * 6.0e4            # FLOP-proportional budget of the target run: n^2 * S
NS = [362, 512, 724, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585]
N_MIN, N_MAX = 256, 11585
ETA_MIN, ETA_MAX = 1e-5, 8e-2
S_MIN = 3.0e3
L_DIV = 8.0                          # the number a diverged run is scored as
LAB_BUDGET = 6.0e10
BUDGET = LAB_BUDGET                  # proxy compute, in n^2*S units (~6% of the target run)
SIGMA = 0.004
WIDTH_CAP = 0.20                     # measured: the oracle's own forecast error is p90=0.066, p95=0.087, so a
                                     # half-width of 0.10 covers 96% of oracle runs.  A looser cap (0.40, the
                                     # first value tried) let gpt-6-astra submit +-0.19 and pass R2 without ever
                                     # having to know its own accuracy -- the interval must cost something.
TAU = 0.80
TOL_ABS = 0.06                       # absolute regret tolerance on the target run's score


def sample_params(seed):
    g = np.random.default_rng(3500 + seed)
    # q is sampled above 1 only.  Measured, not assumed: when q < 1 the "eta_c ~ 1/n" folk rule is already
    # near-optimal (regret ~0.03-0.11, below the noise the agent can resolve), so those instances cannot
    # discriminate.  Above 1 the same rule over-estimates the edge at the target width and the run diverges.
    q = float(g.uniform(1.20, 1.55))
    # anchor the edge at a measurable width instead of sampling C directly, so eta_c(512) always sits
    # comfortably inside the allowed lr range whatever q is
    ec512 = float(g.uniform(4e-3, 2.2e-2))
    return {"C": round(ec512 * 512.0 ** q, 6), "q": round(q, 4),
            "L0": round(float(g.uniform(1.3, 1.7)), 4),
            "Ac": round(float(g.uniform(12.0, 30.0)), 3), "ac": round(float(g.uniform(0.32, 0.50)), 4),
            "Ad": round(float(g.uniform(1.5, 4.0)), 4), "ad": round(float(g.uniform(0.28, 0.42)), 4),
            "rho_star": round(float(g.uniform(0.45, 0.70)), 4), "pen": round(float(g.uniform(0.30, 0.70)), 4),
            "w": round(float(g.uniform(0.04, 0.10)), 4), "sig": SIGMA}


# ------------------------------------------------------------------------------------------------- simulator
def eta_c(p, n):
    return p["C"] * float(n) ** -p["q"]


def p_div(p, rho):
    return 1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, (rho - 1.0) / p["w"]))))


def clean_loss(p, n, S, eta):
    """Loss of a run that did not diverge."""
    rho = eta / eta_c(p, n)
    b = p["L0"] + p["Ac"] * float(n) ** -p["ac"] + p["Ad"] * (eta * S) ** -p["ad"]
    pen = p["pen"] * (max(0.0, rho - p["rho_star"]) / (1 - p["rho_star"])) ** 2 / max(1e-3, 1 - rho) ** 0.5 if rho < 1 else 1e3
    return b + pen


def expected_loss(p, n, S, eta):
    """What the target run is worth: diverged runs score L_DIV."""
    rho = eta / eta_c(p, n)
    if rho >= 1.4:
        return L_DIV
    pd = p_div(p, rho)
    return pd * L_DIV + (1 - pd) * min(clean_loss(p, n, S, eta), L_DIV)


def optimum(p):
    best = None
    for n in NS:
        S = BUD / n ** 2
        if S < S_MIN:
            continue
        for e in np.geomspace(ETA_MIN, ETA_MAX, 220):
            v = expected_loss(p, n, S, float(e))
            if best is None or v < best[0]:
                best = (v, n, S, float(e))
    # local refine in eta around the grid optimum
    v, n, S, e = best
    for e2 in np.geomspace(e / 1.3, e * 1.3, 120):
        v2 = expected_loss(p, n, S, float(e2))
        if v2 < v:
            v, e = v2, float(e2)
    return {"L_opt": v, "n_opt": n, "S_opt": S, "eta_opt": e}


def opt_under_belief(p, q_belief, n_ref=512, safety=0.92, sharp=0.06):
    """The best config reachable by an agent that measures the edge exactly at one small width and then
    assumes eta_c ~ n^-q_belief.  This is a *competent* decoy: right method, one wrong exponent."""
    C_hat = eta_c(p, n_ref) * n_ref ** q_belief
    best = None
    for n in NS:
        S = BUD / n ** 2
        if S < S_MIN:
            continue
        ehat = C_hat * n ** -q_belief
        for f in np.geomspace(0.05, 0.99, 80):
            e = float(f * ehat)
            if not (ETA_MIN <= e <= ETA_MAX):
                continue
            rho_b = e / ehat
            if rho_b >= 1.0:
                continue
            pen = p["pen"] * (max(0.0, rho_b - p["rho_star"]) / (1 - p["rho_star"])) ** 2 / max(1e-3, 1 - rho_b) ** 0.5
            bel = p["L0"] + p["Ac"] * n ** -p["ac"] + p["Ad"] * (e * S) ** -p["ad"] + pen
            pd = 1.0 / (1.0 + math.exp(-(rho_b - safety) / sharp))
            bel = pd * L_DIV + (1 - pd) * min(bel, L_DIV)
            if best is None or bel < best[0]:
                best = (bel, n, S, e)
    return {"L_believed": best[0], "n": best[1], "S": best[2], "eta": best[3],
            "L_true": expected_loss(p, best[1], best[2], best[3])}


def truth(p):
    o = optimum(p)
    n0, S0 = 1024, 2.0e4
    bp = min(((expected_loss(p, n0, S0, float(e)), float(e)) for e in np.geomspace(ETA_MIN, ETA_MAX, 220)))
    nT = o["n_opt"]; ST = BUD / nT ** 2
    b1 = opt_under_belief(p, 1.0)
    bh = opt_under_belief(p, 0.5)
    return {**o,
            "L_q1": b1["L_true"], "n_q1": b1["n"], "eta_q1": b1["eta"],
            "L_qhalf": bh["L_true"], "n_qhalf": bh["n"],
            "L_transplant": expected_loss(p, nT, ST, bp[1]), "eta_proxy": bp[1],
            "L_small": min(expected_loss(p, 1024, BUD / 1024 ** 2, float(e)) for e in np.geomspace(ETA_MIN, ETA_MAX, 220)),
            "L_big": min(expected_loss(p, 11585, BUD / 11585 ** 2, float(e)) for e in np.geomspace(ETA_MIN, ETA_MAX, 220))}


# ------------------------------------------------------------------------------------------------- lab ops
def _cost_train(w, a):
    n = num(a.get("n"), "n", N_MIN, N_MAX, integer=True)
    S = num(a.get("steps"), "steps", 200, 4.0e5, integer=True)
    return float(n) ** 2 * float(S)


def _run_train(w, a, ctx):
    p = w.p
    n = int(a["n"]); S = int(a["steps"]); eta = num(a.get("eta"), "eta", ETA_MIN, ETA_MAX)
    seed = int(num(a.get("seed", 0), "seed", 0, 1e6, integer=True))
    g = rng_for(w.salt, "train", n, S, round(eta, 12), seed)
    rho = eta / eta_c(p, n)
    if g.random() < p_div(p, rho):
        # divergence: report the step it blew up at, like a real loss spike
        frac = float(np.clip(g.beta(1.5, 3.0), 0.02, 0.98))
        return {"diverged": True, "final_loss": None, "diverged_at_step": int(frac * S),
                "n": n, "steps": S, "eta": eta}
    L = clean_loss(p, n, S, eta)
    # a *high* loss is not a divergence: short, low-lr proxies legitimately end up badly undertrained, and
    # conflating the two would make the stability edge unmeasurable from below.
    return {"diverged": False, "final_loss": round(float(L + g.normal(0, p["sig"])), 5),
            "n": n, "steps": S, "eta": eta}


class World(_W):
    NAME = "k3_edge"
    ARTIFACTS = ["config.json"]
    BUDGET_UNIT = "n^2*steps"
    OPS = {"train": (_cost_train, _run_train, "train a proxy model; returns its final loss or that it diverged")}

    def public_spec(self):
        return {"ops": {"train": {
            "args": {"n": "model width, integer %d..%d" % (N_MIN, N_MAX),
                     "steps": "training steps, integer 200..4e5",
                     "eta": "peak learning rate, %.0e..%.0e" % (ETA_MIN, ETA_MAX),
                     "seed": "optional integer; different seeds are independent runs"},
            "cost": "n^2 * steps",
            "returns": "{diverged, final_loss, diverged_at_step}; final_loss carries run-to-run noise"}},
            "target_run": {"flop_budget_n2S": BUD, "allowed_widths": NS, "eta_range": [ETA_MIN, ETA_MAX],
                           "diverged_run_scored_as": L_DIV},
            "deliverable": "/app/config.json"}

    def grade(self, art_dir, ledger=None):
        t = self.cfg.get("truth") or truth(self.p)
        cfg, err = read_json_artifact(art_dir, "config.json")
        items = {}
        if err:
            items["R0_artifact"] = (False, err)
            return {"pass": False, "score": 0.0, "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()},
                    "diag": {}, "ref": t}
        ok0, why, n, S, eta, lo, hi = True, [], None, None, None, None, None
        try:
            n = int(cfg["n"]); eta = float(cfg["eta"])
            lo = float(cfg["predicted_loss"]["lo"]); hi = float(cfg["predicted_loss"]["hi"])
        except Exception as e:
            ok0 = False; why.append("config.json must contain n, eta and predicted_loss {lo,hi}: %s" % type(e).__name__)
        if ok0:
            if n not in NS:
                ok0 = False; why.append("n=%d is not one of the allowed widths %s" % (n, NS))
            if not (ETA_MIN <= eta <= ETA_MAX):
                ok0 = False; why.append("eta=%g outside [%g, %g]" % (eta, ETA_MIN, ETA_MAX))
            if not (hi > lo):
                ok0 = False; why.append("predicted_loss must have hi > lo")
            elif hi - lo > WIDTH_CAP + 1e-9:
                ok0 = False; why.append("interval width %.4f exceeds the cap %.2f" % (hi - lo, WIDTH_CAP))
        items["R0_artifact"] = (ok0, "; ".join(why) or "config.json is valid (n in the allowed set, eta in range, interval width <= %.2f)" % WIDTH_CAP)
        if not ok0:
            return {"pass": False, "score": 0.0, "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()},
                    "diag": {}, "ref": t}
        S = BUD / n ** 2
        L = expected_loss(self.p, n, S, eta)
        gap = t["L_q1"] - t["L_opt"]
        tol = TOL_ABS
        regret = L - t["L_opt"]
        items["R1_config_quality"] = (regret <= tol + 1e-9,
                                      "your run scores %.4f, the best reachable is %.4f -> regret %.4f, tolerance %.4f" % (L, t["L_opt"], regret, tol))
        items["R2_forecast_covers"] = (lo <= L <= hi,
                                       "your interval [%.4f, %.4f] %s the value of your own run (%.4f)" % (lo, hi, "contains" if lo <= L <= hi else "misses", L))
        diag = {"n": n, "steps": S, "eta": eta, "rho": round(eta / eta_c(self.p, n), 4),
                "p_diverge": round(p_div(self.p, eta / eta_c(self.p, n)), 4),
                "forecast_center_err": round((lo + hi) / 2 - L, 4), "interval_width": round(hi - lo, 4)}
        if ledger is not None:
            tr = [r for r in ledger if r["op"] == "train"]
            diag["n_train"] = len(tr)
            diag["n_diverged_probes"] = sum(1 for r in tr if r["result"].get("diverged"))
            diag["max_width_probed"] = max([r["args"].get("n", 0) for r in tr] or [0])
            diag["distinct_widths"] = len(set(r["args"].get("n") for r in tr))
        return {"pass": all(v[0] for v in items.values()),
                "score": round(float(np.clip(1 - regret / max(gap, 1e-9), -1, 1)), 4),
                "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()}, "diag": diag, "ref": t}



# ------------------------------------------------------------------------------------------------- strategies
def _bisect_edge(sess, n, steps, lo=ETA_MIN, hi=ETA_MAX, iters=7, seed=0):
    """Coarse bracket for the edge at width n (one noisy probe per step)."""
    for k in range(iters):
        mid = float(np.clip(math.sqrt(lo * hi), ETA_MIN, ETA_MAX))
        try:
            r = sess.call("train", {"n": n, "steps": steps, "eta": mid, "seed": seed + k})
        except LabError:
            break
        if r["diverged"]:
            hi = mid
        else:
            lo = mid
    return float(np.clip(math.sqrt(lo * hi), ETA_MIN, ETA_MAX))


def _edge_mle(sess, n, steps, seed0, center, reps=2, pts=7, span=(0.6, 1.7)):
    """Replicated ladder around `center` + logistic MLE for eta_c: divergence is stochastic, so a single
    bisection path is badly biased by whichever way the coin fell."""
    obs = []
    for j, e in enumerate(np.geomspace(center * span[0], center * span[1], pts)):
        e = float(np.clip(e, ETA_MIN, ETA_MAX))
        for r in range(reps):
            try:
                res = sess.call("train", {"n": n, "steps": steps, "eta": e, "seed": seed0 + 17 * j + r})
            except LabError:
                return (center, obs) if not obs else (_mle(obs), obs)
            obs.append((e, 1 if res["diverged"] else 0))
    return _mle(obs), obs


def _mle(obs):
    best = None
    for lec in np.linspace(math.log(ETA_MIN), math.log(ETA_MAX), 400):
        for w in (0.03, 0.05, 0.08, 0.12):
            ll = 0.0
            for e, d in obs:
                rho = math.exp(math.log(e) - lec)
                pd = min(max(1 / (1 + math.exp(-min(50, max(-50, (rho - 1) / w)))), 1e-6), 1 - 1e-6)
                ll += math.log(pd if d else 1 - pd)
            if best is None or ll > best[0]:
                best = (ll, math.exp(lec))
    return best[1]


def _measure_edges(sess, widths, seed0=500):
    """Bracket then MLE at each width; returns {n: eta_c_hat}."""
    out = {}
    for n, st in widths:
        c = _bisect_edge(sess, n, st, iters=7, seed=seed0 + n)
        out[n] = _edge_mle(sess, n, st, seed0 + 3 * n, c)[0]
    return out


def _fit_q(edges):
    ns = np.array(sorted(edges), float)
    ec = np.array([edges[n] for n in sorted(edges)], float)
    A = np.vstack([np.ones_like(ns), -np.log(ns)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ec), rcond=None)
    return math.exp(coef[0]), float(coef[1])


def _fit_loss(sess, C_hat, q_hat, probes):
    """Fit L0 + Ac n^-ac + Ad (eta S)^-ad + penalty to stable probe runs."""
    from ..fit import lm_fit
    pts = []
    for n, st, fracs in probes:
        ehat = C_hat * n ** -q_hat
        for f in fracs:
            e = float(np.clip(f * ehat, ETA_MIN, ETA_MAX))
            try:
                r = sess.call("train", {"n": n, "steps": int(st), "eta": e, "seed": 11})
            except LabError:
                continue
            if not r["diverged"]:
                pts.append((n, st, e, r["final_loss"]))
    if len(pts) < 6:
        return None, pts
    X = np.array(pts, float)

    def resid(x):
        L0, Ac, ac, Ad, ad, ps, pw = x
        rho = X[:, 2] / (C_hat * X[:, 0] ** -q_hat)
        pen = pw * (np.maximum(0.0, rho - ps) / max(1e-3, 1 - ps)) ** 2 / np.maximum(1e-3, 1 - rho) ** 0.5
        return L0 + Ac * X[:, 0] ** -ac + Ad * (X[:, 2] * X[:, 1]) ** -ad + pen - X[:, 3]

    x, _ = lm_fit(resid, [1.5, 20.0, 0.40, 2.5, 0.35, 0.60, 0.50],
                  lo=[0.2, 1.0, 0.15, 0.2, 0.15, 0.30, 0.02], hi=[3.0, 90.0, 0.85, 30.0, 0.75, 0.92, 4.0], iters=250)
    return x, pts


def _optimise(C_hat, q_hat, fit, safety=0.90, sharp=0.05):
    L0, Ac, ac, Ad, ad, ps, pw = fit
    best = None
    for n in NS:
        S = BUD / n ** 2
        if S < S_MIN:
            continue
        for e in np.geomspace(ETA_MIN, ETA_MAX, 240):
            e = float(e)
            rho = e / (C_hat * n ** -q_hat)
            if rho >= 1.0:
                continue
            pen = pw * (max(0.0, rho - ps) / max(1e-3, 1 - ps)) ** 2 / max(1e-3, 1 - rho) ** 0.5
            base = L0 + Ac * n ** -ac + Ad * (e * S) ** -ad + pen
            pd = 1.0 / (1.0 + math.exp(-(rho - safety) / sharp))
            v = pd * L_DIV + (1 - pd) * min(base, L_DIV)
            if best is None or v < best[0]:
                best = (v, n, S, e)
    return best


def _submit(art_dir, n, eta, center, half=0.08):
    half = min(half, WIDTH_CAP / 2 - 1e-6)
    json.dump({"n": int(n), "eta": float(eta), "predicted_loss": {"lo": center - half, "hi": center + half}},
              open(os.path.join(art_dir, "config.json"), "w"))


def _anchor_forecast(sess, n, eta, fit, frac=0.55):
    """Predict the target run's loss by anchoring AT the target width instead of extrapolating into it.

    The global surface fit is built from small-width probes, so its capacity term Ac*n^-ac carries all of
    the width extrapolation error (measured: +-0.15..0.32 at the chosen width).  Short runs at the *exact*
    (n, eta) that was chosen cost n^2 * steps with a small `steps`, which is affordable, and they pin the
    width-dependent constant directly.  Only the step-count extrapolation is left, and the step term
    Ad*(eta*S)^-ad is both small and well measured at the target S.
    """
    ad = float(fit[4])
    S_t = BUD / n ** 2
    room = frac * sess.left() / float(n ** 2)
    if room < 400:
        return None
    plan = [r for r in (room * 0.12, room * 0.28, room * 0.55) if r >= 120]
    pts = []
    for k, st in enumerate(plan):
        st = int(st)
        for rep in range(2):
            try:
                r = sess.call("train", {"n": int(n), "steps": st, "eta": float(eta), "seed": 4100 + 31 * k + rep})
            except LabError:
                break
            if not r["diverged"]:
                pts.append(((eta * st) ** -ad, r["final_loss"]))
    if len(pts) < 3:
        return None
    X = np.array([[1.0, x] for x, _ in pts]); y = np.array([v for _, v in pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0] + coef[1] * (eta * S_t) ** -ad)


def strat_oracle(sess, art_dir, rng):
    """Existence proof: measure the stability edge at 5 widths (replicated, MLE), fit eta_c = C n^-q,
    fit the loss surface on stable probes, then optimise the (width, lr) trade-off with a safety margin."""
    edges = {}
    for n, st in ((256, 500), (512, 400), (1024, 250)):
        c = _bisect_edge(sess, n, st, iters=7, seed=900 + n)
        edges[n] = _edge_mle(sess, n, st, 1700 + n, c, reps=4, pts=7)[0]
    C_hat, q_hat = _fit_q(edges)
    fit, pts = _fit_loss(sess, C_hat, q_hat,
                         [(256, 6000, (0.2, 0.5, 0.8)), (362, 4000, (0.3, 0.7)), (512, 4000, (0.2, 0.5, 0.85)),
                          (724, 2000, (0.35, 0.75)), (1448, 600, (0.4,)), (2896, 200, (0.4,)), (5793, 60, (0.4,))])
    if fit is None:
        _submit(art_dir, 2048, 1e-3, 3.0); return
    v, n, S, eta = _optimise(C_hat, q_hat, fit)
    a = _anchor_forecast(sess, n, eta, fit)
    _submit(art_dir, n, eta, a if a is not None else v, half=0.10)


def _belief_strategy(q_belief):
    """A competent agent that measures the edge properly at one small width but *assumes* the exponent."""
    def f(sess, art_dir, rng):
        n_ref = 512
        c = _bisect_edge(sess, n_ref, 400, iters=7, seed=7)
        ec0 = _edge_mle(sess, n_ref, 400, 900, c, reps=4, pts=7)[0]
        C_hat = ec0 * n_ref ** q_belief
        fit, pts = _fit_loss(sess, C_hat, q_belief,
                             [(256, 6000, (0.2, 0.5, 0.8)), (362, 4000, (0.3, 0.7)), (512, 4000, (0.2, 0.5, 0.85)),
                              (724, 2000, (0.35, 0.75)), (1448, 600, (0.4,)), (2896, 200, (0.4,)), (5793, 60, (0.4,))])
        if fit is None:
            _submit(art_dir, 2048, 1e-3, 3.0); return
        v, n, S, eta = _optimise(C_hat, q_belief, fit)
        a = _anchor_forecast(sess, n, eta, fit)
        _submit(art_dir, n, eta, a if a is not None else v, half=0.10)
    return f


def _proxy_best(sess, n0=512, S0=1.2e4):
    best = None
    for e in np.geomspace(ETA_MIN, ETA_MAX, 16):
        try:
            r = sess.call("train", {"n": n0, "steps": int(S0), "eta": float(e), "seed": 2})
        except LabError:
            break
        if not r["diverged"] and (best is None or r["final_loss"] < best[0]):
            best = (r["final_loss"], float(e))
    return best or (3.0, 1e-3)


def strat_transplant(sess, art_dir, rng):
    """M1': proxy-optimal lr taken straight to a large width."""
    L, e = _proxy_best(sess)
    _submit(art_dir, 4096, e, L)


def strat_small(sess, art_dir, rng):
    """No extrapolation at all: smallest width, lr tuned there."""
    L, e = _proxy_best(sess, 1024, 2.0e4)
    _submit(art_dir, 1024, e, L)


def strat_big(sess, art_dir, rng):
    """Capacity maximalist: biggest width, conservatively small lr."""
    L, e = _proxy_best(sess, 512, 8e3)
    _submit(art_dir, 11585, 0.2 * e * 512.0 / 11585, L)


def strat_recite(sess, art_dir, rng):
    """Literature defaults."""
    _submit(art_dir, 4096, 3e-4, 2.2)


def strat_cheat_wide(sess, art_dir, rng):
    _submit(art_dir, 4096, 1e-3, 3.0, half=WIDTH_CAP / 2)


def strat_nop(sess, art_dir, rng):
    pass


NOISY_FAIL = ("q1", "qhalf", "transplant", "small", "big")
STRATEGIES = {"oracle": (strat_oracle, "pass"),
              "q1": (_belief_strategy(1.0), "fail"), "qhalf": (_belief_strategy(0.5), "fail"),
              "transplant": (strat_transplant, "fail"), "small": (strat_small, "fail"), "big": (strat_big, "fail"),
              "recite": (strat_recite, "fail"), "cheat_wide": (strat_cheat_wide, "fail"), "nop": (strat_nop, "fail")}


def instance_gate(p):
    t = truth(p)
    tol = TOL_ABS
    reg = {k: t[k] - t["L_opt"] for k in ("L_q1", "L_qhalf", "L_small", "L_big", "L_transplant")}
    gap = reg["L_q1"]
    # the tolerance must be wide enough that the oracle's *estimation* noise does not decide the verdict,
    # and every decoy must still miss it by at least 2x
    ok = (min(reg.values()) >= 2.0 * tol and t["L_opt"] < 6.0
          and t["n_opt"] not in (NS[0], NS[-1]))
    return ok, {"gap": round(gap, 4), "tol": round(tol, 4), "n_opt": t["n_opt"], "q": p["q"],
                "L_opt": round(t["L_opt"], 4), **{k: round(v, 4) for k, v in reg.items()}}


# ------------------------------------------------------------------------------------------------- agent-facing text
def instruction(p, t):
    tol = TOL_ABS
    return """# Pick the width and learning rate for one fixed-compute pre-training run

Your team has been allocated compute for **exactly one** pre-training run. The allocation fixes the product

    width^2 * steps = {BUD:.3e}

so choosing a wider model buys capacity and pays for it in steps. You choose the model **width** (from the
allowed list) and the **peak learning rate**; everything else in the recipe is fixed.

The catch the team keeps hitting: above some learning rate a run at a given width becomes unstable and
**diverges** - the loss spikes and the run is lost. A diverged target run is scored as **{L_DIV:.1f}**, i.e. a
total write-off. Runs a little below that edge are not free either: they train with visible loss-spike damage.
Where the edge sits depends on the width, and the team has never measured how.

## The lab

Run `/app/bin/lab spec` first. You can train small proxy models: `lab train n=512 steps=4000 eta=1e-3`
returns either `diverged: true` (with the step it blew up at) or a noisy `final_loss`. A run costs
`n^2 * steps` from your lab budget of **{LAB_BUDGET:.1e}**, which is about {frac:.1%} of the target run -
enough for many small runs, nowhere near enough for one at the target width.

Repeating the same `(n, steps, eta)` with a different `seed` gives an independent run.

## Deliverable: `/app/config.json`

    {{"n": 2048, "eta": 0.001, "predicted_loss": {{"lo": 0.0, "hi": 0.0}}}}

* `n` - the width of the target run, one of {NS}. `steps` follows from the budget.
* `eta` - the peak learning rate, in [{ETA_MIN:.0e}, {ETA_MAX:.0e}].
* `predicted_loss` - an interval for what **your own configuration** scores as the target run: its expected
  final loss, counting a divergence as {L_DIV:.1f}. Width at most {WIDTH_CAP:.2f}.

## How it is graded (mechanically, against the simulator; only the file counts)

1. **R0** - `config.json` is valid: `n` in the allowed list, `eta` in range, interval width <= {WIDTH_CAP:.2f}.
2. **R1** - your configuration's score is within **{tol:.4f}** of the best any allowed (width, learning rate)
   can achieve on this run.
3. **R2** - your interval contains your own configuration's score.

All three must hold. Your lab history is not graded - spend the budget as you see fit. When you are done,
write `/app/config.json` and stop.
""".format(BUD=BUD, L_DIV=L_DIV, LAB_BUDGET=LAB_BUDGET, frac=LAB_BUDGET / BUD, NS=NS,
           ETA_MIN=ETA_MIN, ETA_MAX=ETA_MAX, WIDTH_CAP=WIDTH_CAP, tol=tol)


def docs(p):
    return {"docs/recipe.md": """# Fixed training recipe

Everything below is fixed for both the proxies and the target run; only `n` (width), `steps` and `eta` vary.

* decoder-only Transformer, depth scales with width in the usual way, so compute per step ~ n^2
* AdamW, beta = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0
* linear warmup then cosine decay to 10% of peak; warmup is 2% of steps in every run
* the data is a fixed, effectively unlimited pretraining mixture - no repetition at any of these scales
* `final_loss` is the held-out loss at the end of the run; it carries run-to-run noise of roughly 0.004
* a run that diverges returns no loss at all, only the step where the spike happened

Internal notes from the last two allocations:

* "we tuned the learning rate at width 512 and reused it at 4096; that run was lost on day 3" (postmortem,
  no follow-up measurement was made);
* "the largest model we ever tuned directly was width 1024";
* "somebody claimed the stable learning rate goes like 1/width, but nobody measured the exponent here."
""",
            "docs/allocation.md": """# This allocation

    width^2 * steps = {BUD:.3e}     (fixed)
    allowed widths: {NS}
    learning rate range: [{ETA_MIN:.0e}, {ETA_MAX:.0e}]
    a diverged run scores {L_DIV:.1f}

Lab budget for proxies: {LAB_BUDGET:.1e} in the same n^2*steps units.
""".format(BUD=BUD, NS=NS, ETA_MIN=ETA_MIN, ETA_MAX=ETA_MAX, L_DIV=L_DIV, LAB_BUDGET=LAB_BUDGET)}


def hints(p):
    return {1: "\n## Hint\nThe quantity you most need to know is not a loss. Measure how the thing that kills runs "
               "moves when you change the width.\n",
            2: "\n## Hint\nThe critical learning rate follows eta_c(n) = C * n^-q with q neither 1 nor 1/2 for this "
               "setup. Bisect for the edge at several small widths, fit q, and only then trade width against learning "
               "rate - and leave margin, because your estimate of the edge has error and a divergence costs "
               "everything.\n"}
