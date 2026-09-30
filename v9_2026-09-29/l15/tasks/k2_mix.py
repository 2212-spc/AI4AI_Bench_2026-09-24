"""K2 mix-repeat: choose a pre-training data mixture for a data-constrained target run, and forecast its loss.

Mechanism (hidden; transplanted from Chinchilla-style scaling + Muennighoff et al. 2023 data-constrained
repetition + DoReMi/RegMix-style domain transfer, with counterfactual constants):
  eval component i in {general, code, math}:
      L_i(N, D, r) = E_i + A_i (N/1e8)^-alpha + B_i (T_i/1e9)^-beta_i
      T_i = sum_j W_ij * D'_j ,   n_j = r_j * D  (tokens drawn from domain j)
      D'_j = min(n_j, U_j) + U_j R*_j (1 - exp(-max(n_j/U_j - 1, 0)/R*_j))   (effective data under repetition)
  composite = sum_i lam_i L_i
The small-scale proxy runs the agent can afford never exhaust the scarce domains (math, papers, code), so the
optimum found at proxy scale over-weights them; at the target they are repeated many epochs and their value
decays with a domain-specific R* that differs from the literature's R*~15.  The `pool` knob (restricting the
unique tokens a run may draw from, as in data-constrained scaling studies) lets a careful experimenter
reproduce the target's repetition regime at small scale.
"""
import json, math
import numpy as np
from ..core import World as _W, LabError, num, rng_for, read_json_artifact
from ..fit import simplex_grid

DOM = ["web", "code", "math", "papers"]
EV = ["general", "code", "math"]
NT, DT = 2.5e9, 5e11            # target model params / training tokens
N_RANGE, D_RANGE = (5e7, 4e8), (1e9, 2e10)
PROXY_N, PROXY_D = 1e8, 2e9     # "standard" proxy used by the myopic baseline
SIGMA = 0.004                   # per-run, per-eval-set seed noise
# Max width of the predicted-loss interval.  Measured against the oracle's own forecast error over 20
# (seed, salt) pairs on 2026-09-28: median 0.009, p75 0.017, p90 0.045.  At the old cap of 0.050 the
# oracle's honest interval could only be +-0.025 and it missed on 25% of pairs - R2 was failing the
# reference answer, not the agent.  0.090 puts p90 inside a correctly-sized interval while still being
# far narrower than the myopic-to-optimal gap (0.036-0.088 on the shipped instances), so an interval wide
# enough to cover by default is not wide enough to be uninformative.  `cheat_wide` remains a gate.
WIDTH_CAP = 0.090
# Required fraction of the achievable improvement over the myopic mixture.  Raised from 0.80 once the
# oracle stopped being the binding constraint: at 0.80 the five prior-knowledge mixtures cleared R1 on up
# to 18 of 19 gated seeds, at 0.94 on at most 7 of 19, while the mechanism-fitting oracle scores 0.96-0.999.
TAU = 0.94


def sample_params(seed):
    """Draw one instance.

    The bands below are wide on purpose, and the width was measured rather than chosen.  With the original
    narrow draws every instance had essentially the same answer: the best SINGLE FIXED mixture, searched over
    the whole simplex and submitted blind, cleared R1 on 24 of 24 seeds (measured 2026-09-28).  A task whose
    answer does not move between instances is not testing the search - it is testing whether the model has
    seen a sensible pre-training mixture before, and `{web .44, code .40, math .06, papers .10}` is one.

    Four constants had to move together; each was measured alone first and none was enough on its own
    (best-fixed-mixture out of 24 seeds: U 20, beta 23, W 23, lam 21, all four 10).  The reason is that the
    optimum is set by where each domain runs out of unique text (U) against how fast its eval component still
    improves (beta), routed through the transfer matrix (W) and then weighted (lam).  Narrowing any one of
    them pins the optimum back down through the others.
    """
    g = np.random.default_rng(1000 + seed)
    return {
        "E": [round(1.60 + g.uniform(-.05, .05), 3), round(0.55 + g.uniform(-.04, .04), 3), round(0.70 + g.uniform(-.05, .05), 3)],
        "A": [round(0.62 * g.uniform(.85, 1.15), 3), round(0.40 * g.uniform(.85, 1.15), 3), round(0.55 * g.uniform(.85, 1.15), 3)],
        "alpha": round(g.uniform(0.26, 0.36), 3),
        "B": [round(x, 3) for x in g.uniform(0.9, 1.6, 3)],
        # how fast each eval component still improves with more (effective) tokens
        "beta": [round(x, 3) for x in g.uniform(0.12, 0.60, 3)],
        # transfer: W[i][j] is how much domain j's effective tokens count towards eval set i.  The
        # off-diagonals that decide whether a scarce domain is worth repeating (code->general,
        # papers->general, papers->code, code->math) span an order of magnitude across instances.
        "W": [[1.0, round(g.uniform(.05, .45), 3), round(g.uniform(.06, .14), 3), round(g.uniform(.2, .95), 3)],
              [round(g.uniform(.03, .08), 3), 1.0, round(g.uniform(.02, .30), 3), round(g.uniform(.02, .30), 3)],
              [round(g.uniform(.02, .06), 3), round(g.uniform(.05, .55), 3), 1.0, round(g.uniform(.05, .70), 3)]],
        # unique tokens per domain; published in docs/corpora.md, so this is public and varies per instance
        "U": [3e12, float(g.uniform(8e9, 1.2e11)), float(g.uniform(1.5e9, 2.5e10)), float(g.uniform(4e9, 6e10))],
        "Rs": [15.0, round(g.uniform(1.5, 4), 2), round(g.uniform(0.6, 2.0), 2), round(g.uniform(1.5, 5), 2)],
        # the headline composite's weights, also published; a different weighting moves the optimum even
        # when the loss surface itself is unchanged
        "lam": [round(float(x), 3) for x in g.dirichlet([3.0, 3.0, 3.0])],
    }


# ----------------------------------------------------------------------------------------------- simulator
def deff(n, U, Rs):
    n = np.asarray(n, float); U = np.asarray(U, float); Rs = np.asarray(Rs, float)
    x = np.maximum(n / U - 1, 0)
    return np.minimum(n, U) + U * Rs * (1 - np.exp(-x / Rs))


def losses(p, N, D, r, pool=None, Rs=None):
    """r: (..., 4) mixtures.  Returns (composite[...], per-component[..., 3])."""
    r = np.asarray(r, float)
    U = np.array(p["U"]) if pool is None else np.asarray(pool, float)
    R = np.array(p["Rs"] if Rs is None else Rs, float)
    T = deff(r * D, U, R) @ np.array(p["W"]).T
    Li = np.array(p["E"]) + np.array(p["A"]) * (N / 1e8) ** (-p["alpha"]) + np.array(p["B"]) * (np.maximum(T, 1.0) / 1e9) ** (-np.array(p["beta"]))
    return Li @ np.array(p["lam"]), Li


_G = None


def optimum(p, N=NT, D=DT, pool=None, Rs=None):
    global _G
    if _G is None:
        _G = simplex_grid(4, 0.02)
    L, _ = losses(p, N, D, _G, pool, Rs)
    best = _G[int(np.argmin(L))]
    # local refinement on a 0.005 lattice around the coarse optimum
    for step in (0.01, 0.005, 0.0025):
        cand = [best]
        for a in range(4):
            for b in range(4):
                if a != b:
                    for k in (1, 2):
                        c = best.copy(); c[a] += k * step; c[b] -= k * step
                        if c.min() >= 0:
                            cand.append(c)
        cand = np.array(cand)
        L, _ = losses(p, N, D, cand, pool, Rs)
        best = cand[int(np.argmin(L))]
    return best, float(losses(p, N, D, best, pool, Rs)[0])


def truth(p):
    """Reference points for grading (cached in the world file at build time, recomputed if absent)."""
    r_opt, L_opt = optimum(p)
    r_my, _ = optimum(p, PROXY_N, PROXY_D)
    L_my = float(losses(p, NT, DT, r_my)[0])
    r_rec, _ = optimum(p, Rs=[15.0] * 4)
    L_rec = float(losses(p, NT, DT, r_rec)[0])
    return {"r_opt": r_opt.round(4).tolist(), "L_opt": L_opt, "r_myopic": r_my.round(4).tolist(), "L_myopic": L_my,
            "r_recite": r_rec.round(4).tolist(), "L_recite": L_rec}


# ----------------------------------------------------------------------------------------------- lab ops
def _parse_mix(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            raise LabError("mix must be a JSON object like {\"web\":0.7,\"code\":0.1,\"math\":0.1,\"papers\":0.1}")
    if isinstance(x, (list, tuple)):
        if len(x) != 4:
            raise LabError("mix as a list needs 4 numbers in the order %s" % DOM)
        x = dict(zip(DOM, x))
    if not isinstance(x, dict):
        raise LabError("mix must be an object keyed by %s" % DOM)
    bad = [k for k in x if k not in DOM]
    if bad:
        raise LabError("unknown domain(s) %s; domains are %s" % (bad, DOM))
    r = np.array([num(x.get(k, 0.0), "mix." + k, 0.0, None) for k in DOM])
    if r.sum() <= 0:
        raise LabError("mix weights must not all be zero")
    if abs(r.sum() - 1) > 0.02:
        raise LabError("mix weights must sum to 1 (got %.4f)" % r.sum())
    return r / r.sum()


def _parse_pool(w, x):
    U = np.array(w.p["U"], float)
    if x in (None, "", {}):
        return U.copy()
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            raise LabError("pool must be a JSON object {domain: unique_tokens}")
    if not isinstance(x, dict):
        raise LabError("pool must be an object {domain: unique_tokens}")
    out = U.copy()
    for k, v in x.items():
        if k not in DOM:
            raise LabError("unknown domain %r in pool" % k)
        j = DOM.index(k)
        out[j] = num(v, "pool." + k, 1e6, U[j])
    return out


def _cost_train(w, a):
    N = num(a.get("N"), "N", *N_RANGE); D = num(a.get("D"), "D", *D_RANGE)
    return 6.0 * N * D


def _run_train(w, a, ctx):
    N = num(a.get("N"), "N", *N_RANGE); D = num(a.get("D"), "D", *D_RANGE)
    r = _parse_mix(a.get("mix"))
    pool = _parse_pool(w, a.get("pool"))
    comp, Li = losses(w.p, N, D, r, pool)
    Li = Li + ctx["rng"].normal(0, SIGMA, 3)
    n = r * D
    return {"eval_loss": {k: round(float(v), 5) for k, v in zip(EV, Li)},
            "composite": round(float(Li @ np.array(w.p["lam"])), 5),
            "tokens_seen": {k: float("%.4g" % v) for k, v in zip(DOM, n)},
            "unique_tokens_available": {k: float("%.4g" % v) for k, v in zip(DOM, pool)},
            "epochs": {k: round(float(v), 3) for k, v in zip(DOM, n / pool)}}


class World(_W):
    NAME = "k2_mix"
    ARTIFACTS = ["mixture.json"]
    BUDGET_UNIT = "FLOPs"
    OPS = {"train": (_cost_train, _run_train, "train a proxy model and evaluate it")}

    def public_spec(self):
        return {
            "ops": {"train": {
                "args": {"N": "non-embedding parameters, %.0e..%.0e" % N_RANGE,
                         "D": "training tokens, %.0e..%.0e" % D_RANGE,
                         "mix": "sampling weights over %s (sum to 1)" % DOM,
                         "pool": "optional {domain: unique tokens}: restrict the unique data a run may draw from "
                                 "(default: the full corpus for every domain); tokens beyond the pool are repeats"},
                "cost": "6*N*D FLOPs",
                "returns": "eval_loss per eval set (%s), composite (weighted by %s), tokens_seen / "
                           "unique_tokens_available / epochs per domain" % (EV, dict(zip(EV, self.p["lam"])))}},
            "corpus_unique_tokens": dict(zip(DOM, self.p["U"])),
            "target": {"N": NT, "D": DT},
            "deliverable": "/app/mixture.json",
        }

    def grade(self, art_dir, ledger=None):
        t = self.cfg.get("truth") or truth(self.p)
        items = {}
        art, err = read_json_artifact(art_dir, "mixture.json")
        diag = {}
        r = None; iv = None
        if art is None:
            items["R0_artifact"] = (False, err)
        else:
            try:
                r = _parse_mix(art.get("mixture"))
                iv = art.get("predicted_loss") or {}
                lo, hi = float(iv["lo"]), float(iv["hi"])
                if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                    raise ValueError
                if hi - lo > WIDTH_CAP + 1e-12:
                    items["R0_artifact"] = (False, "predicted_loss interval width %.4f > %.3f" % (hi - lo, WIDTH_CAP))
                else:
                    items["R0_artifact"] = (True, "ok")
            except LabError as e:
                items["R0_artifact"] = (False, "bad mixture: %s" % e)
            except Exception:
                items["R0_artifact"] = (False, "predicted_loss must be {\"lo\": x, \"hi\": y} with lo <= hi")
        s = None
        if r is not None:
            L = float(losses(self.p, NT, DT, r)[0])
            s = (t["L_myopic"] - L) / (t["L_myopic"] - t["L_opt"])
            tol = (1 - TAU) * (t["L_myopic"] - t["L_opt"])
            items["R1_mixture_quality"] = (L - t["L_opt"] <= tol + 1e-12, "regret %.4f (tol %.4f), s=%.3f" % (L - t["L_opt"], tol, s))
            if iv is not None and items["R0_artifact"][0]:
                ok = float(iv["lo"]) <= L <= float(iv["hi"])
                items["R2_forecast_covers"] = (ok, "true target loss %.4f, interval [%.4f, %.4f]" % (L, float(iv["lo"]), float(iv["hi"])))
                diag["forecast_center_err"] = (float(iv["lo"]) + float(iv["hi"])) / 2 - L
            else:
                items["R2_forecast_covers"] = (False, "no valid interval")
            diag["target_epochs"] = dict(zip(DOM, np.round(r * DT / np.array(self.p["U"]), 2).tolist()))
            diag["mixture"] = dict(zip(DOM, np.round(r, 4).tolist()))
        else:
            items["R1_mixture_quality"] = (False, "no valid mixture")
            items["R2_forecast_covers"] = (False, "no valid mixture")
        if ledger is not None:
            recs = [x for x in ledger if x.get("op") == "train"]
            diag["n_train"] = len(recs)
            diag["n_pool_runs"] = sum(1 for x in recs if x["args"].get("pool"))
            diag["max_epochs_explored"] = max([max(x["result"]["epochs"].values()) for x in recs] or [0])
        passed = all(v[0] for v in items.values())
        return {"pass": passed, "score": None if s is None else round(s, 4), "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": {"L_opt": t["L_opt"], "r_opt": t["r_opt"], "L_myopic": t["L_myopic"], "r_myopic": t["r_myopic"]}}


# ----------------------------------------------------------------------------------------------- scripted strategies (gates)
def _rsm_optimum(sess, N, D, pool_fn, n_pts, rng, center=None, radius=0.6, prev=None, conc=12):
    """Response-surface search: evaluate Dirichlet-scattered mixtures, fit a quadratic per eval set, minimise composite."""
    lam = np.array(sess.w.p["lam"])
    X, Y = [], []
    if center is None:
        center = np.array([0.55, 0.15, 0.15, 0.15])
    for k in range(n_pts):
        r = rng.dirichlet(center * conc + 0.3)
        out = sess.call("train", {"N": N, "D": D, "mix": dict(zip(DOM, r.tolist())), "pool": pool_fn(D)})
        X.append(r); Y.append([out["eval_loss"][e] for e in EV])
    X = np.array(X); Y = np.array(Y)
    if prev is not None:
        X = np.vstack([prev[0], X]); Y = np.vstack([prev[1], Y])
    # "data mixing law" surrogate (Ye et al. 2024 form): L_i = c_i + k_i * exp(sum_j t_ij r_j), fitted per eval set
    from ..fit import lm_fit
    G = simplex_grid(4, 0.01)
    pred = np.zeros(len(G))
    for i in range(3):
        y = Y[:, i]
        f = lambda x, Z: x[0] + np.exp(x[1] + Z @ x[2:])
        x0 = [y.min() - 0.3, np.log(0.3), 0, 0, 0, 0]
        x, _ = lm_fit(lambda x: f(x, X) - y, x0, lo=[0, -10, -30, -30, -30, -30], hi=[5, 5, 30, 30, 30, 30])
        pred += lam[i] * f(x, G)
    inside = np.abs(G - center).max(1) <= radius
    pred[~inside] = np.inf
    return G[int(np.argmin(pred))], X, Y


def _forecast(sess, r, pool_fn, rng, lam=None):
    """Per-eval-set separable fit  L_i = c_i + a_i (N/1e8)^-al_i + b_i (D/1e9)^-be_i  from an N sweep and a D sweep
    run at mixture r (with the same pool policy), extrapolated to the target; parametric bootstrap for the width."""
    from ..fit import lm_fit
    lam = np.array(sess.w.p["lam"])
    pts = []
    mix = dict(zip(DOM, r.tolist()))
    for N in (5e7, 1e8, 2e8, 4e8, 5e7, 1e8, 2e8, 4e8):
        o = sess.call("train", {"N": N, "D": 1e9, "mix": mix, "pool": pool_fn(1e9)})
        pts.append([N, 1e9] + [o["eval_loss"][e] for e in EV])
    for D in (2e9, 4e9, 8e9, 1.4e10, 2e10):
        o = sess.call("train", {"N": 5e7, "D": D, "mix": mix, "pool": pool_fn(D)})
        pts.append([5e7, D] + [o["eval_loss"][e] for e in EV])
    P = np.array(pts)

    def model(x, NN, DD):
        c, a, al, b, be = x
        return c + a * (NN / 1e8) ** -al + b * (DD / 1e9) ** -be

    def fit_all(Yc):
        out = 0.0
        for i in range(3):
            y = Yc[:, i]
            x, _ = lm_fit(lambda x: model(x, P[:, 0], P[:, 1]) - y, [y.min() - 0.5, 0.5, 0.3, 0.8, 0.3],
                          lo=[0, 0, 0.05, 0, 0.05], hi=[3, 5, 1.0, 5, 1.0])
            out += lam[i] * float(model(x, NT, DT))
        return out
    Y = P[:, 2:]
    base = fit_all(Y)
    boots = [fit_all(Y + rng.normal(0, SIGMA, Y.shape)) for _ in range(30)]
    sd = float(np.std(boots))
    # Extrapolating 250x in D is the dominant error, not run-to-run noise, so the bootstrap sd is
    # inflated rather than quoted: 2*sd underestimated the true error on 25% of gate pairs.
    half = min(max(3.0 * sd, 0.010), WIDTH_CAP / 2)
    return base, half


def _search(sess, rng, pool_fn):
    r1, X, Y = _rsm_optimum(sess, 5e7, 1e9, pool_fn, 24, rng)
    r2, X, Y = _rsm_optimum(sess, 5e7, 1e9, pool_fn, 16, rng, center=r1, radius=0.2, prev=(X, Y), conc=40)
    return r2


# Cost of the N-sweep and D-sweep `_forecast` runs, in FLOPs: 6*N*D summed over its fixed schedule.  The
# search has to leave this much behind or the oracle dies mid-forecast with an insufficient-budget error
# (it did, on the first run at 40 fit points).
FORECAST_COST = 6 * (sum(N * 1e9 for N in (5e7, 1e8, 2e8, 4e8)) * 2 + sum(5e7 * D for D in (2e9, 4e9, 8e9, 1.4e10, 2e10)))


def _mechanism_fit(sess, pool_fn, rng, n_pts=40):
    """Fit the mechanism family itself - scaling law + repetition decay - from scattered (N, D, mixture) runs.

    This is the search the oracle does, and it replaced a model-free response-surface search because the
    response surface, not the task, was setting the bar.  Measured 2026-09-28 on the same seeds and salt:
    RSM scored 0.77 / 0.89 / 0.93 where this scores 0.96 / 0.999 / 0.997, and a third RSM refinement round
    made it WORSE (0.89 -> 0.87, 0.77 -> 0.64) because a quadratic/exponential surrogate fitted to noisy
    points near the optimum has nothing left to learn from a tighter cluster.  A tolerance calibrated
    against the weaker searcher was measuring the searcher.

    What makes the difference is that the agent already knows the shape: the corpus sizes U are published
    in docs/corpora.md, so every run can be placed in epochs rather than tokens, and the only unknowns left
    per eval set are the transfer row W_i and the repetition decays R*.  Fitting those is exactly the
    reasoning the task is meant to reward, and the constants are still not recoverable without running the
    experiments - a blind fit has nothing to fit.
    """
    U = np.array(sess.w.p["U"])
    X, Y, Ns, Ds = [], [], [], []
    center = np.array([0.55, 0.15, 0.15, 0.15])
    for _ in range(n_pts):
        r = rng.dirichlet(center * 6 + 0.3)
        N = float(rng.choice([5e7, 7e7, 1e8])); D = float(rng.choice([1e9, 1.5e9, 2e9]))
        if 6 * N * D > sess.left() - FORECAST_COST * 1.05:   # leave the forecast sweep its budget
            break
        o = sess.call("train", {"N": N, "D": D, "mix": dict(zip(DOM, r.tolist())), "pool": pool_fn(D)})
        X.append(r); Ns.append(N); Ds.append(D); Y.append([o["eval_loss"][e] for e in EV])
    X = np.array(X); Y = np.array(Y); Ns = np.array(Ns); Ds = np.array(Ds)
    from ..fit import lm_fit

    def pred(x, Xr, Nv, Dv):
        E, A, al, B, be = x[0], x[1], x[2], x[3], x[4]
        W = np.abs(x[5:9]); Rs = np.abs(x[9:13]) + 0.05
        n = Xr * Dv[:, None]
        Uu = U[None, :] * Dv[:, None] / DT          # the replica pools this run actually drew from
        De = np.minimum(n, Uu) + Uu * Rs * (1 - np.exp(-np.maximum(n / Uu - 1, 0) / Rs))
        return E + A * (Nv / 1e8) ** (-al) + B * (np.maximum(De @ W, 1.0) / 1e9) ** (-be)

    G = simplex_grid(4, 0.01)
    tot = np.zeros(len(G)); lam = np.array(sess.w.p["lam"])
    for i in range(3):
        y = Y[:, i]
        x, _ = lm_fit(lambda x: pred(x, X, Ns, Ds) - y,
                      [y.min() - 0.4, 0.5, 0.30, 1.0, 0.3, 0.2, 0.2, 0.2, 0.2, 5, 3, 1.5, 3],
                      lo=[0, 0, 0.1, 0.01, 0.05] + [0.0] * 4 + [0.3] * 4,
                      hi=[3, 3, 0.6, 5, 0.9] + [1.5] * 4 + [30] * 4)
        tot += lam[i] * pred(x, G, np.full(len(G), NT), np.full(len(G), DT))
    # The forecast is deliberately NOT taken from this fit.  Tried and measured 2026-09-28: predicting the
    # target loss from the same fitted mechanism was biased by +0.23 to +0.60 on 8 of 10 (seed, salt) pairs
    # - far outside any admissible interval - while the separate `_forecast` sweep was biased by +0.001 to
    # +0.010 on the same pairs.  The mechanism fit is good at RANKING mixtures (its job here) and bad at
    # absolute level, because W, B and lam trade off against each other: a fit can get every comparison
    # right with the whole surface shifted.  Choosing the mixture and forecasting its loss are different
    # estimation problems on the same data, and the task asks for both on purpose.
    return G[int(np.argmin(tot))]


def strat_oracle(sess, art_dir, rng):
    """Existence proof: replicate the target's repetition regime at small scale (pools scaled by D/D_target)."""
    U = np.array(sess.w.p["U"])
    pool_fn = lambda D: dict(zip(DOM, np.maximum(U * D / DT, 1e6).tolist()))
    r = _mechanism_fit(sess, pool_fn, rng)
    c, h = _forecast(sess, r, pool_fn, rng, None)
    _write(art_dir, r, c - h, c + h)


def strat_oracle_rsm(sess, art_dir, rng):
    """The former oracle, kept as a `fail` strategy: same replica pools, model-free response-surface search.

    Demoting it is the point.  It does everything the task asks conceptually - it reproduces the target's
    repetition regime at proxy scale - and still lands 5-25 points short of the achievable optimum, so once
    TAU is set by what a mechanism-aware fit can reach, "had the right idea but fitted a generic surface"
    is no longer a pass.  It also guards the bar from below: if a future edit makes the task easy enough
    for the blind surrogate again, this fails the gate loudly.
    """
    U = np.array(sess.w.p["U"])
    pool_fn = lambda D: dict(zip(DOM, np.maximum(U * D / DT, 1e6).tolist()))
    r = _search(sess, rng, pool_fn)
    c, h = _forecast(sess, r, pool_fn, rng, None)
    _write(art_dir, r, c - h, c + h)


def strat_myopic(sess, art_dir, rng):
    """Competent but myopic: same search and forecast, but with the full corpus at proxy scale (no repetition)."""
    pool_fn = lambda D: None
    r = _search(sess, rng, pool_fn)
    c, h = _forecast(sess, r, pool_fn, rng, None)
    _write(art_dir, r, c - h, c + h)


def strat_recite(sess, art_dir, rng):
    """Knows every hidden constant except R*: uses the literature value R*=15 everywhere; perfect forecast."""
    r, _ = optimum(sess.w.p, Rs=[15.0] * 4)
    L = float(losses(sess.w.p, NT, DT, r)[0])
    _write(art_dir, r, L - 0.01, L + 0.01)


# Mixtures a competent practitioner can write down from prior knowledge alone, before touching the lab.
# `pool_gate` holds the shipped instances to the rule that none of these may clear R1 more often than a
# random mixture does; the names are only for the gate report.
PRIOR_MIXES = {"llama_ish": [0.67, 0.15, 0.04, 0.14], "even_ish": [0.55, 0.25, 0.08, 0.12],
               "code_heavy": [0.45, 0.35, 0.08, 0.12], "web_dominant": [0.80, 0.10, 0.04, 0.06],
               "quality_heavy": [0.40, 0.25, 0.10, 0.25]}


def strat_prior_mixture(sess, art_dir, rng):
    """Skip the search entirely: submit a standard pre-training mixture from prior knowledge, then spend the
    WHOLE budget forecasting that one mixture honestly (with the replica pools, as the oracle does).

    This is the strategy the task has to beat, and it is not a cheat: R0 and R2 are satisfied on the merits -
    the forecast really is measured, it is only the mixture that was never searched for.  Everything rides on
    R1.  Measured 2026-09-28 on the pool as it then stood, the best single fixed mixture cleared R1 on 18 of
    19 gated seeds at TAU=0.80, and `even_ish` alone cleared 14 of 19 - so a model that had merely seen a
    sensible mixture before scored like one that had done the experiment.  `pool_gate` is what keeps that
    from being true of the instances actually shipped; this strategy is what checks it per instance.
    """
    U = np.array(sess.w.p["U"])
    pool_fn = lambda D: dict(zip(DOM, np.maximum(U * D / DT, 1e6).tolist()))
    r = np.array(PRIOR_MIXES[sorted(PRIOR_MIXES)[int(rng.integers(len(PRIOR_MIXES)))]])
    c, h = _forecast(sess, r, pool_fn, rng, None)
    _write(art_dir, r, c - h, c + h)


def strat_natural(sess, art_dir, rng):
    U = np.array(sess.w.p["U"]); r = U / U.sum()
    L = float(losses(sess.w.p, NT, DT, r)[0])
    _write(art_dir, r, L - 0.01, L + 0.01)


def strat_cheat_wide(sess, art_dir, rng):
    r, _ = optimum(sess.w.p)
    _write(art_dir, r, 0.0, 10.0)


def strat_nop(sess, art_dir, rng):
    pass


def _write(art_dir, r, lo, hi):
    import os
    json.dump({"mixture": dict(zip(DOM, [float(x) for x in r])), "predicted_loss": {"lo": float(lo), "hi": float(hi)}},
              open(os.path.join(art_dir, "mixture.json"), "w"), indent=1)


STRATEGIES = {"oracle": (strat_oracle, "pass"), "oracle_rsm": (strat_oracle_rsm, "fail"),
              "myopic": (strat_myopic, "fail"), "recite": (strat_recite, "fail"),
              "natural": (strat_natural, "fail"), "prior_mixture": (strat_prior_mixture, "fail"),
              "cheat_wide": (strat_cheat_wide, "fail"), "nop": (strat_nop, "fail")}
# `prior_mixture` draws its mixture from a menu, so on the occasional instance one of the five happens to
# land inside the accepted region and it passes.  That is the guess being lucky, not a leak, and it is held
# to a rate rather than to zero - the same treatment k8's random-guess decoys get.
NOISY_FAIL = ("prior_mixture",)
BUDGET = 5.0e19


def instance_gate(p):
    """Cheap, noiseless screens used to pick seeds before the (noisy) strategy gates."""
    t = truth(p)
    U = np.array(p["U"])
    r_rep, _ = optimum(p, 5e7, 2e9, pool=U * 2e9 / DT)
    L_rep = float(losses(p, NT, DT, r_rep)[0])
    s_rep = (t["L_myopic"] - L_rep) / (t["L_myopic"] - t["L_opt"])
    s_rec = (t["L_myopic"] - t["L_recite"]) / (t["L_myopic"] - t["L_opt"])
    gap = t["L_myopic"] - t["L_opt"]
    ok = gap >= 0.035 and s_rec <= 0.75 and s_rep >= 0.92
    return ok, {"gap": round(gap, 4), "s_recite": round(s_rec, 3), "s_replica": round(s_rep, 3), **t}


def pool_gate(seeds, tol=None):
    """Pool-level screen: no single fixed mixture may clear R1 across the instances actually shipped.

    K2's per-instance gates cannot see this defect, and did not.  Each gated seed had a well-identified
    optimum, a real gap over the myopic answer, and decoys that all failed - and yet ONE mixture, searched
    over the whole simplex against the pool and submitted blind with no lab calls at all, cleared R1 on 18
    of 19 of them (measured 2026-09-28).  Whether the answer moves between instances is a property of the
    POOL, exactly as cause-balance is in k8; a per-instance screen is structurally unable to check it.

    Two rates are reported.  `best_fixed` is an upper bound computed WITH the answers (the strongest
    mixture for this specific pool, which no agent could find) and is the honest worst case.  `prior` is
    what a model can actually do from prior knowledge, using the same menu `strat_prior_mixture` plays.
    Both are compared against `chance`: the rate a mixture drawn at random from the simplex clears R1,
    which is not zero because the accepted region has real volume.

    Returns (ok, info).
    """
    ps = [sample_params(sd) for sd in seeds]
    ts = [truth(p) for p in ps]
    tol_i = [(1 - TAU) * (t["L_myopic"] - t["L_opt"]) for t in ts]

    def clears(r):
        out = []
        for p, t, tl in zip(ps, ts, tol_i):
            out.append(float(losses(p, NT, DT, np.asarray(r, float))[0]) - t["L_opt"] <= tl + 1e-12)
        return np.array(out)

    G = simplex_grid(4, 0.02)
    cnt = np.zeros(len(G))
    for p, t, tl in zip(ps, ts, tol_i):
        L, _ = losses(p, NT, DT, G)
        cnt += (L - t["L_opt"] <= tl + 1e-12)
    n = len(ps)
    best_i = int(np.argmax(cnt))
    best_fixed = float(cnt[best_i]) / n
    prior = {nm: float(clears(r).mean()) for nm, r in PRIOR_MIXES.items()}
    # chance = mean accepted volume of the simplex, i.e. how often a blind random mixture would clear R1
    chance = float(np.mean([((losses(p, NT, DT, G)[0] - t["L_opt"]) <= tl + 1e-12).mean()
                            for p, t, tl in zip(ps, ts, tol_i)]))
    worst_prior = max(prior.items(), key=lambda x: x[1])
    # `best_fixed` can never go below 1/n: each instance's own optimum clears that instance.  So the bar is
    # not a tolerance around chance, it is the crisp property "NO single mixture serves two instances" -
    # i.e. best_fixed at the 1/n floor.  Stating it as `chance + 0.25` would have been an arbitrary number
    # that happened to pass; this one says what is actually required and cannot be met by shrinking n.
    # The prior menu is held to the same bar: knowing one standard mixture may serve at most one instance.
    floor = 1.0 / n
    ok = bool(best_fixed <= floor + 1e-9 and worst_prior[1] <= floor + 1e-9)
    return ok, {"n": n, "chance": round(chance, 4), "floor": round(floor, 4), "best_fixed": round(best_fixed, 4),
                "best_fixed_mix": np.round(G[best_i], 3).tolist(), "worst_prior": (worst_prior[0], round(worst_prior[1], 4)),
                "prior": {k: round(v, 4) for k, v in prior.items()}}


def diverse_pool(seeds, k=None, min_sep=0.18):
    """Pick a subset of gated seeds whose optima are spread out, so no one mixture can serve them all.

    Greedy max-min over r_opt in L1: take the seeds in order, keeping one only if its optimum is at least
    `min_sep` away from every optimum already kept.  Selection-time, deterministic, and for the same reason
    as k8's `balanced_pool`: the property is about what ships, so it is imposed on what ships.
    """
    kept, kept_r = [], []
    for sd in seeds:
        r = np.array(truth(sample_params(sd))["r_opt"], float)
        if all(np.abs(r - q).sum() >= min_sep for q in kept_r):
            kept.append(sd); kept_r.append(r)
        if k and len(kept) >= k:
            break
    return kept


# ----------------------------------------------------------------------------------------------- agent-facing text
def instruction(p, t):
    tol = (1 - TAU) * (t["L_myopic"] - t["L_opt"])
    return f"""# Choose the data mixture for a data-constrained pre-training run

You are the data lead for a pre-training run that will be launched exactly once with a fixed shape:
a **2.5B-parameter** (non-embedding) model trained on **500B tokens**. The training corpus has four domains -
`web`, `code`, `math`, `papers` - each with a fixed, finite amount of unique text (see `/app/docs/corpora.md`).
Your job: choose the sampling mixture for that run, and forecast its final composite evaluation loss so the
planning team can budget follow-up work.

You cannot run the target. You can run small proxy trainings in a simulated lab. Run `/app/bin/lab spec`
first. Each `train` call trains a fresh model with N parameters on D tokens using your mixture and returns its
held-out losses on three eval sets (general, code, math) and their weighted composite. Like real training runs,
proxy results carry run-to-run noise. Proxy compute is limited: {BUDGET:.0e} FLOPs in total, a run costs 6*N*D.
Example:

    /app/bin/lab train N=1e8 D=2e9 --json '{{"mix": {{"web": 0.7, "code": 0.1, "math": 0.1, "papers": 0.1}}}}'

## Deliverable: `/app/mixture.json`

    {{"mixture": {{"web": 0.0, "code": 0.0, "math": 0.0, "papers": 0.0}},
     "predicted_loss": {{"lo": 0.0, "hi": 0.0}}}}

* `mixture` - sampling weights (non-negative, sum to 1). The target run draws `mixture[d] * 500B` tokens from
  domain d; if that is more than the domain's unique tokens, the domain's data is repeated (several epochs).
* `predicted_loss` - an interval for the target run's final composite eval loss (its expected value, without
  run-to-run noise) *for the mixture you submit*. Width at most {WIDTH_CAP}.

## How it is graded (mechanically, against the simulator; only the file counts)

1. `mixture.json` is valid (weights, interval, width <= {WIDTH_CAP}).
2. The target run's composite loss with your mixture is within **{tol:.4f}** of the lowest composite loss any
   mixture can reach for this target run.
3. Your interval contains the target run's composite loss with your mixture.

All three must hold. Your lab history is not graded; spend the budget however you think is best. When you are
done, make sure `/app/mixture.json` is written, then stop.
"""


def docs(p):
    U = dict(zip(DOM, p["U"]))
    return {"docs/corpora.md": f"""# Training corpora (unique tokens after dedup)

| domain | unique tokens | notes |
|---|---|---|
| web    | {U['web']:.1e} | filtered CommonCrawl-style web text; effectively inexhaustible at our scale |
| code   | {U['code']:.1e} | permissively licensed source code, deduplicated at file level |
| math   | {U['math']:.1e} | math web pages, textbooks and worked solutions |
| papers | {U['papers']:.1e} | scientific papers (full text) |

Eval sets (held out, never trained on): `general` (broad web/knowledge text), `code` (code completion),
`math` (math problems with solutions).  The team's headline number is the composite
`{p['lam'][0]} * general + {p['lam'][1]} * code + {p['lam'][2]} * math`.

The target run: 2.5B non-embedding parameters, 500B training tokens, single run, standard recipe
(the proxies use the same recipe at smaller size).
"""}


def hints(p):
    return {1: "\n## Hint\nAsk whether your proxy runs face the same conditions the target run will face.\n",
            2: "\n## Hint\nAt 500B tokens the scarce domains will be repeated for many epochs; proxy runs at a few "
               "billion tokens never repeat anything unless you make them. The `pool` argument of `train` limits the "
               "unique tokens a run may draw from.\n"}
