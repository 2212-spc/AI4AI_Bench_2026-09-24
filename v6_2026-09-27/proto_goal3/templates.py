"""Assumption-violation (AV) templates.  Each turns (card, world th, design d, seed) into an *item*:

  truth        the correct answer, computed from the world
  ref(rng, R)  the reference estimator executed R times against a noisy lab with the item's run budget
  kappa        l1 norm of the reference's linearised weights on observed means, in units where the answer's
               own primary observable has weight 1 (1 = interpolate / average / select; >1 = extrapolation-like
               amplification; the v5 law says a hard item needs kappa ~ 1)
  routes       biased shortcut routes (noiseless), scored by the certificate B
  red          extra routes a red team adds (scored in G9; an item certified without them may be a false positive)
  reference    route names that ARE the intended derivation (excluded from B, as in routes.certify)
  valid        consistent alternative methods (NOT shortcuts; info only - scoring them as rivals would be a bug)
  free         the honest method applied to data the item gives away for free (G7 no-free-readout: >= 2T)
  shown        salient numbers the task shows in the answer's units (G7 also requires each >= 1T from the truth:
               an agent that copies a shown number must not be graded correct)
  alts         alternative literal readings {name: (value, clause)}; G8 adds the clause when value is > T away
  census_rule  'G18' (>=2 numerical + >=1 structural family) for off-grid targets, 'AV' otherwise
  av_fams      the route families that ARE the template's assumption violation; census requires one of them

Templates (codes follow the v6 error analysis, E_synthesis.md):
  M1 ctx_transfer     validity domain: "holds where I measured => holds where asked"
  M2 censored_units   partial information: "not point-identified / not observed => no information"
  M3 winners_curse    error budget: "the notebook's best number is the number"
  M6 floor_fraction   reference point: "the baseline is at zero / at its asymptote"
"""
import numpy as np

from cards import Y, S
from routes_v5 import Estimand, numerical_routes

TEMPLATES = ("M1", "M2", "M3", "M6")


def _span(card):
    return card.hi - card.lo


# ================================================================================================ M1
def prior_M1(card, rng):
    sp = _span(card)
    return dict(xs=float(rng.uniform(card.lo + 0.15 * sp, card.hi - 0.15 * sp)),
                h=float(rng.uniform(0.05, 0.25) * sp), u=float(rng.uniform(0.25, 0.75)),
                n=int(rng.choice([40, 80, 160])), r_nb=float(rng.choice([1.0, 1.5, 2.0])))


def _m1_grid(xs, u, h):
    xl, xr = xs - u * h, xs + (1 - u) * h
    return xl, xr, [xl - h, xl, xr, xr + h]


def _m1_num(card, th, xs, u, h):
    _, _, grid = _m1_grid(xs, u, h)
    y1 = lambda t: Y(card, t, th, 1)
    return numerical_routes(Estimand(y1, grid, xs, y1(xs), name="M1"))


def build_M1(card, th, d, seed):
    xs, h, u, n, r_nb = d["xs"], d["h"], d["u"], d["n"], d["r_nb"]
    xl, xr, grid = _m1_grid(xs, u, h)
    if grid[0] < card.lo - 1e-9 or grid[-1] > card.hi + 1e-9:
        return None, "design outside the card's axis"
    y0 = lambda t: Y(card, t, th, 0)
    y1 = lambda t: Y(card, t, th, 1)
    truth = y1(xs)
    wl, wr = 1 - u, u
    sl, sr = S(card, xl, th), S(card, xr, th)
    nl = int(np.clip(round(n * wl * sl / (wl * sl + wr * sr)), 2, n - 2))
    nr = n - nl
    y1l, y1r = y1(xl), y1(xr)

    def ref(rng, R):
        return wl * (y1l + sl / np.sqrt(nl) * rng.standard_normal(R)) + wr * (y1r + sr / np.sqrt(nr) * rng.standard_normal(R))

    # Route-order test.  The target sits INSIDE the bought bracket, so some "numerical shortcuts" are merely
    # other interpolators: their error is O(h^2) curvature, the same order and constant as the reference's own
    # remainder, so no tolerance the reference passes can separate them from it (B would be capped near the
    # reference's bias share).  Halve the bracket: a route whose bias shrinks ~4x (order >= 1.5) is
    # reference-equivalent and is moved to `valid`; a route whose bias shrinks ~2x (order ~1: read-off, endpoint,
    # evaluate-at-centre) never interpolated and stays a shortcut.
    # R7:plugin needs a production run at the bracket centre, which the lab does not sell -> not executable here
    num_h = {k: v for k, v in _m1_num(card, th, xs, u, h).items() if k != "R7:plugin"}
    num_h2 = _m1_num(card, th, xs, u, h / 2)
    routes, valid, order = {}, {}, {}
    for k, v in num_h.items():
        b1, b2 = abs(v - truth), abs(num_h2.get(k, np.nan) - truth)
        p = float(np.log2(b1 / b2)) if b1 > 0 and b2 > 0 else float("inf")
        order[k] = round(p, 2)
        (valid if p >= 1.5 else routes)[("valid:" + k) if p >= 1.5 else k] = v
    routes["R5:two_point"] = num_h["R5:two_point"]          # the reference itself (excluded from B by name)
    valid.pop("valid:R5:two_point", None)
    xn = xl if u <= 0.5 else xr
    routes["B_prior:ctx0"] = y0(xs)
    # red team: three more ways to carry the notebook's context into production
    h_nb = _span(card) / 12
    xx = np.array([xs - 1.5 * h_nb, xs - 0.5 * h_nb, xs + 0.5 * h_nb, xs + 1.5 * h_nb, xl, xr])
    yy = np.array([y0(t) for t in xx[:4]] + [y1l, y1r])
    # tier 2 (red team): calibrated transfers - use the notebook's curve plus the bought production point(s),
    # under an assumption about HOW the context changes the curve (constant offset / ratio / slope / pooled fit)
    red = {"naive_ignore:ctx:offset@near": y0(xs) + y1(xn) - y0(xn),
           "naive_ignore:ctx:ratio@near": y0(xs) * y1(xn) / y0(xn),
           "naive_ignore:ctx:pool": float(np.polyval(np.polyfit(xx, yy, 2), xs)),
           "R3:ctx0_slope@near": y1(xn) + (y0(xn + 1e-4) - y0(xn - 1e-4)) / 2e-4 * (xs - xn)}
    valid["valid:control_variate"] = y0(xs) + wl * (y1l - y0(xl)) + wr * (y1r - y0(xr))
    r = np.random.default_rng(seed + 17)
    pl, pr = y1l + r_nb * sl * r.standard_normal(), y1r + r_nb * sr * r.standard_normal()
    free = {"free:pilot_interp": wl * pl + wr * pr}
    shown = {"shown:pilot@xl": pl, "shown:pilot@xr": pr}
    spec = dict(grid=[round(g, 3) for g in grid], target=round(xs, 3), n=n, route_order=order,
                notebook="ctx0 dense sweep (13 pts, 3-seed means) + one production-context pilot run at each bracket "
                "point (noise x%.1f)" % r_nb)
    return dict(truth=truth, ref=ref, kappa=wl + wr, routes=routes, red=red, reference=["R5:two_point"],
                valid=valid, free=free, shown=shown, alts={}, census_rule="G18", av_fams=("B_prior", "naive_ignore"),
                scale="rel", n=n, sd_ideal=S(card, xs, th) / np.sqrt(n), spec=spec), None


def repair_M1(card, th, d, res, rng):
    g, near = res["first_fail"], res.get("nearest") or ""
    d = dict(d)
    sp = _span(card)
    if g in ("G7_free",):
        if d["n"] * 2 <= 400:
            d["n"] *= 2
            return d, "free pilot readout within 2T -> demand more precision (2x runs, T shrinks)"
        d["r_nb"] = d["r_nb"] * 2
        return d, "free pilot readout within 2T and n capped -> make the notebook's production pilots noisier (x2)"
    if g == "G3_oracle_bias":
        d["h"] = d["h"] * 0.6
        return d, "interpolation remainder dominates T -> narrow the bracket (h x0.6)"
    if g in ("G5_B", "G9_redteam"):
        if near.startswith(("B_prior", "naive_ignore", "R3:ctx0")) or g == "G9_redteam":
            # the context effect is too small at xs: move the target to where the notebook/production gap is largest
            xsg = np.linspace(card.lo + 0.15 * sp, card.hi - 0.15 * sp, 41)
            gap = np.array([abs(Y(card, t, th, 1) - Y(card, t, th, 0)) for t in xsg])
            d["xs"] = float(xsg[int(np.argmax(gap))] + rng.normal(0, 0.02 * sp))
            return d, "context-transfer route nearest -> move target to the largest-context-gap region"
        if res["noise_limited"] and d["n"] * 2 <= 400:
            d["n"] *= 2
            return d, "noise-limited (oracle bias share < 0.5) -> buy 2x runs"
        d["h"] = d["h"] * 1.5
        return d, "numerical route %s nearest and n capped -> widen bracket (h x1.5)" % near
    if g == "G4_tight" and d["n"] * 2 <= 400:
        d["n"] *= 2
        return d, "T too loose -> 2x runs"
    return None, "no repair rule for %s" % g


# ================================================================================================ M2
def prior_M2(card, rng):
    sp = _span(card)
    K = int(rng.choice([7, 9, 11]))
    xs = float(rng.uniform(card.lo + 0.2 * sp, card.hi - 0.1 * sp))
    return dict(K=K, m=int(rng.integers(1, K // 2)), xs=xs,
                x_nb=float(max(card.lo, xs - rng.uniform(0.1, 0.3) * sp)),
                n=int(K * rng.choice([5, 11, 21])))


def build_M2(card, th, d, seed):
    K, m, xs, x_nb, n = d["K"], d["m"], d["xs"], d["x_nb"], d["n"]
    if not (1 <= m < K / 2) or K % 2 == 0:
        return None, "censored count must leave the median identified"
    r = np.random.default_rng(seed)
    q, q2 = r.standard_normal(K), r.standard_normal(K)
    v = np.array([float(card.unit_y(xs, th, q[j], q2[j])) for j in range(K)])
    vnb = np.array([float(card.unit_y(x_nb, th, q[j], q2[j])) for j in range(K)])
    vs = np.sort(v)
    dirn = card.censor_dir
    if dirn > 0:
        c = 0.5 * (vs[K - m - 1] + vs[K - m])
        cens = v > c
    else:
        c = 0.5 * (vs[m - 1] + vs[m])
        cens = v < c
    truth = float(np.median(v))
    nu = max(3, n // K)
    nu -= (nu % 2 == 0)
    sg = S(card, xs, th)

    def ref(rng, R):
        obs = v[None, :, None] + sg * rng.standard_normal((R, K, nu))
        obs = np.where(dirn * (obs - c) > 0, dirn * np.inf, obs)
        return np.median(np.median(obs, axis=2), axis=1)

    unc = ~cens
    routes = {"drop:censored": float(np.median(v[unc])),
              "B_guess:mean_uncensored": float(np.mean(v[unc])),
              "B_guess:threshold": float(c),            # "the median sits at the censoring threshold" (added after
              "R1:notebook_config": float(np.median(vnb))}   # rendering examples: c is shown and can sit next to the median)
    red = {"naive_ignore:unit_slope": float(np.median(vnb) + np.median(v[unc] - vnb[unc])),
           "drop:censored_mean_nb_shift": float(np.mean(vnb) + np.mean(v[unc] - vnb[unc]))}
    vc = np.where(cens, c, v)
    s_idx = np.argsort(v)
    trim = v[s_idx[m:K - m]] if dirn > 0 else v[s_idx[m:K - m]]
    valid = {"valid:censor_as_c": float(np.median(vc)), "valid:symmetric_trim": float(np.median(trim))}
    clause = "the median is over all %d units, including those logged only as a bound" % K
    spec = dict(K=K, m_censored=int(cens.sum()), c=round(float(c), 4), target=round(xs, 3), n=nu * K, runs_per_unit=nu,
                notebook="all %d units measured at the older config x=%.2f" % (K, x_nb))
    return dict(truth=truth, ref=ref, kappa=1.0, routes=routes, red=red, reference=[], valid=valid, free={},
                shown={"shown:threshold_c": float(c), "shown:notebook_median": float(np.median(vnb))},
                alts={"alt:completed_units_only": (routes["drop:censored"], clause)}, census_rule="AV",
                av_fams=("drop",), scale="rel", n=nu * K, sd_ideal=sg / np.sqrt(nu * K), spec=spec), None


def repair_M2(card, th, d, res, rng):
    g, near = res["first_fail"], res.get("nearest") or ""
    d = dict(d)
    sp = _span(card)
    if g in ("G5_B", "G9_redteam"):
        if near.startswith("R1") or near.startswith("naive_ignore:unit_slope") or near.startswith("drop:censored_mean"):
            d["x_nb"] = float(max(card.lo, d["x_nb"] - 0.15 * sp))
            return d, "notebook-config route nearest -> make the notebook's config staler"
        if near.startswith(("drop", "B_guess")) and d["m"] + 1 < d["K"] / 2:
            d["m"] += 1
            return d, "drop-censored route nearest -> censor one more unit (m+1)"
        if res["noise_limited"] and d["n"] * 2 <= 400:
            d["n"] = d["n"] * 2
            return d, "noise-limited -> 2x runs"
        if d["K"] + 2 <= 13:
            d["K"] += 2
            d["n"] = int(d["n"] * (d["K"]) / (d["K"] - 2))
            return d, "m capped -> 2 more units (K+2)"
    if g == "G4_tight" and d["n"] * 2 <= 400:
        d["n"] *= 2
        return d, "T too loose -> 2x runs"
    return None, "no repair rule for %s" % g


# ================================================================================================ M3
def prior_M3(card, rng):
    sp = _span(card)
    return dict(G=int(rng.choice([6, 8, 12])), r_nb=float(rng.choice([1.0, 1.5, 2.0])),
                x0=float(rng.uniform(card.lo + 0.3 * sp, card.hi - 0.1 * sp)), n=int(rng.choice([20, 40, 80])))


def build_M3(card, th, d, seed):
    G, r_nb, x0, n = d["G"], d["r_nb"], d["x0"], d["n"]
    r = np.random.default_rng(seed)
    delta = th["tau_r"] * r.standard_normal(G)
    delta[0] = 0.0                                   # recipe 0 is the incumbent, whose value is published
    y00 = Y(card, x0, th)
    tj = y00 * np.exp(delta) if card.recipe_rel else y00 + delta   # recipe effects are multiplicative on step counts
    sg = S(card, x0, th)
    snb = r_nb * sg
    nb = tj + snb * r.standard_normal(G)
    sel = int(np.argmin(nb) if card.better == "min" else np.argmax(nb))
    truth = float(tj[sel])

    def ref(rng, R):
        return truth + sg / np.sqrt(n) * rng.standard_normal(R)

    routes = {"B_guess:notebook_value": float(nb[sel]), "R7:menu_mean": float(np.mean(nb))}
    if sel != 0:
        routes["R1:incumbent_value"] = float(tj[0])
    srt = np.sort(nb)
    top2 = srt[:2] if card.better == "min" else srt[-2:]
    sgn = 1.0 if card.better == "min" else -1.0
    red = {"R7:top2_mean": float(np.mean(top2)), "B_guess:notebook_plus_1se": float(nb[sel] + sgn * snb)}
    mb = float(np.mean(nb))
    ss = float(np.sum((nb - mb) ** 2))
    shrink = max(0.0, 1.0 - (G - 3) * snb ** 2 / ss) if ss > 0 else 0.0
    free = {"free:james_stein": mb + (nb[sel] - mb) * shrink}
    # the incumbent's value is published (that is what makes R1:incumbent_value a route).  Found by rendering the
    # certified examples: when the notebook picks the incumbent (sel == 0) the answer is this published number.
    shown = {"shown:incumbent_published": float(tj[0]), "shown:notebook_best": float(nb[sel])}
    best = float(np.min(tj) if card.better == "min" else np.max(tj))
    clause = "the true value of the recipe the notebook selected, not of the best recipe on the menu"
    spec = dict(G=G, selected=sel, x0=round(x0, 3), n=n, notebook="one run per recipe, eval noise sd %.4f (stated)" % snb)
    return dict(truth=truth, ref=ref, kappa=1.0, routes=routes, red=red, reference=[], valid={}, free=free,
                shown=shown, alts={"alt:best_recipe": (best, clause)}, census_rule="AV", av_fams=("B_guess",), scale="rel",
                n=n, sd_ideal=sg / np.sqrt(n), spec=spec), None


def repair_M3(card, th, d, res, rng):
    g, near = res["first_fail"], res.get("nearest") or ""
    d = dict(d)
    if g == "G7_free" or g == "G5_B" or g == "G9_redteam":
        if res["noise_limited"] and d["n"] * 2 <= 400:
            d["n"] *= 2
            return d, "%s with noise-limited T -> 2x runs" % g
        if d["r_nb"] < 6:
            d["r_nb"] *= 1.5
            return d, "%s and n capped -> noisier notebook eval (r_nb x1.5)" % g
    if g == "G4_tight" and d["n"] * 2 <= 400:
        d["n"] *= 2
        return d, "T too loose -> 2x runs"
    return "RESAMPLE", "world-driven failure -> resample world"


# ================================================================================================ M6
def prior_M6(card, rng):
    sp = _span(card)
    xa = float(rng.uniform(card.lo, card.lo + 0.35 * sp))
    xb = float(xa + rng.uniform(0.15, 0.4) * sp)
    return dict(xa=xa, xb=xb, x_top=float(rng.uniform(xb + 0.1 * sp, card.hi)), n=int(rng.choice([40, 80, 160])),
                r_nb=float(rng.choice([1.0, 2.0])), dt=float(rng.uniform(0.4, 1.2)))


def build_M6(card, th, d, seed):
    if card.floor_kind is None:
        return None, "N/A: card declares no physical floor"
    xa, xb, xt, n, r_nb = d["xa"], d["xb"], d["x_top"], d["n"], d["r_nb"]
    if not (card.lo - 1e-9 <= xa < xb < xt <= card.hi + 1e-9):
        return None, "design outside the card's axis"
    F = card.floor(th)
    ya, yb, yt = Y(card, xa, th), Y(card, xb, th), Y(card, xt, th)
    truth = (ya - yb) / (ya - F)
    sa, sb = S(card, xa, th), S(card, xb, th)
    den = ya - F
    wa, wb = (1 - truth), 1.0
    r = np.random.default_rng(seed + 31)
    if card.floor_kind == "probe":
        sF = S(card, xt, th)
        wF = truth
        tot = wa * sa + wb * sb + wF * sF
        na = max(2, int(round(n * wa * sa / tot)))
        nb_ = max(2, int(round(n * wb * sb / tot)))
        nF = max(2, n - na - nb_)

        def ref(rng, R):
            A = ya + sa / np.sqrt(na) * rng.standard_normal(R)
            Bv = yb + sb / np.sqrt(nb_) * rng.standard_normal(R)
            Fh = F + sF / np.sqrt(nF) * rng.standard_normal(R)
            return (A - Bv) / (A - Fh)

        kappa = wa + wb + wF
        grid = [xa, xb, xt]
        nbA, nbB, nbF = ya + r_nb * sa * r.standard_normal(), yb + r_nb * sb * r.standard_normal(), F + r_nb * sF * r.standard_normal()
        free = {"free:notebook_probe": (nbA - nbB) / (nbA - nbF)}
    else:   # 'inv': F = intercept of y on z = 2**-x over the three largest sold configs
        dt = d.get("dt", 0.5)                            # spacing of the three top configs (log2 batch units)
        tops = [xt - 2 * dt, xt - dt, xt]
        if tops[0] <= xb:
            return None, "top points overlap the comparison configs"
        z = 2.0 ** -np.array(tops)
        st = np.array([S(card, t, th) for t in tops])
        yts = np.array([Y(card, t, th) for t in tops])
        Xd = np.stack([np.ones(3), z], 1)
        # intercept weights of OLS (equal runs per top point)
        wint = np.linalg.pinv(Xd)[0]
        l1 = float(np.sum(np.abs(wint)))
        wF = truth
        tot = wa * sa + wb * sb + wF * l1 * float(np.mean(st))
        na = max(2, int(round(n * wa * sa / tot)))
        nb_ = max(2, int(round(n * wb * sb / tot)))
        nt = max(2, (n - na - nb_) // 3)

        def ref(rng, R):
            A = ya + sa / np.sqrt(na) * rng.standard_normal(R)
            Bv = yb + sb / np.sqrt(nb_) * rng.standard_normal(R)
            Ym = yts[None, :] + st[None, :] / np.sqrt(nt) * rng.standard_normal((R, 3))
            Fh = Ym @ wint
            return (A - Bv) / (A - Fh)

        kappa = wa + wb + wF * l1
        grid = [xa, xb] + tops
        nbA, nbB = ya + r_nb * sa * r.standard_normal(), yb + r_nb * sb * r.standard_normal()
        nbT = yts + r_nb * st * r.standard_normal(3)
        free = {"free:notebook_regress": (nbA - nbB) / (nbA - float(nbT @ wint))}
    routes = {"naive_ignore:floor": (ya - yb) / ya, "R8:saturated": (ya - yb) / (ya - yt)}
    if card.nominal_floor is not None:
        routes["B_prior:nominal_floor"] = (ya - yb) / (ya - card.nominal_floor)
    xm = 0.5 * (xb + xt)
    ym = Y(card, xm, th)
    Fsec = yt - (ym - yt)
    red = {"R4:secant_floor": (ya - yb) / (ya - Fsec)}
    spec = dict(xa=round(xa, 3), xb=round(xb, 3), grid=[round(g, 3) for g in grid], n=n,
                notebook="one run at xa, xb and one floor-estimate run (noise x%.1f)" % r_nb)
    return dict(truth=truth, ref=ref, kappa=kappa, routes=routes, red=red, reference=[], valid={}, free=free,
                shown={}, alts={}, census_rule="AV", av_fams=("naive_ignore", "R8", "B_prior"), scale="abs", n=n,
                sd_ideal=sb / np.sqrt(n) / den, spec=spec), None


def repair_M6(card, th, d, res, rng):
    g, near = res["first_fail"], res.get("nearest") or ""
    d = dict(d)
    sp = _span(card)
    if g in ("G5_B", "G9_redteam"):
        if near.startswith("B_prior"):
            return "RESAMPLE", "world floor close to the published constant -> resample world"
        if near.startswith(("R8", "R4")):
            nt = d["x_top"] - 0.15 * sp
            if nt > d["xb"] + 0.1 * sp:
                d["x_top"] = float(nt)
                return d, "saturated/secant floor route nearest -> lower the largest sold config"
        if res["noise_limited"] and d["n"] * 2 <= 400:
            d["n"] *= 2
            return d, "noise-limited -> 2x runs"
    if g == "G6_kappa":
        d["xa"] = float(max(card.lo, d["xa"] - 0.1 * sp))
        return d, "kappa too high -> start the comparison further from the floor (xa lower)"
    if g == "G7_free":
        if d["n"] * 2 <= 400:
            d["n"] *= 2
            return d, "free notebook estimate within 2T -> demand more precision (2x runs, T shrinks)"
        d["r_nb"] *= 2
        return d, "free notebook estimate within 2T and n capped -> noisier notebook (x2)"
    if g == "G4_tight" and d["n"] * 2 <= 400:
        d["n"] *= 2
        return d, "T too loose -> 2x runs"
    return None, "no repair rule for %s" % g


# ================================================================================================ search knobs
# The driver-aware proposer (compiler.run_cell_search) mutates designs inside these boxes.  kind: 'pos' = position
# on the card's axis (additive, sd 8% of span), 'scale' = positive magnitude (log-normal step), 'int' / 'odd' /
# 'choice' = discrete.  The boxes deliberately reach past the realism line (r_nb >= 4, n >= 300) so that the
# search can report how much of a cell's fertility needs an implausible notebook or budget.
def knobs(card, tpl):
    sp, lo, hi = _span(card), card.lo, card.hi
    if tpl == "M1":
        return dict(xs=("pos", lo, hi), h=("scale", 0.02 * sp, 0.35 * sp), u=("frac", 0.1, 0.9),
                    n=("scale_int", 20, 400), r_nb=("scale", 1.0, 6.0))
    if tpl == "M2":
        return dict(K=("odd", 5, 15), m=("int", 1, 7), xs=("pos", lo, hi), x_nb=("pos", lo, hi), n=("scale_int", 15, 400))
    if tpl == "M3":
        return dict(G=("choice", (4, 6, 8, 12, 16)), r_nb=("scale", 1.0, 6.0), x0=("pos", lo, hi), n=("scale_int", 10, 400))
    if tpl == "M6":
        return dict(xa=("pos", lo, hi), xb=("pos", lo, hi), x_top=("pos", lo, hi), n=("scale_int", 20, 400),
                    r_nb=("scale", 1.0, 6.0), dt=("scale", 0.25, 1.5))
    raise KeyError(tpl)


def mutate(card, tpl, d, rng, k_moves=1):
    d = dict(d)
    ks = knobs(card, tpl)
    for key in rng.choice(sorted(ks), size=k_moves, replace=False):
        kind, *b = ks[key]
        v = d[key]
        if kind == "pos":
            v = float(np.clip(v + rng.normal(0, 0.08 * _span(card)), b[0], b[1]))
        elif kind == "frac":
            v = float(np.clip(v + rng.normal(0, 0.15), b[0], b[1]))
        elif kind == "scale":
            v = float(np.clip(v * np.exp(rng.normal(0, 0.4)), b[0], b[1]))
        elif kind == "scale_int":
            v = int(np.clip(round(v * np.exp(rng.normal(0, 0.5))), b[0], b[1]))
        elif kind == "odd":
            v = int(np.clip(v + 2 * rng.choice([-1, 1]), b[0], b[1]))
        elif kind == "int":
            v = int(np.clip(v + rng.choice([-1, 1]), b[0], b[1]))
        elif kind == "choice":
            v = int(rng.choice(b[0]))
        d[key] = v
    if tpl == "M2":
        d["n"] = int(max(d["n"], 3 * d["K"]))
    return d


PRIOR = {"M1": prior_M1, "M2": prior_M2, "M3": prior_M3, "M6": prior_M6}
BUILD = {"M1": build_M1, "M2": build_M2, "M3": build_M3, "M6": build_M6}
REPAIR = {"M1": repair_M1, "M2": repair_M2, "M3": repair_M3, "M6": repair_M6}
