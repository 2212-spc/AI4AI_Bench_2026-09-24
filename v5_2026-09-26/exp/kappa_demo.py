"""The constructive half of the difficulty law: two estimands, one lab, one budget, one noise level.

`sweep_B.py` established that

    B  =  bias_geom / (2.25 * kappa * sigma / sqrt(n))

and that the two geometry knobs (reach, curvature) are falsified for e05 q2, because that item's reference
estimator extrapolates a three-parameter tanh fit, so its kappa is 7-30 and rises faster than the geometric
gap.  Consequence: that estimand needs ~987 runs to reach B = 3 against a 130-run budget.

That is a statement about one *family* of estimands, not about difficulty.  This script builds the matched
pair that isolates kappa and nothing else.  Both items live on servelab's S4 card (Pollaczek-Khinchine
M/G/1), both are answered from the same `load` service with the same lognormal measurement noise
`sig_lat / sqrt(dur / DUR_REF)`, and both are given the same run budget.  They differ in one respect:

  item A  "saturation rate"   -- at what arrival rate does p99 cross an SLO?  The lab only sells rates up to
                                 a utilisation ceiling, so the reference estimator must fit the two free
                                 constants of the queueing form on the gentle part of the curve and invert
                                 the fit past the ceiling.  kappa >> 1 by construction.

  item B  "mixed-traffic p99"  -- what is the mean p99 over a declared traffic mix whose buckets the lab
                                 *does* sell?  The reference estimator buys each bucket and averages.  No
                                 fit, no extrapolation, no inversion: kappa = 1 exactly.

Both are honest questions a serving engineer is actually paid to answer, and both have a natural wrong
answer.  A's is "extrapolate the latency curve you measured"; B's is "use the average load", which is the
Jensen gap and, because E[T] blows up as utilisation approaches one, is enormous: the p99 you will actually
serve is set by the busy buckets, not by the mean rate.

Prediction before running: B_A ~ 1 (unaffordable, like e05 q2), B_B >> 3 at the same cost.

    python3 exp/kappa_demo.py [--reps 4] [--mc 24] [--out exp/kappa_demo.json]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalelab import lab as L, routes as RT
from scalelab.labs import servelab as S

# ------------------------------------------------------------------------------------------- the world
# S4's only drawn constant is the squared coefficient of variation of the service time; the rest of the
# serving physics comes from servelab's BASE.  Drawing cs2 matters here: it is the *independent world
# constant* that sets item B's shortcut bias, which is what the design rule asks for.
BITS = 16
SEQ = 1024
BATCH = 32
DUR = 20.0

# Item B's traffic mix, declared as utilisations so the shape is readable; the rates follow from E[S].
# Equal weights, so the reference estimator is the plain average of the buckets.
MIX_RHO = (0.25, 0.40, 0.55, 0.70, 0.82, 0.90)

# Item A: the lab refuses to load-test above this utilisation ("beyond the safe operating envelope"), so the
# buyable grid stops well short of the answer.  Six buyable rates, same count as item B's buckets.
A_RHO_CEIL = 0.55
A_GRID_RHO = (0.15, 0.23, 0.31, 0.39, 0.47, 0.55)
A_SLO_MS = 900.0

FLOOR_REL = 0.01          # 1% of the truth: the tolerance can never be tighter than the reporting precision


def world(ws):
    """A servelab world with S4's constant drawn, at the reference operating point."""
    rng = np.random.default_rng(0x5A4E + ws)
    p = dict(S.BASE)
    p["cs2"] = float(rng.uniform(0.4, 3.0))
    pf = S.full(p)
    es = S.service_time(pf, BITS, SEQ, BATCH)
    return pf, es


def p99_of(pf, rate):
    """Noiseless p99 latency in ms at one arrival rate, from the card itself."""
    st = S.queue_stats(pf, BITS, SEQ, BATCH, rate)
    return float(st["p99_ms"]) if st.get("stable") else float("inf")


def session(pf, salt):
    """A `load`-capable session with the caps lifted, so the cost of a plan is counted rather than enforced.

    The spec is written here rather than lifted from `bp/e02_spec_tail_cex.py`, which is the only shipped
    servelab blueprint and exposes `svc: ["bench"]` only - no `load` service and no `rate` knob.  A real
    blueprint for either item needs this spec; the demo is also the spec's first test.
    """
    spec = {"lab": "servelab",
            "knobs": {"svc": {"type": "choice", "values": ["bench", "load"], "default": "load"},
                      "seq": {"type": "float", "min": 256, "max": 4096, "int": True, "default": SEQ},
                      "batch": {"type": "float", "min": 1, "max": 1024, "int": True, "default": BATCH},
                      "rate": {"type": "float", "min": 0.1, "max": 200.0, "default": 1.0},
                      "dur": {"type": "float", "min": 5.0, "max": 40.0, "default": DUR}},
            "fixed": {"bits": BITS},
            "caps": {"run_cost": 1e12, "total_cost": 1e15, "max_runs": 10 ** 6}}
    sess = L.Session(pf, spec, salt)
    sess.caps = dict(spec["caps"])
    return sess


def measure_p99(sess, rate, rng):
    """One purchased `load` run, returning the reported p99 in ms."""
    r = sess.run({"svc": "load", "seq": SEQ, "batch": BATCH, "rate": float(rate), "dur": DUR,
                  "seed": int(rng.integers(1e6))})
    if r.get("status") == "unstable":
        raise ValueError("unstable at rate %.4f" % rate)
    return float(r["p99_ms"])


# ------------------------------------------------------------------------------- item B: kappa = 1
def estimand_B(pf, es):
    """Mean p99 over the declared mix.  Instrument axis = arrival rate; the grid IS the mix."""
    rates = [r / es for r in MIX_RHO]
    truth = float(np.mean([p99_of(pf, r) for r in rates]))
    # `target` is the mix's mean rate: it is what the naive route evaluates at, and it is the only scalar
    # instrument value the question can be said to be "about".  The honest answer is R7:average, which
    # `certify` is told to treat as the reference rather than as a rival.
    est = RT.Estimand(resp=lambda r: p99_of(pf, r), grid=rates, target=float(np.mean(rates)),
                      truth=truth, name="svc.mix_p99")
    return est, rates, truth


def oracle_B(pf, rates, reps, mc):
    """Buy every bucket `reps` times and average.  Linear in the measurements, so kappa = 1."""
    errs, truth = [], float(np.mean([p99_of(pf, r) for r in rates]))
    for t in range(mc):
        sess = session(pf, "kappaB%03d" % t)
        rng = np.random.default_rng(500000 + 7919 * t)
        per = []
        for r in rates:
            per.append(float(np.mean([measure_p99(sess, r, rng) for _ in range(reps)])))
        errs.append(abs(float(np.mean(per)) - truth))
    return errs, len(rates) * reps


# ------------------------------------------------------------------------------- item A: kappa >> 1
def _fit_pk(rates, ets):
    """Least squares for the M/G/1 mean sojourn time E[T](r) = S + c*r*S^2/(1 - r*S).

    S enters nonlinearly, so grid it and solve the single remaining coefficient exactly - the same shape as
    `e05._fit_intercept`, and the same weakness: over a gentle stretch of the curve, S and c trade off
    against each other, which is precisely the ill-conditioning this item is built to have.
    """
    r = np.asarray(rates, float); y = np.asarray(ets, float)
    best = None
    lo, hi = 0.2 * min(1.0 / max(r), 1.0), 0.999 / max(r)
    for s in np.exp(np.linspace(math.log(lo), math.log(hi), 481)):
        if np.any(r * s >= 1.0):
            continue
        g = r * s * s / (1.0 - r * s)                      # the basis function multiplying c
        z = y - s
        den = float(g @ g)
        if den <= 0:
            continue
        c = float(g @ z) / den
        res = float(np.sum((z - c * g) ** 2))
        if best is None or res < best[0]:
            best = (res, float(s), c)
    if best is None:
        raise ValueError("no feasible service time on the grid")
    return best[1], best[2]


def _sat_rate(s, c, et_slo):
    """Invert E[T](r) = et_slo for the arrival rate.  All times in seconds.

    E[T] = S + c r S^2/(1-rS) = T  ->  c r S^2 = (T-S)(1-rS)  ->  r = (T-S) / (S (cS + T - S)).
    """
    d = float(et_slo) - float(s)
    if d <= 0:
        raise ValueError("SLO is below the service time itself")
    den = s * (c * s + d)
    if den <= 0:
        raise ValueError("no positive saturation rate")
    return d / den


def estimand_A(pf, es):
    """The arrival rate at which p99 crosses the SLO, from a grid that stops at A_RHO_CEIL."""
    rates = [r / es for r in A_GRID_RHO]
    et_slo = A_SLO_MS / math.log(100.0) / 1e3               # p99 = ln(100) E[T]; seconds
    # Invert the exact card rather than a fit: E[T] = S(1 + k rho/(1-rho)) with k = (1+cs2)/2, so
    # a = E[T]/S = 1 + k rho/(1-rho)  ->  rho = (a-1)/(a-1+k).
    a = et_slo / es
    k = (1.0 + pf["cs2"]) / 2.0
    rho_star = (a - 1.0) / (a - 1.0 + k)
    truth = rho_star / es
    est = RT.Estimand(resp=lambda r: p99_of(pf, r), grid=rates, target=A_SLO_MS, truth=truth,
                      invert=True, name="svc.sat_rate")
    return est, rates, truth, rho_star


def oracle_A(pf, rates, reps, mc, es):
    """Fit the queueing form on the buyable rates and invert past the ceiling.  kappa is what we measure."""
    _, _, truth, _ = estimand_A(pf, es)
    errs, wild = [], 0
    for t in range(mc):
        sess = session(pf, "kappaA%03d" % t)
        rng = np.random.default_rng(900000 + 7919 * t)
        ets = []
        for r in rates:
            p99 = float(np.mean([measure_p99(sess, r, rng) for _ in range(reps)]))
            ets.append(p99 / math.log(100.0) / 1e3)        # seconds, the unit `_fit_pk` works in
        try:
            s, c = _fit_pk(rates, ets)
            est = _sat_rate(s, c, A_SLO_MS / math.log(100.0) / 1e3)
        except (ValueError, ZeroDivisionError):
            wild += 1
            continue
        if not math.isfinite(est) or est <= 0 or est > 50.0 / es:
            wild += 1
            continue
        errs.append(abs(est - truth))
    return errs, wild, len(rates) * reps


# ------------------------------------------------------------------------------------------- reporting
def p90(xs):
    return float(np.percentile(np.asarray(xs, float), 90)) if xs else float("inf")


def certify(name, est, truth, errs, floor_rel, reference=()):
    t90 = p90(errs)
    tol = max(floor_rel * abs(truth), 2.25 * t90)
    routes = RT.numerical_routes(est)
    cert = RT.certify(truth, tol, routes, reference=reference)
    kappa_free = 2.25 * t90 / max(tol, 1e-300)
    return {"name": name, "truth": truth, "p50": float(np.percentile(errs, 50)) if errs else None,
            "p90": t90, "tol": tol, "tol_rel": tol / abs(truth), "B": cert["B"],
            "nearest": cert["nearest"], "n_routes": cert["n_routes"], "at_floor": kappa_free < 1.0,
            "table": cert["table"], "census": RT.census({k: v for k, v in routes.items()
                                                         if k not in set(reference)})}


SWEEP_REPS = (1, 2, 4, 8, 16, 32)


def sweep(pf, es, args):
    """Both items against the same precision ladder, so kappa is read off as a price rather than asserted.

    Each row buys `reps` repetitions of every grid point, which costs the same number of runs in both items
    (six grid points each, by construction).  If B is noise-limited in both, the two curves are parallel
    lines of slope +1/2 in log-log and the vertical offset between them *is* log kappa; the item with the
    worse conditioning then needs (kappa_ratio)^2 times the budget for the same difficulty.
    """
    estA, ratesA, truthA, rho_star = estimand_A(pf, es)
    estB, ratesB, truthB = estimand_B(pf, es)
    print("item A truth %.4f req/s (rho*=%.4f);  item B truth %.2f ms" % (truthA, rho_star, truthB))
    print("%5s %6s | %9s %9s %7s %6s | %9s %9s %7s"
          % ("reps", "runs", "A p90", "A T", "A B", "wild", "B p90", "B T", "B B"))
    rows = []
    for reps in SWEEP_REPS:
        eA, wildA, nA = oracle_A(pf, ratesA, reps, args.mc, es)
        eB, nB = oracle_B(pf, ratesB, reps, args.mc)
        cA = certify("svc.sat_rate", estA, truthA, eA, FLOOR_REL)
        cB = certify("svc.mix_p99", estB, truthB, eB, FLOOR_REL, reference=("R7:average",))
        rows.append({"reps": reps, "n_runs": nA, "A": cA, "B": cB, "wild": wildA})
        print("%5d %6d | %9.4f %9.4f %7.2f %6d | %9.3f %9.3f %7.2f"
              % (reps, nA, cA["p90"], cA["tol"], cA["B"], wildA, cB["p90"], cB["tol"], cB["B"]))

    print()
    out = {}
    for k in ("A", "B"):
        # Fit only the rows still above the reporting floor: once T hits the floor, B stops responding to
        # precision and including those rows would flatten the slope for a reason that is not conditioning.
        use = [r for r in rows if not r[k]["at_floor"] and r[k]["B"] > 0]
        if len(use) < 3:
            print("item %s: only %d rows above the tolerance floor, no fit" % (k, len(use)))
            out[k] = {"slope": None}
            continue
        ln, lb = np.log([r["n_runs"] for r in use]), np.log([r[k]["B"] for r in use])
        slope, icpt = np.polyfit(ln, lb, 1)
        need = math.exp((math.log(3.0) - icpt) / slope) if slope > 0 else float("inf")
        cap = max(r[k]["B"] for r in rows if r[k]["at_floor"]) if any(r[k]["at_floor"] for r in rows) else None
        print("item %s (%s): d log B / d log n = %+.2f   B = 3 at %.0f runs%s"
              % (k, use[0][k]["name"], slope, need,
                 "   (tolerance floor caps B at %.1f)" % cap if cap else ""))
        out[k] = {"slope": float(slope), "intercept": float(icpt), "need_runs_for_B3": float(need),
                  "floor_cap": cap}
    if out["A"].get("need_runs_for_B3") and out["B"].get("need_runs_for_B3"):
        ra, rb = out["A"]["need_runs_for_B3"], out["B"]["need_runs_for_B3"]
        print("\nPRICE OF B = 3:  %.0f runs for the extrapolating item, %.0f for the aggregating one"
              " -- a factor of %.0f." % (ra, rb, ra / max(rb, 1e-9)))
        print("that factor is kappa^2.  kappa ~ %.1f, and it is the whole difference between an item family"
              " that fits a budget and one that does not." % math.sqrt(ra / max(rb, 1e-9)))

    # The second half of the mechanism, and the counter-intuitive half.  kappa is the denominator; the
    # numerator moves the same way, because a shortcut that is *nearly right* is what makes an extrapolation
    # feel hard in the first place.  A 14%-past-the-ceiling extrapolation has a 2.6% shortcut bias; a Jensen
    # gap across a convex blow-up has an 18% one.  Both factors favour aggregation, so the intuition that
    # extrapolation is the hard question is exactly backwards for benchmark construction.
    print("\nwhere the two items' difficulty actually comes from:")
    print("%-16s %14s %14s %14s %10s" % ("item", "shortcut bias", "% of truth", "floor T (1%)", "B ceiling"))
    for k, est, truth in (("A", estA, truthA), ("B", estB, truthB)):
        r0 = rows[-1][k]["table"][0]
        bias = r0["abs_err"]
        print("%-16s %14.4f %13.1f%% %14.4f %10.1f"
              % (rows[-1][k]["name"], bias, 100 * bias / abs(truth), FLOOR_REL * abs(truth),
                 bias / (FLOOR_REL * abs(truth))))
        out[k]["bias_geom"] = float(bias)
        out[k]["bias_frac"] = float(bias / abs(truth))
        out[k]["B_ceiling"] = float(bias / (FLOOR_REL * abs(truth)))
    if out["A"].get("B_ceiling", 9e9) < 3.0:
        print("item A's ceiling is below the floor of 3: no budget whatsoever makes that estimand hard,")
        print("because its nearest shortcut is only %.1f%% wrong and the answer is only reportable to %.0f%%."
              % (100 * out["A"]["bias_frac"], 100 * FLOOR_REL))

    if args.out:
        path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"mode": "sweep", "ws": args.ws, "mc": args.mc, "cs2": pf["cs2"],
                       "E_S_ms": 1e3 * es, "floor_rel": FLOOR_REL,
                       "rows": [{"reps": r["reps"], "n_runs": r["n_runs"], "wild": r["wild"],
                                 "A": {kk: r["A"][kk] for kk in ("p90", "tol", "B", "nearest", "at_floor")},
                                 "B": {kk: r["B"][kk] for kk in ("p90", "tol", "B", "nearest", "at_floor")}}
                                for r in rows],
                       "fit": out}, fh, indent=1, default=float)
        print("\nwrote %s" % path)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=int, default=2)
    ap.add_argument("--reps", type=int, default=4, help="purchased repetitions per grid point (both items)")
    ap.add_argument("--mc", type=int, default=24, help="independent noise draws for the tolerance")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep reps on both items and report the runs each needs to reach B = 3")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pf, es = world(args.ws)
    print("servelab world ws=%d: cs2=%.4f  E[S]=%.2f ms at batch=%d seq=%d  saturation rate=%.3f req/s"
          % (args.ws, pf["cs2"], 1e3 * es, BATCH, SEQ, 1.0 / es))
    print("measurement noise: sig_lat=%.4f at dur=%.0f s (lognormal, per reported quantile)\n"
          % (pf["sig_lat"] / math.sqrt(DUR / S.DUR_REF), DUR))

    if args.sweep:
        return sweep(pf, es, args)

    rows = {}

    # ---------------------------------------------------------------- A
    estA, ratesA, truthA, rho_star = estimand_A(pf, es)
    print("item A  saturation rate: at what arrival rate does p99 cross %.0f ms?" % A_SLO_MS)
    print("  buyable utilisations %s -> rates %.2f .. %.2f req/s"
          % (list(A_GRID_RHO), min(ratesA), max(ratesA)))
    print("  truth: rho*=%.4f, rate*=%.4f req/s  (the answer sits %.0f%% past the buyable ceiling)"
          % (rho_star, truthA, 100.0 * (rho_star / A_RHO_CEIL - 1.0)))
    errsA, wildA, nA = oracle_A(pf, ratesA, args.reps, args.mc, es)
    rows["A"] = certify("svc.sat_rate", estA, truthA, errsA, FLOOR_REL)
    rows["A"].update({"n_runs": nA, "wild": wildA, "kind": "extrapolate+invert"})
    print("  oracle: fit E[T](r) on the buyable grid, invert past the ceiling -- %d runs, %d/%d wild"
          % (nA, wildA, args.mc))
    print("  p90|err| = %.4f req/s  ->  T = %.4f (%.2f%% of truth)   B = %.2f  nearest %s\n"
          % (rows["A"]["p90"], rows["A"]["tol"], 100 * rows["A"]["tol_rel"], rows["A"]["B"],
             rows["A"]["nearest"]))

    # ---------------------------------------------------------------- B
    estB, ratesB, truthB = estimand_B(pf, es)
    print("item B  mixed-traffic p99: mean p99 over the declared mix (equal weights)")
    print("  buckets rho %s -> rates %.2f .. %.2f req/s" % (list(MIX_RHO), min(ratesB), max(ratesB)))
    print("  truth: %.2f ms   naive 'use the average load' = p99(mean rate) = %.2f ms"
          % (truthB, p99_of(pf, float(np.mean(ratesB)))))
    errsB, nB = oracle_B(pf, ratesB, args.reps, args.mc)
    rows["B"] = certify("svc.mix_p99", estB, truthB, errsB, FLOOR_REL, reference=("R7:average",))
    rows["B"].update({"n_runs": nB, "wild": 0, "kind": "aggregate"})
    print("  oracle: buy every bucket %dx and average -- %d runs, no fit" % (args.reps, nB))
    print("  p90|err| = %.3f ms  ->  T = %.3f (%.2f%% of truth)   B = %.2f  nearest %s\n"
          % (rows["B"]["p90"], rows["B"]["tol"], 100 * rows["B"]["tol_rel"], rows["B"]["B"],
             rows["B"]["nearest"]))

    # ---------------------------------------------------------------- the comparison
    print("=" * 100)
    print("%-16s %-20s %6s %10s %10s %8s %8s  %s"
          % ("item", "reference estimator", "runs", "truth", "T", "T/truth", "B", "nearest route"))
    for k in ("A", "B"):
        r = rows[k]
        print("%-16s %-20s %6d %10.4f %10.4f %7.2f%% %8.2f  %s"
              % (r["name"], r["kind"], r["n_runs"], r["truth"], r["tol"], 100 * r["tol_rel"], r["B"],
                 r["nearest"]))
    print("=" * 100)
    ratio = rows["B"]["B"] / max(rows["A"]["B"], 1e-9)
    print("same lab, same noise, same %d-run budget: B differs %.0fx (%.2f vs %.2f)."
          % (rows["A"]["n_runs"], ratio, rows["B"]["B"], rows["A"]["B"]))
    print("the only difference is the conditioning of the reference estimator:")
    print("  A extrapolates a two-parameter fit past the buyable ceiling  -> kappa >> 1, B unaffordable")
    print("  B averages six directly purchased measurements               -> kappa = 1,  B = %.1f"
          % rows["B"]["B"])
    if rows["B"]["at_floor"]:
        print("  B's tolerance is at the %.0f%% reporting floor, so its B is set by geometry alone and"
              % (100 * FLOOR_REL))
        print("  buying more reps cannot raise it further -- the floor is the right place to be.")
    print("\nfor each item the nearest shortcut and its distance:")
    for k in ("A", "B"):
        print("  %s (%s):" % (rows[k]["name"], rows[k]["kind"]))
        for r in rows[k]["table"][:5]:
            print("    %-18s %12.4f  %7.2f T   %s"
                  % (r["route"], r["value"], r["over_T"], RT.FAMILIES.get(r["route"].split(":")[0], "")))
        print("    census: %s" % rows[k]["census"])

    if args.out:
        path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"ws": args.ws, "reps": args.reps, "mc": args.mc, "cs2": pf["cs2"],
                       "E_S_ms": 1e3 * es, "sig_lat_eff": pf["sig_lat"] / math.sqrt(DUR / S.DUR_REF),
                       "floor_rel": FLOOR_REL, "items": rows}, fh, indent=1, default=float)
        print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
