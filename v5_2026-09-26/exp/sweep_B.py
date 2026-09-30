"""Which difficulty knob actually binds?  Two sweeps against the certificate B.

B = min over buyable routes |route - truth| / T, with T = max(floor, 2.25 * p90(oracle error)).  Both the
numerator and the denominator are properties of the same world, so a knob only raises difficulty if it moves
the numerator faster than the denominator.  This script sweeps the two candidates one at a time, everything
else held fixed, on e05 q2 - the one v4 item that separated two frontier models.

  --knob reach   how far down toward the target the lab will sell.  The shipped length-penalty grid is
                 scaled by a fraction, keeping its shape and its twelve points, so only the cap moves.
                 reach = 1 - x_min / x_run.  This is the "put the question out of reach" knob.

  --knob curv    how sharply the response saturates inside the buyable region, via the length term's
                 saturation scale `sl`.  `sl` enters `proxy_reward` only, not `mean_len`, so the grid of
                 lengths the lab sells is untouched: this is curvature at constant reach.  This is the
                 "make the shortcut's functional form wrong" knob.

  --knob noise   the measurement noise itself (`sig_proxy`, `sig_len`), at constant geometry.  This one is
                 not a difficulty proposal; it is the *diagnostic*.  A route's bias is a fixed property of
                 the world and does not depend on sigma, so if the tolerance were driven by noise alone then
                 B would scale like 1/sigma and halving the noise would double B.  If instead B is flat in
                 sigma, the tolerance is not being set by noise: it is being set by the same extrapolation
                 remainder that makes the route wrong, and the item has an O(1) ceiling on B that no amount
                 of geometry can lift.  Call this the orthogonality test.

The prediction being tested is the second-order remainder of the secant route,

    B_R4  ~  |y''| * (target - t_near)^2 / (2 T),

which says reach enters squared and curvature enters linearly.  What that expression hides is that T is not
a constant: the oracle extrapolates over the same grid, so anything that lengthens the extrapolation also
disperses the reference estimate.  Reach moves both; curvature moves only the numerator, and in fact helps
the oracle, because curvature inside the sample is what pins the saturation scale.

The oracle column is e05's own `_q2_measure` run through a real noisy `Session`, not a re-implementation, so
the tolerance reported here is the tolerance the build would assign.

    python3 exp/sweep_B.py --knob reach [--reps 24]
    python3 exp/sweep_B.py --knob curv  [--reps 24]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalelab import build as B, lab as L, routes as RT
from scalelab.bp import e05_matched_kl_far as E5
from scalelab.labs import rllab as R

WS = 2
# Reach: fractions of the shipped LP_MAX=8.0.  Below about 0.15 the sweep stops pinning the saturation
# scale and the fit walks off its grid, which is non-identifiability, not difficulty.
REACH_FRACS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.17, 0.12)
# Curvature: multipliers on the drawn `sl`.  Smaller sl saturates harder.  `_fit_intercept` grids S over
# [15, 4000], so a multiplier that pushes sl below ~20 is outside what the reference estimator can fit and
# would measure the fit's grid, not the item.
CURV_MULTS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.13)
# Noise: multipliers on the reported standard deviations q2 actually reads.  A factor of 8 either way is
# plenty to tell a 1/sigma slope from a flat line.
NOISE_MULTS = (4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
NOISE_KEYS = ("sig_proxy", "sig_len")
# Repetitions: multipliers on the oracle's own measurement plan (Q2_BASE base runs, Q2_REPS per grid
# point).  This is the knob the *pipeline* can turn - sigma is a world constant with a physical meaning,
# whereas how many runs the reference plan averages is a design choice, paid for out of the item's budget.
REPS_MULTS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
WILD = 5.0                                    # |share| beyond this is a diverged fit, not a noisy one


def variant(w, sl=None, noise=None):
    """A world with one constant overridden.  The notebook is not regenerated because nothing downstream of
    here reads it: `_q2_measure` buys its own runs and the Estimand is analytic."""
    if sl is None and noise is None:
        return w
    p = dict(w["p"])
    if sl is not None:
        p["sl"] = float(sl)
    if noise is not None:
        pf0 = w["pf"]
        for k in NOISE_KEYS:
            p[k] = float(pf0[k]) * float(noise)
    return dict(w, p=p, pf=R.full(p))


def estimand(w, grid_lp):
    """e05 q2 as an Estimand, over the length grid this cap buys."""
    p = w["p"]; pf = w["pf"]
    run = E5._ppo(p["q_KLT"], p["q_ST"])
    tot = E5._proxy(pf, run)
    x_run = E5._len(pf, run) - pf["len0"]
    qual = pf["wq"] * pf["alpha"] * math.sqrt(E5._kl(pf, run))

    def resp(x):
        return qual + pf["wl"] * math.tanh(x / pf["sl"])

    xs = [E5._len(pf, E5._ppo(p["q_KLT"], p["q_ST"], len_pen=lp)) - pf["len0"] for lp in grid_lp]
    truth = E5._q2_truth(p)
    est = RT.Estimand(resp=resp, grid=xs, target=0.0, truth=truth,
                      to_answer=lambda y: 1.0 - y / tot, op=x_run, name="e05.q2")
    # |y''| at the midpoint of the extrapolation, for the second-order prediction.
    xm = 0.5 * min(xs)
    u = xm / pf["sl"]
    ypp = abs(-2.0 * pf["wl"] / pf["sl"] ** 2 * math.tanh(u) * (1.0 - math.tanh(u) ** 2))
    return est, xs, x_run, truth, ypp / tot          # y'' in *answer* units


def oracle_errors(w, truth, reps):
    """|estimate - truth| over `reps` independent noise draws, using e05's own measurement function.

    `_q2_measure` reads `E5.LP_GRID` at call time, so the caller sets it; that keeps this faithful to the
    shipped estimator instead of re-deriving it (a re-derivation is how the first version of this sweep got
    `len0` from a PPO run instead of the best-of-1 initial policy, and reported a 3x too-wide tolerance).
    """
    errs, wild = [], 0
    for t in range(reps):
        sess = L.Session(w["pf"], w["spec"], "%s/sweep%d" % (w["salt"], t))
        sess.caps = {"run_cost": 1e12, "total_cost": 1e15, "max_runs": 10 ** 6}
        try:
            est, _ = E5._q2_measure(sess, np.random.default_rng(90000 + 7919 * t))
        except Exception:
            wild += 1
            continue
        if not math.isfinite(est) or abs(est) > WILD:
            wild += 1
            continue
        errs.append(abs(est - truth))
    return errs, wild


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", choices=("reach", "curv", "noise", "reps"), default="reach")
    ap.add_argument("--reps", type=int, default=E5.Q2_MC)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_B_%s.json" % a.knob)

    w0 = B.make_world(E5, WS)
    shipped_grid = E5.LP_GRID
    sl0 = w0["p"]["sl"]
    print("e05 ws=%d  knob=%s  reps=%d   shipped LP_MAX=%.1f  drawn sl=%.1f  sig_proxy=%.4g  floor=%.3f\n" %
          (WS, a.knob, a.reps, E5.LP_MAX, sl0, w0["pf"]["sig_proxy"], E5.FLOOR_SHARE))
    hdr = ("%6s %7s %7s %7s %9s %8s %5s %8s %7s %8s  %-15s %8s" %
           ("mult", "LP_max", "sl", "reach", "truth", "p90", "wild", "T_cal", "B_cal", "B_pred",
            "nearest@cal", "B_floor")) + " %6s %6s" % ("runs", "rho")
    print(hdr); print("-" * len(hdr))
    rows = []
    try:
        muls = {"reach": REACH_FRACS, "curv": CURV_MULTS, "noise": NOISE_MULTS,
                "reps": REPS_MULTS}[a.knob]
        base0, reps0 = E5.Q2_BASE, E5.Q2_REPS
        for m in muls:
            if a.knob == "reps":
                E5.Q2_BASE = max(2, int(round(base0 * m)))
                E5.Q2_REPS = max(1, int(round(reps0 * m)))
            grid = tuple(lp * m for lp in shipped_grid) if a.knob == "reach" else shipped_grid
            w = variant(w0, sl=(sl0 * m if a.knob == "curv" else None),
                        noise=(m if a.knob == "noise" else None))
            est, xs, x_run, truth, ypp = estimand(w, grid)
            reach = 1.0 - min(xs) / x_run
            gen = RT.numerical_routes(est)
            c_floor = RT.certify(truth, E5.FLOOR_SHARE, gen)

            E5.LP_GRID = grid                              # what `_q2_measure` sweeps
            errs, wild = oracle_errors(w, truth, a.reps)
            p90 = float(np.percentile(errs, 90)) if errs else float("nan")
            t_cal = max(E5.FLOOR_SHARE, 2.25 * p90) if errs else float("nan")
            c_cal = (RT.certify(truth, t_cal, gen) if errs
                     else {"B": float("nan"), "nearest": "-", "table": []})
            b_pred = ypp * min(xs) ** 2 / (2.0 * t_cal) if errs else float("nan")
            n_runs = len(grid) * E5.Q2_REPS + E5.Q2_BASE
            rho = n_runs / float(E5.MAX_RUNS)
            print("%6.2f %7.2f %7.1f %7.3f %9.5f %8.4f %5d %8.4f %7.2f %8.2f  %-15s %8.2f %6d %6.2f" %
                  (m, max(grid), w["pf"]["sl"], reach, truth, p90, wild, t_cal, c_cal["B"], b_pred,
                   c_cal["nearest"], c_floor["B"], n_runs, rho))
            rows.append({"mult": m, "lp_max": max(grid), "sl": w["pf"]["sl"], "reach": reach,
                         "x_min": min(xs), "truth": truth, "p90": p90, "n_wild": wild, "n_ok": len(errs),
                         "T_cal": t_cal, "B_cal": c_cal["B"], "B_pred": b_pred,
                         "nearest_cal": c_cal["nearest"], "B_floor": c_floor["B"],
                         "n_runs": n_runs, "rho": rho,
                         "table": c_cal["table"][:8]})
    finally:
        E5.LP_GRID = shipped_grid
        E5.Q2_BASE, E5.Q2_REPS = base0, reps0

    with open(out, "w") as f:
        json.dump({"bp": E5.ID, "ws": WS, "knob": a.knob, "reps": a.reps, "sl0": sl0,
                   "floor": E5.FLOOR_SHARE, "rows": rows}, f, indent=1)
    print("\nwrote %s" % out)

    live = [r for r in rows if math.isfinite(r["B_cal"]) and r["n_wild"] == 0]
    if live:
        best = max(live, key=lambda r: r["B_cal"])
        print("B_cal peaks at mult=%.2f: B=%.2f (nearest %s), T=%.4f, %d/%d reps clean" %
              (best["mult"], best["B_cal"], best["nearest_cal"], best["T_cal"],
               best["n_ok"], best["n_ok"] + best["n_wild"]))
    b = [r["B_cal"] for r in rows]
    if a.knob == "reps":
        ok = [r for r in rows if r["n_wild"] == 0 and math.isfinite(r["B_cal"])]
        if len(ok) >= 3:
            sl_, _ = np.polyfit(np.log([r["n_runs"] for r in ok]), np.log([r["B_cal"] for r in ok]), 1)
            print("\nECONOMICS: d log B / d log n_runs = %+.2f  (theory: +0.50, since p90 ~ sigma/sqrt(n))"
                  % sl_)
            ship = [r for r in rows if r["mult"] == 1.0]
            if ship and ship[0]["B_cal"] > 0:
                need = ship[0]["n_runs"] * (3.0 / ship[0]["B_cal"]) ** 2
                print("  to reach B=3.0 from B=%.2f at %d runs needs about %.0f runs, i.e. rho=%.1f"
                      % (ship[0]["B_cal"], ship[0]["n_runs"], need, need / E5.MAX_RUNS))
                print("  the item's whole budget is %d runs, so this estimand %s afford B=3"
                      % (E5.MAX_RUNS, "can" if need <= E5.MAX_RUNS else "CANNOT"))
                print("  the geometric ceiling is B_floor=%.2f, reached when T hits the floor %.3f"
                      % (rows[0]["B_floor"], E5.FLOOR_SHARE))
    if a.knob == "noise":
        ok = [(r["mult"], r["B_cal"]) for r in rows if r["n_wild"] == 0 and math.isfinite(r["B_cal"])]
        if len(ok) >= 3:
            sl_, _ = np.polyfit(np.log([m for m, _ in ok]), np.log([v for _, v in ok]), 1)
            print("\nORTHOGONALITY TEST: d log B / d log sigma = %+.2f" % sl_)
            print("  -1 means the tolerance is set by noise alone and B is a real knob (halve sigma,")
            print("     double B).   0 means the tolerance is set by the same extrapolation remainder that")
            print("     biases the route, so B has an O(1) ceiling and this item family cannot be made hard.")
            print("  verdict: %s" % ("noise-limited, B is buyable" if sl_ <= -0.6 else
                                     "REMAINDER-LIMITED, B is capped -- change the estimand, not the geometry"))
    mono = all(b[i + 1] >= b[i] - 1e-9 for i in range(len(b) - 1))
    print("B_cal over the sweep: %s  (%s)" % (" -> ".join("%.2f" % v for v in b),
                                              "monotone up" if mono else "NOT monotone"))
    print("B_floor over the sweep: %s  (geometry alone)" %
          " -> ".join("%.1f" % r["B_floor"] for r in rows))
    print("T_cal over the sweep:   %s  (the denominator)" %
          " -> ".join("%.3f" % r["T_cal"] for r in rows))


if __name__ == "__main__":
    main()
