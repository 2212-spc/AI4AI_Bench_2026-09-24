"""Regression test for the route algebra: the generic enumerator must reproduce, from a declaration alone,
the two hand-written shortcut routes that a frontier run was actually observed to take.

Background.  On e05 ws=2 (2026-09-26) gpt-6-astra answered q2 by measuring a slope in proxy units per token
and multiplying it by the run's excess length, twice, missing by 4.16 and 4.06 tolerances.  That route was
added to `e05._q2_slope_routes` by hand *after* the run found it - the hole existed because the rival list
was hand-written.  `scalelab.routes` enumerates the same family mechanically from an `Estimand`, so the test
is: does the mechanical enumeration contain the hand-written values?

    python3 tools/test_routes.py            exit 0 on success, 1 with a diff on failure
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalelab import build as B, routes as RT
from scalelab.bp import e05_matched_kl_far as E5
from scalelab.labs import rllab as R

WS = 2
FAILS = []


def check(name, got, want, rtol=1e-6):
    ok = want is not None and got is not None and abs(got - want) <= rtol * max(1.0, abs(want))
    print("  %-22s got %-14.8g want %-14.8g %s" % (name, got if got is not None else float("nan"),
                                                   want if want is not None else float("nan"),
                                                   "ok" if ok else "MISMATCH"))
    if not ok:
        FAILS.append(name)


def q2_estimand(p):
    """e05 q2 declared as an Estimand.

    Instrument: the *excess mean length* over the zero-distance length, which `len_pen` moves at a fixed KL.
    Response:   the proxy reward at that length.
    Target:     zero excess length - the quality-only limit, which no length penalty reaches because the
                penalty is capped.
    Answer:     the length term's share of the run's own proxy reward.
    """
    pf = R.full(p)
    run = E5._ppo(p["q_KLT"], p["q_ST"])
    len0, tot = pf["len0"], E5._proxy(pf, run)
    x_run = E5._len(pf, run) - len0

    def resp(x):
        """Proxy reward at excess length x, on the mechanism rather than through `len_pen`: the penalty only
        buys a grid of lengths, and the routes need the response as a function of length itself."""
        return pf["wq"] * pf["alpha"] * math.sqrt(E5._kl(pf, run)) + pf["wl"] * math.tanh(x / pf["sl"])

    grid = [E5._len(pf, E5._ppo(p["q_KLT"], p["q_ST"], len_pen=lp)) - len0 for lp in E5.LP_GRID]
    return RT.Estimand(resp=resp, grid=grid, target=0.0, truth=E5._q2_truth(p),
                       to_answer=lambda y: 1.0 - y / tot, op=x_run, name="e05.q2")


def main():
    print("building e05 ws=%d ..." % WS)
    w = B.make_world(E5, WS)
    p = w["p"]
    hand = E5._q2_slope_routes(p)
    truth = E5._q2_truth(p)

    est = q2_estimand(p)
    gen = RT.numerical_routes(est)

    print("\ntruth (length share) = %.6f" % truth)
    print("hand-written routes  = %s" % {k: round(v, 6) for k, v in hand.items()})
    print("\ngenerated routes:")
    for k in sorted(gen):
        print("  %-18s %.6f" % (k, gen[k]))

    print("\n-- the enumerator must contain the hand-written routes --")
    check("R3:local@op ~ local", gen.get("R3:local@op"), hand["local"], rtol=2e-3)
    check("R4:secant@op ~ span", gen.get("R4:secant@op"), hand["span"], rtol=2e-3)
    check("R1:readoff ~ extrapolate", gen.get("R1:readoff"), E5._q2_wrong_extrapolate(p), rtol=2e-3)

    # The certificate, at the tolerance the shipped instance actually carries.
    print("\n-- certificate at the shipped tolerance --")
    inst_tol = None
    try:
        inst = B.build_instance("e05_matched_kl_far", WS)
        inst_tol = inst["cal"]["tol"]["q2"]
    except Exception as exc:                      # a full build is slow; fall back to the floor
        print("  (full build unavailable: %r)" % (exc,))
    tol = inst_tol if inst_tol else E5.FLOOR_SHARE
    cert = RT.certify(truth, tol, dict(gen, **{"skip:baseline": p["q_vbase"], "skip:extrapolate": p["q_vstop"],
                                               "skip:separate": 0.0, "skip:len_pen": 1.0}))
    print("  tol = %.5f%s" % (tol, "" if inst_tol else " (floor; shipped tol is larger)"))
    print("  B = %.2f   nearest = %s   routes = %d" % (cert["B"], cert["nearest"], cert["n_routes"]))
    print("  census: %s" % RT.census(cert["table"] and {r["route"]: r["value"] for r in cert["table"]}))
    for r in cert["table"][:6]:
        print("    %-20s %10.6f  %6.2f T" % (r["route"], r["value"], r["over_T"]))

    # gpt-6-astra's measured miss was 4.16 / 4.06 T on the shipped tolerance.  If the enumerator is right,
    # the span route's over_T should land in that neighbourhood when the shipped tolerance is used.
    if inst_tol:
        span_T = abs(hand["span"] - truth) / inst_tol
        print("\n  span route sits at %.2f T (frontier run missed by 4.16 / 4.06 T)" % span_T)
        if not 3.0 <= span_T <= 5.5:
            FAILS.append("span route does not reproduce the measured 4.1 T miss")

    print("\n%s (%d failure%s)" % ("FAIL" if FAILS else "PASS", len(FAILS), "" if len(FAILS) == 1 else "s"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
