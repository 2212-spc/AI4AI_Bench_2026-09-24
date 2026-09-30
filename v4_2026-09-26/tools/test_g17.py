"""G17 regression: the gate must fail the defect it was written for, not just pass everything.

    python3 tools/test_g17.py

Two positive controls and one negative.  `build_instance` is run three times on e05 world 2:

  * as shipped                      -> G17 passes (q5's key is the integer the answer format allows)
  * with q5's key put back to the fraction 2162.6907  -> G17 must fail q5, because the 2.25*p90 rule then
    widens the band to 0.696 and parks the oracle at 0.44 of it
  * t02, whose edge finder leaves a deterministic residual it cannot afford to shrink -> G17 must pass,
    since that residual sits inside a declared floor rather than inside a band it bought

A gate that cannot be shown to fail is not evidence of anything, so this runs in the same sweep as the
builds.  Exit status is non-zero if any of the three comes out the other way.
"""
import os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from scalelab import build as B                          # noqa: E402
from scalelab.bp import e05_matched_kl_far as E05        # noqa: E402

fails = []


def check(label, cond, detail=""):
    print("%-58s %s %s" % (label, "ok" if cond else "FAIL", detail))
    if not cond:
        fails.append(label)


g = B.build_instance("e05_matched_kl_far", 2)["gates"]["G17_oracle_bias"]
check("e05 ws2 as shipped: G17 passes", g["pass"], str(g["biased_items"]))
check("e05 ws2 q5 has no deterministic offset left",
      "q5" not in g["median_over_p90"], str(g["median_over_p90"].get("q5")))

orig = E05.items
try:
    def fractional_key(p, ctx, tol=None):               # the defect, reintroduced on purpose
        its = orig(p, ctx, tol)
        for it in its:
            if it["id"] == "q5":
                it["key"] = {"lo": p["q_nfull"], "hi": p["q_nfull"]}
        return its
    E05.items = fractional_key
    g = B.build_instance("e05_matched_kl_far", 2)["gates"]["G17_oracle_bias"]
    bad = g["biased_items"]
    check("fractional q5 key: G17 fails", not g["pass"], str(sorted(bad)))
    check("fractional q5 key: q5 is the flagged item", list(bad) == ["q5"], str(list(bad)))
    check("fractional q5 key: flagged as ratio 1.0, band above floor",
          bad.get("q5", {}).get("ratio") == 1.0 and bad.get("q5", {}).get("tol_above_floor") is True,
          str(bad.get("q5")))
finally:
    E05.items = orig

g = B.build_instance("t02_stability_edge", 2)["gates"]["G17_oracle_bias"]
det = {k: v for k, v in g["median_over_p90"].items() if v["ratio"] > 0.9}
check("t02 ws2: deterministic residual is detected", bool(det), str(sorted(det)))
check("t02 ws2: G17 still passes (residual inside a declared floor)", g["pass"],
      str({k: v["tol_above_floor"] for k, v in det.items()}))

print("\n%d/%d checks passed" % (7 - len(fails), 7))
sys.exit(1 if fails else 0)
