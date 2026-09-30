"""Recover the sealed arm's censoring mechanism from the agent-visible files alone.

This module exists to make a claim checkable rather than asserted.  The sealed arm of family B ships the
same archive with the explanatory write-up removed: the removal log carries two undocumented internal
codes, and nothing says when the quality check runs, which code is that check, or what the metric is
rounded to.  The task is only fair if those three things are *determined* by the published files.  So the
generator does not certify inferability by author opinion; it runs this program, which reads nothing but
`/app`, and asserts that what comes out is the truth the key was built from.

Three findings, each from a different kind of regularity:

  * **when the check runs.**  If the archiver re-checked its store against the current floor, no surviving
    row could sit below the latest floor.  Many do.  If it checked at some later sweep, the removal
    instants would cluster on sweep times; instead, for one of the two codes every removal instant equals
    the run's own finishing instant, to the second, across dozens of separate worker-days.  That pins the
    check to the moment the row is written.
  * **which code is the check.**  The same code is the one whose recovered values - where the stdout
    scrape happened to catch them - are all strictly below the floor in force when the run finished, while
    no surviving row is.  The other code removes a whole worker-day of rows at one instant, later than
    every one of those runs finished.
  * **what the metric is rounded to.**  The published values have a greatest common divisor, and it is
    not an artefact of a handful of round numbers: the gcd is stable when computed on either source
    (recorded rows, scraped percentages) separately.

The consequence the estimand needs follows from the first two: a row that was removed by the write-time
check is bounded *above* by the floor it failed, less one grid step, since the check is strict and the
value is on the grid; a row removed with its worker-day file had already been written, so it had already
cleared the floor in force when it finished, and is bounded *below*; a run with no row and no removal
entry was never checked at all and is bounded by neither.
"""
import csv
import glob
import json
import os
from fractions import Fraction
from math import gcd

import bounds_b as B


def _rows(app, name):
    with open(os.path.join(app, name)) as fh:
        return list(csv.DictReader(fh))


def discover(app):
    plan = _rows(app, "plan.csv")
    launch = _rows(app, "launch_log.csv")
    results = _rows(app, "eval_results.csv")
    retention = _rows(app, "retention_log.csv")
    runtime = _rows(app, "runtime.csv")
    policy = json.load(open(os.path.join(app, "retention_policy.json")))["versions"]
    suites = json.load(open(os.path.join(app, "suites.json")))
    rec = {}
    for p in sorted(glob.glob(os.path.join(app, "recovered", "*.csv"))):
        for r in _rows(os.path.join(app, "recovered"), os.path.basename(p)):
            rec[r["run_id"]] = Fraction(r["pass_rate_pct"]) / 100
    fin = {r["run_id"]: r["finished_at"] for r in runtime}
    worker = {r["run_id"]: r["worker"] for r in launch}
    suite = {r["run_id"]: r["suite"] for r in plan}
    res = {r["run_id"]: Fraction(r["pass_rate"]) for r in results}

    def floor_at(iso):
        t = B.iso(iso)
        for p in policy:
            if B.iso(p["from"]) <= t < B.iso(p["to"]):
                return Fraction(str(p["floor"]))
        return Fraction(0)

    def base(rid):
        sp = suites[suite[rid]]
        return Fraction(0) if sp["scoring"] == "normalized" else Fraction(str(sp["random_guess"]))

    # -- is the check continuous, retroactive, or done once when the row is written? -------------------
    latest = Fraction(str(policy[-1]["floor"]))
    below_latest = sum(1 for rid, v in res.items() if v < latest)
    below_own = sum(1 for rid, v in res.items() if v < floor_at(fin[rid]))

    # -- profile each undocumented code ---------------------------------------------------------------
    codes = {}
    for r in retention:
        codes.setdefault(r["reason"], []).append(r)
    prof = {}
    for code, ent in sorted(codes.items()):
        at_finish = sum(1 for e in ent if e["dropped_at"] == fin[e["run_id"]])
        later = sum(1 for e in ent if B.iso(e["dropped_at"]) > B.iso(fin[e["run_id"]]))
        seen = [e["run_id"] for e in ent if e["run_id"] in rec]
        under = [r for r in seen if rec[r] < floor_at(fin[r])]
        prof[code] = {
            "n": len(ent),
            "removal_instant_equals_the_finish": at_finish,
            "removal_instant_is_later": later,
            "distinct_removal_instants": len({e["dropped_at"] for e in ent}),
            "distinct_worker_days": len({(worker[e["run_id"]], fin[e["run_id"]][:10]) for e in ent}),
            "recovered_values_seen": len(seen),
            "recovered_values_below_the_finish_floor": len(under),
        }

    screen = [c for c, p in prof.items()
              if p["removal_instant_equals_the_finish"] == p["n"] and p["distinct_worker_days"] > 1
              and p["recovered_values_seen"] > 0
              and p["recovered_values_below_the_finish_floor"] == p["recovered_values_seen"]]
    batch = [c for c, p in prof.items()
             if p["removal_instant_is_later"] == p["n"] and p["distinct_removal_instants"] == 1
             and p["distinct_worker_days"] == 1]

    # -- the rounding grid ----------------------------------------------------------------------------
    def grid_of(vals):
        g = 0
        for v in vals:
            u = v * 1000000
            if u.denominator != 1:
                return None
            g = gcd(g, u.numerator)
        return Fraction(g, 1000000) if g else None

    grids = {"recorded": grid_of(res.values()), "scraped": grid_of(rec.values())}
    grids["both"] = grid_of(list(res.values()) + list(rec.values()))

    # -- the three missing-data classes, and the side each is bounded on ------------------------------
    drop = {r["run_id"]: r for r in retention}
    launched = {r["run_id"] for r in launch}
    cls = {"screened": [], "worker_day_lost": [], "no_row_no_entry": []}
    for rid in sorted(launched):
        if rid in res:
            continue
        e = drop.get(rid)
        k = ("screened" if e and e["reason"] in screen else
             "worker_day_lost" if e else "no_row_no_entry")
        cls[k].append(rid)
    bounded = {"screened": "above", "worker_day_lost": "below", "no_row_no_entry": "neither"}
    # the last of those is not an assumption: some runs in it have scraped values strictly below the
    # floor in force when they finished, so they demonstrably never passed the check.
    refuting = [rid for rid in cls["no_row_no_entry"] if rid in rec and rec[rid] < floor_at(fin[rid])]
    lifted = [rid for rid in cls["worker_day_lost"] if floor_at(fin[rid]) > base(rid)]

    return {
        "surviving_rows_below_the_latest_floor": below_latest,
        "surviving_rows_below_their_own_finish_floor": below_own,
        "code_profiles": prof,
        "screen_code": screen[0] if len(screen) == 1 else None,
        "batch_code": batch[0] if len(batch) == 1 else None,
        "grid": float(grids["both"]) if grids["both"] else None,
        "grid_per_source": {k: (float(v) if v else None) for k, v in grids.items()},
        "class_sizes": {k: len(v) for k, v in cls.items()},
        "bounded_side": bounded,
        "runs_refuting_a_bound_on_the_unlogged_class": len(refuting),
        "worker_day_lost_runs_whose_bound_actually_lifts": len(lifted),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(discover(sys.argv[1] if len(sys.argv) > 1 else "/app"), indent=1))
