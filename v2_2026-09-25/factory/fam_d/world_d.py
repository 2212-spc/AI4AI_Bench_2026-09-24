"""World generator for family D: an archive shaped so that the value of a *future* measurement is
decided by the day a run sits on rather than by anything about the run.

Family D reuses family B's archive semantics - a sweep, a write-time quality screen, a sink outage, a
per-day scheduler digest that pins each day's sum - because that substrate is what creates the coupling
the family is about: on a day whose total is published, the unrecorded runs are not free of one another.
What is new here is not the mechanism but the question, and the world has to be shaped for the question.

On an ordinary sweep day the answer is boring.  Many unrecorded runs from many configurations share the
day, the day's surplus is comfortably inside every capacity, and recovering one run narrows the contrast
by exactly its own box width over the cell size.  An agent that looks at two or three such days learns a
rule - *value = box width / n, and only members of the two cells matter* - and that rule is wrong
everywhere it matters.  So the generator plants three day shapes that break it, each by a different
mechanism, and each corresponding to a real thing that happens to a sweep:

  * **a failed late campaign (days A and B).**  A curriculum tried at the end of the sweep scored barely
    above chance, and the worker it ran on had lost its results sink, so none of its rows exist.  Those
    runs have enormous boxes - anything from the suite's floor to 1 - but the day's *surplus* is tiny,
    because the true values sit just above the bottom of those boxes.  Recovering one of them therefore
    buys nothing at all: the surplus it frees simply moves onto its neighbours.  What does buy something
    is recovering one of the two small screened runs that happened to share the day and belong to neither
    cell, because their boxes are the capacity that absorbs the surplus.  The widest boxes on the day are
    worthless and the narrowest are decisive.
  * **a successful late campaign (day C).**  Same shape, opposite extreme: the true values sit near the
    *top* of their boxes, so the day's surplus is close to its capacity.  Here recovering a run forces the
    remaining surplus *down*, because what is left can no longer absorb it.  An agent that assumes the
    surplus stays where it was gets this day wrong in a way that no amount of care about box widths
    repairs.
  * **the ordinary days** of the base sweep, which are left exactly as they are, because the family needs
    the misleading regime to be present and plentiful.

Everything the campaign adds is consistent with the archive's own rules: the campaign runs appear in the
plan and the launch log, their worker and window match a documented sink outage, the screened runs that
share their days carry proper retention entries, and the daily digest is recomputed from the true values,
including the ones no file records.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import world_b as W                                                             # noqa: E402

DAY = 86400
LATE_FROM = W.T0 + 20 * DAY                       # 2026-07-11, well inside policy v3
LATE_TO = W.T0 + 24 * DAY
LATE_WORKER = "w09"                               # the worker whose sink died in the second incident
SIDE_WORKER = "w05"                               # still writing rows, so its runs get screened normally
SIDE_CELL = ("code_ms", "c0", "none", "greedy")   # the runs that share the campaign days, in neither cell

# Each campaign is a cell that exists only in the late phase.  `usage` is how far above its own lower
# bound each run's true value sits - that, not the box, is what decides whether recovering it is worth
# anything - and `side` gives the true values of the screened runs that share the day.
#
# The three days are tuned so that, writing t for the day's surplus and cP/cO for the campaign and side
# capacities: days 0 and 1 have t < cO (the side boxes alone can absorb everything, so campaign reveals
# are worthless and side reveals are not), day 1 additionally has t below a *single* side box while day 0
# needs both, which is what makes the cheapest plan non-greedy, and day 2 has t > cP (the day is nearly
# saturated, so any reveal forces the remaining surplus down).  `gen_d.py` re-derives and asserts all of
# this from the emitted archive rather than trusting these numbers.
CAMPAIGNS = [
    {"cell": ("arc_lite", "c4", "dense", "greedy"), "day": 0, "worker": LATE_WORKER,
     "usage": [0.0400, 0.0600, 0.0300, 0.0500, 0.0200, 0.0700, 0.0400, 0.0300],
     "side": [0.0500, 0.0900]},
    {"cell": ("arc_lite", "c4", "dense", "greedy"), "day": 1, "worker": LATE_WORKER,
     "usage": [0.0300, 0.0500, 0.0400, 0.0400, 0.0400],
     "side": [0.1000]},
    {"cell": ("arc_lite", "c5", "dense", "greedy"), "day": 2, "worker": LATE_WORKER,
     "usage": [0.6300, 0.6300, 0.6300],
     "side": [0.3500, 0.3000]},
]


def build(seed=0):
    w = W.build(seed)
    nxt = 1 + max(int(p["run_id"][1:]) for p in w["plan"])
    side_floor = W.floor_of(SIDE_CELL[0])
    for ci, camp in enumerate(CAMPAIGNS):
        s, c, r, d = camp["cell"]
        base = W.floor_of(s)
        day0 = LATE_FROM + camp["day"] * DAY
        assert LATE_FROM <= day0 < LATE_TO
        for j, u in enumerate(camp["usage"]):
            rid = "r%05d" % nxt
            nxt += 1
            done = day0 + 7 * 3600 + 900 * j
            y = round(base + u, 4)
            w["plan"].append({"run_id": rid, "suite": s, "curriculum": c, "retrieval": r,
                              "decoder": d, "planned_start": W.ts(done - 5400)})
            w["launches"].append({"run_id": rid, "started_at": W.ts(done - 5400),
                                  "worker": camp["worker"]})
            w["runtime"].append({"run_id": rid, "finished_at": W.ts(done)})
            w["hidden"][rid] = y
            w["meta"][rid] = {"cell": camp["cell"], "worker": camp["worker"], "start": done - 5400,
                              "done": done, "lost": True}
            # no eval_results row and no retention entry: the sink was down, so the value never arrived
        for j, v in enumerate(camp["side"]):
            rid = "r%05d" % nxt
            nxt += 1
            done = day0 + 11 * 3600 + 1200 * j
            y = round(v, 4)
            assert y < W.policy_floor(done), "a side run must actually fail the screen"
            assert y >= side_floor
            w["plan"].append({"run_id": rid, "suite": SIDE_CELL[0], "curriculum": SIDE_CELL[1],
                              "retrieval": SIDE_CELL[2], "decoder": SIDE_CELL[3],
                              "planned_start": W.ts(done - 4200)})
            w["launches"].append({"run_id": rid, "started_at": W.ts(done - 4200),
                                  "worker": SIDE_WORKER})
            w["runtime"].append({"run_id": rid, "finished_at": W.ts(done)})
            w["retention"].append({"run_id": rid, "dropped_at": W.ts(done),
                                   "reason": "below_retention_floor"})
            w["hidden"][rid] = y
            w["meta"][rid] = {"cell": SIDE_CELL, "worker": SIDE_WORKER, "start": done - 4200,
                              "done": done, "lost": False}

    w["runtime"].sort(key=lambda x: x["run_id"])
    w["retention"].sort(key=lambda x: (x["dropped_at"], x["run_id"]))
    digest = {}
    for f in w["runtime"]:
        day = f["finished_at"][:10]
        a, b = digest.get(day, (0, 0.0))
        digest[day] = (a + 1, round(b + w["hidden"][f["run_id"]], 4))
    w["digest"] = [{"date": k, "n_runs": v[0], "sum_pass_rate": "%.4f" % v[1]}
                   for k, v in sorted(digest.items())]
    w["late"] = {"from": W.ts(LATE_FROM), "to": W.ts(LATE_TO), "worker": LATE_WORKER,
                 "days": sorted({W.ts(LATE_FROM + c["day"] * DAY)[:10] for c in CAMPAIGNS})}
    return w


LATE_CURRICULA = sorted({c["cell"][1] for c in CAMPAIGNS})
CAMPAIGN_DAYS = sorted({W.ts(LATE_FROM + c["day"] * DAY)[:10] for c in CAMPAIGNS})


if __name__ == "__main__":
    w = build(0)
    print("plan", len(w["plan"]), "launched", len(w["launches"]), "results", len(w["results"]),
          "retention", len(w["retention"]), "days", len(w["digest"]), "late", w["late"])
