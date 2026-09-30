"""World generator for family B - `b-bounds-a`: sharp partial identification over a broken eval archive.

Why this family exists.  Family S was solved by gpt-6-astra because the truth lived inside a parametric
class the model could guess (truncated normal + a batch-level recording rate); once it had guessed the
class it certified itself with a 20-replication parametric bootstrap against the published tolerances.
So family B removes the model entirely.  There is no sampling distribution to bootstrap: the estimand is
a *finite-population* mean over the runs that were actually launched, and the answer is not a point but
the sharp interval of values that mean could take given everything the archive does and does not record.
Simulation cannot certify a wrong answer here, because there is nothing random left to simulate - what is
left is whether the agent partitioned every launched run into the right evidence class and took the
extreme feasible completion in each class.  Valid-but-loose bounds fail exactly like wrong ones.

The archive is deliberately shaped so that four independent bookkeeping mistakes each move an endpoint:

  * `plan.csv` lists planned runs, `launch_log.csv` lists the ones that actually started - the denominator
    is the launched set, and planned-not-launched runs must not dilute it;
  * the archiver screens every row against the quality floor *in force when the row is written* and keeps
    the timestamp, not the floor, so a per-run bound needs a temporal join against `retention_policy.json`
    (half-open intervals, one run sits exactly on a boundary).  The screen cuts both ways, which is the
    move the family is really about: a row that was deleted for being under the floor is bounded *above*
    by it, and a row that was written and then lost for an unrelated reason - a whole shard file
    disappeared in a migration - is bounded *below* by it, because it had already passed the screen.  A
    run whose row never reached the archive at all was never screened and is bounded by neither;
  * a logger outage lost result rows for reasons unrelated to the outcome, and a secondary sink under
    `recovered/` holds some of those values in percent units - including duplicates of rows that survived,
    values for runs that were never launched, and, most consequentially, values for runs the screen had
    deleted, which converts a bounded run into a known one;
  * two of the three suites report normalised scores, whose floor is 0, while the third is bounded below
    by its random-guess rate - so the lower endpoint depends on a documented two-step rule.
"""
import hashlib
import json
import os

SUITES = {
    "arc_lite": {"random_guess": 0.25, "scoring": "raw"},
    "gsm_plus": {"random_guess": 0.02, "scoring": "normalized"},
    "code_ms": {"random_guess": 0.05, "scoring": "normalized"},
}
CURRICULA = ["c0", "c1", "c2", "c3"]
RETRIEVAL = ["none", "bm25", "dense"]
DECODER = ["greedy", "sample"]
KNOBS = ["suite", "curriculum", "retrieval", "decoder"]

PER_CELL = 36
T0 = 1782000000                                   # 2026-06-21T00:00:00Z, start of the sweep
POLICY = [                                        # half-open [from_ts, to_ts) in the published json
    {"version": "v1", "from": T0, "to": T0 + 6 * 86400, "floor": 0.18},
    {"version": "v2", "from": T0 + 6 * 86400, "to": T0 + 13 * 86400, "floor": 0.27},
    {"version": "v3", "from": T0 + 13 * 86400, "to": T0 + 40 * 86400, "floor": 0.36},
]
OUTAGE = (T0 + 6 * 86400, T0 + 12 * 86400 + 43200)          # logger outage window, keyed on *start* time
OUTAGE_WORKERS = {"w03", "w07"}                             # only these two workers lost their sink
# The scheduler pinned each configuration to a worker, so an outage takes out whole slices of the grid
# rather than a uniform 2% everywhere.  Two cells were run as a single burst inside the outage window.
BURST_CELLS = {("code_ms", "c1", "bm25", "sample"), ("arc_lite", "c2", "dense", "greedy")}


def ts(x):
    """Epoch seconds -> the ISO-8601 Z form used everywhere in the archive."""
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(x)))


def _u(*parts):
    """Deterministic uniform(0,1) from a tuple of labels - no global RNG state, so any subset of the
    world can be regenerated in isolation and the files are byte-stable across runs."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _norm(*parts):
    """Box-Muller from two independent deterministic uniforms."""
    import math
    u1 = min(max(_u("n1", *parts), 1e-12), 1 - 1e-12)
    u2 = _u("n2", *parts)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)


def cells():
    out = []
    for s in SUITES:
        for c in CURRICULA:
            for r in RETRIEVAL:
                for d in DECODER:
                    out.append((s, c, r, d))
    return out


# Cells that were planned but never started at all - they exist in plan.csv and nowhere else, and some of
# them also appear in the recovery shards, which is the denominator trap.
NEVER_LAUNCHED = {
    ("arc_lite", "c3", "dense", "sample"),
    ("gsm_plus", "c0", "bm25", "sample"),
    ("code_ms", "c2", "none", "greedy"),
    ("code_ms", "c3", "dense", "greedy"),
}


def cell_center(cell):
    """Hidden mean pass_rate for a cell.  Never revealed; used only to synthesise run outcomes."""
    s, c, r, d = cell
    base = {"arc_lite": 0.52, "gsm_plus": 0.44, "code_ms": 0.31}[s]
    base += {"c0": 0.0, "c1": 0.035, "c2": 0.062, "c3": 0.049}[c]
    base += {"none": 0.0, "bm25": 0.028, "dense": 0.055}[r]
    base += {"greedy": 0.0, "sample": -0.019}[d]
    base += 0.045 * (_u("cellfx", *cell) - 0.5)               # idiosyncratic per-cell offset
    if (s, r) == ("code_ms", "dense"):
        base += 0.037                                          # one real interaction worth finding
    return base


def floor_of(suite):
    """Documented two-step rule: normalised suites are floored at 0, raw suites at their guess rate."""
    sp = SUITES[suite]
    return 0.0 if sp["scoring"] == "normalized" else sp["random_guess"]


def policy_floor(t):
    for p in POLICY:
        if p["from"] <= t < p["to"]:
            return p["floor"]
    return None


def build(seed=0):
    """Returns the whole archive plus the hidden values, all as lists of plain dicts."""
    plan, launches, hidden, meta = [], [], {}, {}
    i = 0
    for cell in cells():
        s, c, r, d = cell
        fl = floor_of(s)
        ctr = cell_center(cell)
        for k in range(PER_CELL):
            i += 1
            rid = "r%05d" % i
            if cell in BURST_CELLS:
                start = OUTAGE[0] + 600 + int((OUTAGE[1] - OUTAGE[0] - 1200) * _u("start", seed, rid))
            else:
                start = T0 + int(86400 * 18 * _u("start", seed, rid)) + int(600 * _u("m", seed, rid))
            plan.append({"run_id": rid, "suite": s, "curriculum": c, "retrieval": r,
                         "decoder": d, "planned_start": ts(start)})
            if cell in NEVER_LAUNCHED:
                continue
            if _u("launch", seed, rid) < 0.045:                # planned, never actually started
                continue
            pinned = "w%02d" % (1 + int(10 * _u("pin", *cell)))
            worker = pinned if _u("wk", seed, rid) < 0.82 else "w%02d" % (1 + int(10 * _u("wk2", rid)))
            if cell in BURST_CELLS:
                worker = "w03" if _u("wk", seed, rid) < 0.93 else "w05"
            launches.append({"run_id": rid, "started_at": ts(start), "worker": worker})
            y = ctr + 0.085 * _norm(seed, rid)
            y = min(0.995, max(fl, round(y, 4)))
            hidden[rid] = y
            meta[rid] = {"cell": cell, "worker": worker, "start": start,
                         "done": start + 3600 + int(7200 * _u("dur", seed, rid)),
                         "lost": (OUTAGE[0] <= start < OUTAGE[1]) and worker in OUTAGE_WORKERS}

    # Exactly one run finishes on a policy boundary instant.  Because the published intervals are
    # half-open, that run is screened against the *later*, higher floor - which is also what decides
    # whether its row survives at all, so the convention is not a cosmetic detail.  The run is taken from
    # the lowest-scoring suite and its value re-drawn into the gap between the two floors; since the row
    # is deleted either way the value is never published, so the edit leaves no observable trace beyond
    # the one it is meant to leave.  It must also be a run the recovery scrape misses - a recovered value
    # would pin it exactly and the convention would stop mattering.
    edge = POLICY[1]["from"]
    boundary = None
    for rid in sorted(meta):
        m = meta[rid]
        if (not m["lost"] and m["cell"][0] == "code_ms" and edge - 10800 <= m["done"] < edge
                and _u("rec_ret", seed, rid) >= 0.22):
            m["done"] = edge
            hidden[rid] = round(POLICY[0]["floor"] + 0.088 * _u("edge", seed, rid), 4)
            boundary = rid
            break
    assert boundary is not None, "no run available to sit on the policy boundary"

    # The archiver screens each result row against the floor in force *when the row is written*, and
    # refuses to keep it if it is under.  Two consequences the archive never states and the agent has to
    # notice: a surviving row is evidence its value cleared the floor of its own finishing day (useless
    # while the row is there - decisive once the row is deleted for an unrelated reason), and a run whose
    # row never reached the archiver at all was never screened, so it carries no such evidence.  Both
    # halves are checkable in the published data, which is what licenses the deduction.
    results, retention, recovered = [], [], []
    for rid in sorted(meta):
        m, y = meta[rid], hidden[rid]
        if m["lost"]:                                          # the writer died; nothing was screened
            if _u("rec_out", seed, rid) < (0.18 if m["cell"] in BURST_CELLS else 0.55):
                recovered.append({"run_id": rid, "pass_rate_pct": round(100 * y, 2)})
        elif y < policy_floor(m["done"]):
            retention.append({"run_id": rid, "dropped_at": ts(m["done"]),
                              "reason": "below_retention_floor"})
            if _u("rec_ret", seed, rid) < 0.22:                # ... but the secondary sink kept the value
                recovered.append({"run_id": rid, "pass_rate_pct": round(100 * y, 2)})
        else:
            results.append({"run_id": rid, "pass_rate": "%.4f" % y})
            if _u("rec_dup", seed, rid) < 0.03:                # duplicate of a row that survived
                recovered.append({"run_id": rid, "pass_rate_pct": round(100 * y, 2)})

    # A second deletion mechanism, and the one that carries the family's real difficulty: the archiver
    # stores rows in one file per worker per day, and a storage incident lost one whole file.  The rows it
    # held had already been written, so they had already cleared that day's floor; the loss itself is
    # blind to the values.  Selection is therefore by (worker, day) only - never by score - and the day is
    # taken from the strictest policy era so that the surviving-row deduction actually bites.
    have = sorted(r["run_id"] for r in results)
    by = {}
    for rid in have:
        by.setdefault((ts(meta[rid]["done"])[:10], meta[rid]["worker"]), []).append(rid)
    shard = None
    for key in sorted(by):
        ids = by[key]
        if (key[0] >= ts(POLICY[2]["from"])[:10] and 12 <= len(ids) <= 20
                and len({meta[r]["cell"] for r in ids}) >= 4):
            shard = {"date": key[0], "worker": key[1], "run_ids": ids}
            break
    assert shard is not None, "no (worker, day) file in the strict era has the right size"
    gone = set(shard["run_ids"])
    results[:] = [r for r in results if r["run_id"] not in gone]
    recovered[:] = [r for r in recovered if r["run_id"] not in gone]
    drop_t = (max(meta[r]["done"] for r in gone) // 86400 + 2) * 86400 + 15420
    shard["dropped_at"] = ts(drop_t)
    for rid in sorted(gone):
        retention.append({"run_id": rid, "dropped_at": ts(drop_t), "reason": "shard_file_lost"})

    # Recovery shards also carry rows for runs that were never launched (stale planning artefacts) and a
    # handful of exact duplicates, so the shards cannot be trusted as a run list.
    never = [p["run_id"] for p in plan if p["run_id"] not in hidden]
    for j, rid in enumerate(never[:: max(1, len(never) // 14)][:14]):
        recovered.append({"run_id": rid, "pass_rate_pct": round(100 * (0.2 + 0.5 * _u("ghost", rid)), 2)})
    for rec in list(recovered)[:: max(1, len(recovered) // 9)][:9]:
        recovered.append(dict(rec))

    # The scheduler's own accounting is untouched by the results-writer outage and predates the
    # retention sweep, so its daily totals still cover runs whose rows are gone.  A day's total pins the
    # *sum* of the values that vanished that day, which is what makes the cells compete: two
    # configurations that ran on the same day cannot both be pushed to their individual extremes.
    finished = [{"run_id": rid, "finished_at": ts(meta[rid]["done"])} for rid in sorted(meta)]
    digest = {}
    for f in finished:
        d = f["finished_at"][:10]
        a, b = digest.get(d, (0, 0.0))
        digest[d] = (a + 1, round(b + hidden[f["run_id"]], 4))
    digest = [{"date": d, "n_runs": v[0], "sum_pass_rate": "%.4f" % v[1]}
              for d, v in sorted(digest.items())]

    recovered.sort(key=lambda x: (_u("shuf", x["run_id"], x["pass_rate_pct"])))
    results.sort(key=lambda x: x["run_id"])
    retention.sort(key=lambda x: (x["dropped_at"], x["run_id"]))
    return {"plan": plan, "launches": launches, "results": results, "retention": retention,
            "recovered": recovered, "hidden": hidden, "meta": meta, "shard": shard,
            "runtime": finished, "digest": digest, "boundary": boundary}


def _iso(s):
    import calendar
    import time
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


if __name__ == "__main__":
    w = build(0)
    print("plan", len(w["plan"]), "launched", len(w["launches"]), "results", len(w["results"]),
          "retention", len(w["retention"]), "recovered", len(w["recovered"]))
    print("boundary", w["boundary"], "digest days", len(w["digest"]), "runtime", len(w["runtime"]))
    print("shard", {k: (v if k != "run_ids" else len(v)) for k, v in w["shard"].items()},
          "cells", len({w["meta"][r]["cell"] for r in w["shard"]["run_ids"]}),
          "floor", policy_floor(_iso(w["shard"]["date"] + "T12:00:00Z")))
