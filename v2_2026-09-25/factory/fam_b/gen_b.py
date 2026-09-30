"""Build, certify and export `b-bounds-a`.

Pipeline, in the order the gates run:

  G1  emit the archive from the world generator;
  G2  *validity* - the hidden per-run values, which the agent never sees, must lie inside the published
      support of their own run, and the hidden cell mean inside the published interval, for all 68
      launched cells (not just the queried ones);
  G3  *sharpness* - each endpoint must be attained by an explicit feasible completion;
  G4  *independent agreement* - `truth_b.py`, written separately with exact rational arithmetic, must
      reproduce every published item to 1e-6 (the published endpoints are rounded to six decimals);
  G5  *item selection* - queries are chosen greedily so that every decoy in `bounds_b.CANDIDATES` misses
      at least MIN_FAIL items; the items exist to kill the decoys, not the other way round;
  G6  *coverage* - the decision items must contain all three labels, and complete-case point thinking
      must change at least two of them;
  G7  *tolerance* - the published tolerance must be far below the smallest gap any decoy has to clear;
  G8  *inferability* - the one mechanism the agent has to discover rather than read (a written row has
      already passed the quality screen, so losing it later bounds it from below) must be confirmable and
      refutable from the published files alone, and must move the key by far more than the tolerance.
"""
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
import bounds_b as B                                                          # noqa: E402
import harbor as H                                                           # noqa: E402
import truth_b as T                                                          # noqa: E402
import world_b as W                                                          # noqa: E402

TOL = 5e-4
MIN_FAIL = 2
N_CELL, N_CONTRAST, N_DECISION = 8, 8, 8
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/b-bounds-a"

INCIDENTS = """# Incident notes for the 2026-06 evaluation sweep

**INC-4471 - evaluation sink outage (workers w03, w07).**  Between 2026-06-27T00:00:00Z and
2026-07-03T12:00:00Z the primary results writer on workers w03 and w07 dropped its connection.  Runs that
*started* in that window on those two workers never got a row into the archive at all - the number never
left the worker.  The failure was in the writer, not in the evaluation: it does not depend in any way on
what the run scored.

Some of those values were later scraped out of the per-worker stdout tails and are kept under
`recovered/`, one shard per collection pass.  The stdout banner prints the score as a percentage, so the
shards carry `pass_rate_pct`; it is the same quantity as `pass_rate`, in different units.  The scrape was
opportunistic:

  * it also re-captured runs whose primary row was fine, so a run can appear in both places;
  * it also captured runs whose row had already been removed by the quality screen;
  * the shards were assembled from planning tickets, so they contain a few run ids for runs that were
    never launched at all, and a few rows appear twice.

Where a run appears more than once the values agree; treat the shards as a value source, never as a run
list.

**Write-time quality screen.**  A result row is written the moment its run finishes, and the archiver
checks it against the quality floor in force at that moment; a row under the floor is not kept.  The
removals are listed in `retention_log.csv` with reason `below_retention_floor` and the instant of the
check, and the floors are versioned in `retention_policy.json`.  The screen has run unchanged for the
whole sweep.  It is part of the archiver, so the only thing it ever inspects is a row that reached the
archive.

**INC-4506 - lost shard file.**  The archiver keeps result rows in one file per worker per UTC day.
During a storage migration the file holding worker %(worker)s's rows for %(date)s was lost along with its
replica; the rows it held are gone and are listed in `retention_log.csv` with reason `shard_file_lost`.
The migration walked the store by file name.  Nothing in it looked inside a file, so which rows were lost
has nothing to do with what any of them said.

**Scheduler accounting.**  The scheduler keeps its own ledger, on a different host from the results
writer and untouched by either incident: `runtime.csv` records when each launched run finished, and
`daily_digest.csv` records, for each UTC day, how many runs finished that day and what their `pass_rate`
values added up to.  The digest is written at 23:59 from the scores the scheduler received directly from
each worker, so its totals still cover runs whose rows the archive screened out, lost, or never received.
`daily_digest.csv` is exact to the four decimals it prints.

**Not affected.**  `plan.csv`, `launch_log.csv`, `retention_log.csv`, `runtime.csv` and
`daily_digest.csv` are complete for the whole sweep.
"""


METRIC = """# `pass_rate`

Every run reports exactly one number, `pass_rate`, on the suite named in its plan row.  What the harness
writes depends on how the suite is scored, which `suites.json` records along with `random_guess`, the
score an uninformed answer earns on that suite:

  * `scoring: "raw"` - the fraction of the suite's items the run passed, reported as is, except that the
    harness never emits a value below `random_guess`: a run that scores under the baseline is written out
    at the baseline.
  * `scoring: "normalized"` - the raw fraction rescaled so that `random_guess` maps to 0 and a perfect
    run maps to 1, clipped below at 0.

Either way the number lies in [0, 1] and is rounded to four decimals before anything else sees it, so
every `pass_rate` in this archive - recorded, screened out, or lost - is a multiple of 0.0001.
"""


def emit(app, w):
    os.makedirs(os.path.join(app, "recovered"), exist_ok=True)
    H.csv_rows(os.path.join(app, "plan.csv"),
               ["run_id", "suite", "curriculum", "retrieval", "decoder", "planned_start"], w["plan"])
    H.csv_rows(os.path.join(app, "launch_log.csv"), ["run_id", "started_at", "worker"], w["launches"])
    H.csv_rows(os.path.join(app, "eval_results.csv"), ["run_id", "pass_rate"], w["results"])
    H.csv_rows(os.path.join(app, "retention_log.csv"), ["run_id", "dropped_at", "reason"], w["retention"])
    shards = [[] for _ in range(4)]
    for i, r in enumerate(w["recovered"]):
        shards[i % 4].append(r)
    for i, sh in enumerate(shards):
        H.csv_rows(os.path.join(app, "recovered", "shard_%02d.csv" % i), ["run_id", "pass_rate_pct"], sh)
    json.dump({"metric": "pass_rate", "note": "intervals are half-open: [from, to)",
               "versions": [{"version": p["version"], "from": W.ts(p["from"]), "to": W.ts(p["to"]),
                             "floor": p["floor"]} for p in W.POLICY]},
              open(os.path.join(app, "retention_policy.json"), "w"), indent=1)
    json.dump(W.SUITES, open(os.path.join(app, "suites.json"), "w"), indent=1)
    H.csv_rows(os.path.join(app, "runtime.csv"), ["run_id", "finished_at"], w["runtime"])
    H.csv_rows(os.path.join(app, "daily_digest.csv"), ["date", "n_runs", "sum_pass_rate"], w["digest"])
    H.w(os.path.join(app, "incidents.md"),
        INCIDENTS % {"worker": w["shard"]["worker"], "date": w["shard"]["date"]})
    H.w(os.path.join(app, "metric_card.md"), METRIC)


def launched_cells(arch):
    out = {}
    for r in arch.plan:
        c = tuple(r[k] for k in B.KNOBS)
        if r["run_id"] in arch.launched:
            out[c] = out.get(c, 0) + 1
    return {c: n for c, n in out.items() if n > 0}


def as_spec(c):
    return dict(zip(B.KNOBS, c))


def build_items(arch, cells):
    """Every syntactically legal item, before selection."""
    items = {"cells": [], "contrasts": [], "decisions": []}
    for c in sorted(cells):
        items["cells"].append({"id": None, "cell": as_spec(c)})
    pairs = []
    for a in sorted(cells):
        for b in sorted(cells):
            if a < b and sum(1 for i in range(4) if a[i] != b[i]) == 1 and a[0] == b[0]:
                pairs.append((a, b))
    for a, b in pairs:
        items["contrasts"].append({"id": None, "from": as_spec(a), "to": as_spec(b)})
        for m in (0.0, 0.01, 0.02, 0.04):
            items["decisions"].append({"id": None, "from": as_spec(a), "to": as_spec(b), "margin": m})
    return items


def eval_one(ctx, kind, q):
    if kind == "cells":
        return ctx.cell(tuple(q["cell"][k] for k in B.KNOBS))
    d = ctx.diff(tuple(q["from"][k] for k in B.KNOBS), tuple(q["to"][k] for k in B.KNOBS))
    if d is None:
        return None
    return d if kind == "contrasts" else B.decide(d, q["margin"], ctx.spec)


def differs(kind, ref, got):
    if got is None:
        return True
    if kind == "decisions":
        return ref != got
    return abs(ref[0] - got[0]) > TOL or abs(ref[1] - got[1]) > TOL


def gap(kind, ref, got):
    if kind == "decisions":
        return 1.0 if ref != got else 0.0
    if got is None:
        return 1.0
    return max(abs(ref[0] - got[0]), abs(ref[1] - got[1]))


def select(arch, items):
    """Greedy set cover over the decoys, weighted so that scarcity beats volume.

    Scoring an item by how many decoys it kills is the obvious rule and it is the wrong one: a decoy that
    only two or three items in the whole grid can expose - the half-open policy boundary, which turns on a
    single run - loses every comparison to a decoy that half the grid exposes, and the shipped item set
    ends up unable to tell whether the agent got the rare rule right.  Each decoy therefore carries a
    weight of 1/(number of items that can kill it), so an item that covers a nearly-uncoverable decoy
    outranks an item that piles on to well-covered ones.
    """
    ctx = B.Ctx(arch, B.REF)
    ref = {k: [eval_one(ctx, k, q) for q in items[k]] for k in items}
    dec = {}
    for name, spec in B.CANDIDATES.items():
        c = B.Ctx(arch, spec)
        dec[name] = {k: [eval_one(c, k, q) for q in items[k]] for k in items}
    killable = {n: sum(1 for k in items for i in range(len(items[k]))
                       if ref[k][i] is not None and differs(k, ref[k][i], dec[n][k][i]))
                for n in B.CANDIDATES}
    weight = {n: 1.0 / max(1, killable[n]) for n in B.CANDIDATES}
    need = {n: MIN_FAIL for n in B.CANDIDATES}
    chosen = {"cells": [], "contrasts": [], "decisions": []}
    cap = {"cells": N_CELL, "contrasts": N_CONTRAST, "decisions": N_DECISION}
    used_cells, used_pairs, gaps = set(), set(), []
    rng = random.Random(7)
    order = [k for k in ("cells", "contrasts", "decisions")]

    # Label coverage first: a decision set that never says "yes" would let an agent notice that the
    # cautious label is always safe.  One item per label is seeded before the greedy pass, each chosen
    # to kill as much rare decoy weight as it can while carrying its label.
    for lab in ("yes", "no", "cannot_tell"):
        pick = None
        for i, q in enumerate(items["decisions"]):
            if ref["decisions"][i] != lab:
                continue
            sig = (tuple(sorted(q["from"].items())), tuple(sorted(q["to"].items())))
            if ("decisions", sig) in used_pairs:
                continue
            kills = [n for n in need if need[n] > 0 and differs("decisions", ref["decisions"][i],
                                                                dec[n]["decisions"][i])]
            sc = (sum(weight[n] for n in kills), len(kills), rng.random())
            if pick is None or sc > pick[0]:
                pick = (sc, i, q, sig, kills)
        if pick is None:
            continue
        _, i, q, sig, kills = pick
        q = dict(q)
        q["id"] = "d%02d" % (len(chosen["decisions"]) + 1)
        chosen["decisions"].append(q)
        used_pairs.add(("decisions", sig))
        for n in kills:
            need[n] -= 1

    while sum(len(v) for v in chosen.values()) < sum(cap.values()):
        best = None
        for k in order:
            if len(chosen[k]) >= cap[k]:
                continue
            for i, q in enumerate(items[k]):
                if ref[k][i] is None:
                    continue
                sig = (tuple(sorted(q.get("cell", {}).items())) if k == "cells" else
                       (tuple(sorted(q["from"].items())), tuple(sorted(q["to"].items()))))
                if k == "cells" and sig in used_cells:
                    continue
                if k != "cells" and (k, sig) in used_pairs:
                    continue
                kills = [n for n in need if need[n] > 0 and differs(k, ref[k][i], dec[n][k][i])]
                score = (sum(weight[n] for n in kills), len(kills),
                         min([gap(k, ref[k][i], dec[n][k][i]) for n in kills] or [0]), rng.random())
                if best is None or score > best[0]:
                    best = (score, k, i, q, sig, kills)
        if best is None:
            break
        _, k, i, q, sig, kills = best
        q = dict(q)
        q["id"] = {"cells": "b%02d", "contrasts": "k%02d", "decisions": "d%02d"}[k] % (len(chosen[k]) + 1)
        chosen[k].append(q)
        (used_cells if k == "cells" else used_pairs).add(sig if k == "cells" else (k, sig))
        for n in kills:
            need[n] -= 1
        gaps += [gap(k, ref[k][i], dec[n][k][i]) for n in kills if k != "decisions"]
    return chosen, need, gaps, killable


def certify(arch, w, queries):
    """G2-G4 and G6-G7.  Returns the certificate dict; raises on a hard violation.

    G2 asks whether the published constraints are *true*: every hidden value inside the support the
    archive implies for it, every hidden configuration mean inside the interval the task will grade, every
    day's hidden values adding up to the digest total, and the reference optimisation never clamping an
    infeasible surplus.  G3 asks whether the intervals are *tight*, and it does so the only way that
    settles it - by building the assignment that is supposed to attain each endpoint and re-checking that
    assignment against every constraint in the archive before re-deriving the endpoint from it.
    """
    cert = {}
    hidden = w["hidden"]
    cells = launched_cells(arch)
    ctx = B.Ctx(arch, B.REF)
    worst_run, worst_cell = 0.0, 0.0
    for rid in sorted(arch.launched):
        a_, b_ = ctx.sup[rid]
        y = hidden[rid]
        assert a_ - 1e-9 <= y <= b_ + 1e-9, ("run outside its published support", rid, a_, y, b_)
        worst_run = max(worst_run, max(a_ - y, y - b_))
    for c in cells:
        pop = arch.population(c, B.REF)
        lo, hi = ctx.cell(c)
        m = sum(hidden[r] for r in pop) / len(pop)
        assert lo - 1e-9 <= m <= hi + 1e-9, ("cell mean outside interval", c, lo, m, hi)
        worst_cell = max(worst_cell, min(m - lo, hi - m))
    day_err = 0.0
    for d, rids in ctx.unk.items():
        day_err = max(day_err, abs(sum(hidden[r] for r in rids) - ctx.R[d]))
    assert day_err < 1e-9 and ctx.clamped == 0.0, (day_err, ctx.clamped)
    cert["G2_validity"] = {"runs_checked": len(arch.launched), "cells_checked": len(cells),
                           "worst_support_violation": worst_run, "worst_day_total_error": day_err,
                           "reference_surplus_clamped": ctx.clamped,
                           "tightest_cell_slack": round(worst_cell, 6)}

    # G3: the endpoints are attained.  Build the witness, re-check it against *every* constraint, then
    # recompute the estimand from it - a sharp interval that no legal assignment reaches is a wrong one.
    sharp = {}
    for kind, q in (("cell", queries["cells"][0]), ("contrast", queries["contrasts"][0])):
        for side, mx in (("lo", False), ("hi", True)):
            if kind == "cell":
                cell = tuple(q["cell"][k] for k in B.KNOBS)
                n, ksum, weight, _ = ctx.parts(cell)
                want = ctx.cell(cell)[0 if side == "lo" else 1]
            else:
                ca = tuple(q["from"][k] for k in B.KNOBS)
                cb = tuple(q["to"][k] for k in B.KNOBS)
                na, ka, wa, _ = ctx.parts(ca)
                nb, kb, wb, _ = ctx.parts(cb)
                weight = {}
                for r, x in wb.items():
                    weight[r] = weight.get(r, 0.0) + x / nb
                for r, x in wa.items():
                    weight[r] = weight.get(r, 0.0) - x / na
                want = ctx.diff(ca, cb)[0 if side == "lo" else 1]
            y = ctx.witness(weight, mx)
            assert all(ctx.sup[r][0] - 1e-9 <= y[r] <= ctx.sup[r][1] + 1e-9 for r in y), "witness support"
            for d, rids in ctx.unk.items():
                assert abs(sum(y[r] for r in rids) - ctx.R[d]) < 1e-8, ("witness day total", d)
            if kind == "cell":
                got = sum(y[r] for r in arch.population(cell, B.REF)) / n
            else:
                got = (sum(y[r] for r in arch.population(cb, B.REF)) / nb
                       - sum(y[r] for r in arch.population(ca, B.REF)) / na)
            sharp["%s_%s" % (kind, side)] = {"endpoint": round(want, 8), "witness": round(got, 8),
                                             "attained": abs(got - want) < 1e-8}
    cert["G3_sharpness"] = sharp
    assert all(v["attained"] for v in sharp.values()), sharp

    # G3b: procedures the author expected to be wrong and that measurement showed to be equivalent.
    eq = {}
    for name, spec in B.EQUIVALENT.items():
        got = B.answer(arch, queries, spec)
        ref_ = B.answer(arch, queries, B.REF)
        d = [qid for k in ("cells", "contrasts") for qid in ref_[k]
             if max(abs(ref_[k][qid]["lo"] - got[k][qid]["lo"]),
                    abs(ref_[k][qid]["hi"] - got[k][qid]["hi"])) > 1e-9]
        d += [qid for qid in ref_["decisions"] if ref_["decisions"][qid] != got["decisions"][qid]]
        eq[name] = {"disagreements": d}
        assert not d, ("a procedure recorded as equivalent now disagrees", name, d)
    cert["G3b_equivalences_hold"] = eq

    # G3c: readings too close to grade apart.  Each must stay inside half the tolerance on every endpoint
    # and must not move a single decision, so that an agent taking the other reading still scores 1.
    wt = {}
    for name, spec in B.WITHIN_TOL.items():
        got = B.answer(arch, queries, spec)
        ref_ = B.answer(arch, queries, B.REF)
        worst = max(max(abs(ref_[k][q]["lo"] - got[k][q]["lo"]), abs(ref_[k][q]["hi"] - got[k][q]["hi"]))
                    for k in ("cells", "contrasts") for q in ref_[k])
        moved = [q for q in ref_["decisions"] if ref_["decisions"][q] != got["decisions"][q]]
        wt[name] = {"worst_endpoint_gap": round(worst, 8), "headroom_vs_tol": round(TOL / worst, 1),
                    "decisions_moved": moved}
        assert worst < 0.5 * TOL and not moved, (name, wt[name])
    cert["G3c_within_tolerance_readings"] = wt

    ours = B.answer(arch, queries, B.REF)
    theirs = T.solve(arch.app, queries)
    diffs = []
    for k in ("cells", "contrasts"):
        for qid in ours[k]:
            for e in ("lo", "hi"):
                if abs(ours[k][qid][e] - theirs[k][qid][e]) > 1e-6:
                    diffs.append((qid, e, ours[k][qid][e], theirs[k][qid][e]))
    for qid in ours["decisions"]:
        if ours["decisions"][qid] != theirs["decisions"][qid]:
            diffs.append((qid, "decision", ours["decisions"][qid], theirs["decisions"][qid]))
    cert["G4_independent_agreement"] = {"disagreements": diffs, "compared_at": 1e-6,
                                        "n_items": sum(len(v) for v in queries.values())}
    assert not diffs, diffs

    labels = sorted(set(ours["decisions"].values()))
    cc = B.answer(arch, queries, B.CANDIDATES["complete_case_point_estimate"])
    flipped = [q for q in ours["decisions"] if ours["decisions"][q] != cc["decisions"][q]]
    cert["G6_coverage"] = {"decision_labels": labels, "flipped_by_point_estimate": flipped}
    assert len(labels) == 3 and len(flipped) >= 2

    # G8: the screen the survivorship deduction rests on must be *confirmable and refutable from the
    # published files alone*, or the deduction is a private author convention and the task is unfair.
    #   (a) not one surviving row is under the floor in force when it was written - 1.9k confirmations;
    #   (b) `dropped_at` equals `finished_at` for every quality removal, which is what pins the screen to
    #       the finishing instant rather than to some later sweep;
    #   (c) recovered values for outage runs sit *below* their finishing floor, which refutes the
    #       over-general reading that every unrecorded run cleared it - the boundary of the rule is
    #       visible, not just the rule;
    #   (d) the deduction actually moves the shipped key, by a margin far above the tolerance.
    viol = [rid for rid in arch.res
            if arch.res[rid] < arch._pol(B.iso(arch.finished_at[rid]), B.REF) - 1e-12]
    mism = [rid for rid, e in arch.drop.items()
            if e["reason"] == "below_retention_floor" and e["dropped_at"] != arch.finished_at[rid]]
    lost = sorted(rid for rid, e in arch.drop.items() if e["reason"] == "shard_file_lost")
    outage = [rid for rid in sorted(arch.launched) if rid not in arch.res and rid not in arch.drop]
    refute = [rid for rid in outage
              if rid in arch.rec and arch.rec[rid] / 100.0 < arch._pol(B.iso(arch.finished_at[rid]),
                                                                       B.REF) - 1e-12]
    naive = B.answer(arch, queries, B.CANDIDATES["surviving_row_evidence_ignored"])
    moved = max([max(abs(ours[k][q]["lo"] - naive[k][q]["lo"]), abs(ours[k][q]["hi"] - naive[k][q]["hi"]))
                 for k in ("cells", "contrasts") for q in ours[k]] or [0.0])
    cert["G8_mechanism_is_inferable"] = {
        "rows_confirming_the_screen": len(arch.res), "rows_violating_it": viol[:4],
        "quality_removals_timed_at_the_finish": len([1 for e in arch.drop.values()
                                                     if e["reason"] == "below_retention_floor"]),
        "quality_removals_mistimed": mism[:4],
        "shard_lost_runs": len(lost), "shard_lost_cells": len({arch.cell_of[r] for r in lost}),
        "shard_lost_runs_with_a_recovered_value": [r for r in lost if r in arch.rec],
        "outage_values_below_their_floor": len(refute),
        "key_movement_if_the_deduction_is_skipped": round(moved, 6)}
    assert not viol and not mism and len(refute) >= 3 and len(lost) >= 10, cert["G8_mechanism_is_inferable"]
    assert not [r for r in lost if r in arch.rec] and moved > 10 * TOL, \
        cert["G8_mechanism_is_inferable"]
    return cert, ours


def scan(arch, queries):
    """Final decoy scan on the *selected* items - reported verbatim in the certificate."""
    ref = B.answer(arch, queries, B.REF)
    rows = {}
    for name, spec in B.CANDIDATES.items():
        got = B.answer(arch, queries, spec)
        bad, worst = [], 0.0
        for k in ("cells", "contrasts"):
            for qid in ref[k]:
                d = max(abs(ref[k][qid]["lo"] - got[k][qid]["lo"]),
                        abs(ref[k][qid]["hi"] - got[k][qid]["hi"]))
                if d > TOL:
                    bad.append(qid)
                    worst = max(worst, d)
        for qid in ref["decisions"]:
            if ref["decisions"][qid] != got["decisions"][qid]:
                bad.append(qid)
        rows[name] = {"n_fail": len(bad), "items": bad[:6], "worst_endpoint_error": round(worst, 5)}
    return rows


INSTR = """You are the on-call analyst for a model-evaluation fleet.  A six-week sweep finished; the
archive in `/app` is what survived of it.

Every planned run appears in `plan.csv` with its four configuration knobs (`suite`, `curriculum`,
`retrieval`, `decoder`).  `launch_log.csv` says which of them actually started.  Each launched run
produced a single number, its `pass_rate` on the suite it was assigned - but the archive lost some of
those numbers, in three different ways.  `incidents.md` is the on-call write-up of what happened to this
sweep and `metric_card.md` says what the number means; read both, and read the logs, because how much
each surviving record still constrains a missing value is for you to work out.  Nothing in `/app` is a
decoy: every file is exactly what it claims to be, and no file is wrong.

Your job is *not* to guess the missing numbers.  It is to report, for a handful of configurations,
exactly what the surviving records still pin down about them - no more and no less.

## The quantity

Fix a configuration `c` (one value for each of the four knobs).  `L(c)` is the set of run ids that appear
in `launch_log.csv` *and* whose `plan.csv` row has exactly those four knob values; count each run id once,
however many rows mention it.  `N(c) = |L(c)|`.

`m(c)` is the plain average of the true `pass_rate` of the runs in `L(c)`, over all `N(c)` of them,
including the ones whose value the archive lost.  Every launched run has a true `pass_rate`; the
archive's failure to record it does not remove the run from the average.  `m(c)` is a property of these
particular runs, not of some larger population they were drawn from - there is nothing to extrapolate to
and nothing to resample.

## What to report

Call an assignment of a value to every launched run whose `pass_rate` the archive did not keep
*consistent* if it contradicts nothing in `/app`.  For each item in `/app/queries.json`, report the
**sharp** interval: the smallest closed interval containing the quantity's value under every consistent
assignment.  An interval that is correct but wider than necessary is wrong, and so is a single number
reported as a degenerate interval when the quantity is not pinned down.

  * `cells`: the interval for `m(c)`.
  * `contrasts`: the interval for `m(to) - m(from)`.
  * `decisions`: is `m(to) - m(from)` greater than `margin`?  Answer `"yes"` if that holds under every
    consistent assignment, `"no"` if it holds under none of them, and `"cannot_tell"` otherwise.

Write `/app/answers.json`:

```json
{
  "cells":     {"b01": {"lo": 0.4131, "hi": 0.4131}},
  "contrasts": {"k01": {"lo": -0.0182, "hi": 0.0413}},
  "decisions": {"d01": "cannot_tell"}
}
```

Every id in `queries.json` must appear.  Endpoints are graded to an absolute tolerance of 5e-4, so give at
least four decimals; the answer is exact arithmetic, not an estimate.  Scoring is all-or-nothing over the
%d items.  Do not edit anything under `/app` except `answers.json`.
"""


def main():
    w = W.build(0)
    app = os.path.join(OUT, "environment", "app")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(app)
    emit(app, w)
    arch = B.Arch(app)
    cells = launched_cells(arch)
    items = build_items(arch, cells)
    queries, need, gaps, killable = select(arch, items)
    short = {n: k for n, k in need.items() if k > 0}
    cert, ref = certify(arch, w, queries)
    sc = scan(arch, queries)
    cert["G5_selection"] = {"underkilled": short, "items_able_to_kill_each_decoy": killable,
                            "decoys": sc}
    cert["G7_tolerance"] = {"tol": TOL, "min_decoy_endpoint_error":
                            round(min([v["worst_endpoint_error"] for v in sc.values()
                                       if v["worst_endpoint_error"] > 0] or [0]), 5)}
    ok = (not short) and all(v["n_fail"] >= MIN_FAIL for v in sc.values())

    json.dump(queries, open(os.path.join(app, "queries.json"), "w"), indent=1)
    n_items = sum(len(v) for v in queries.values())
    key = {"tol": TOL, "cells": ref["cells"], "contrasts": ref["contrasts"],
           "decisions": ref["decisions"]}

    H.w(os.path.join(OUT, "instruction.md"),
        INSTR % n_items + H.SUFFIX_T.format(t=5400))
    H.w(os.path.join(OUT, "task.toml"),
        H.task_toml(artifacts=["answers.json"], family="b-bounds-a",
                    tags=["partial-identification", "missing-data", "provenance", "ai4ai-eval"],
                    agent_timeout=5400, expert_hours=2.5, verifier_timeout=300))
    os.makedirs(os.path.join(OUT, "tests"))
    shutil.copy(os.path.join(HERE, "verify_b.py"), os.path.join(OUT, "tests", "verify_b.py"))
    json.dump(key, open(os.path.join(OUT, "tests", "key.json"), "w"), indent=1)
    H.w(os.path.join(OUT, "tests", "test.sh"),
        "#!/bin/bash\nset -eu\npython3 /tests/verify_b.py\n", mode=0o755)
    H.w(os.path.join(OUT, "tests", "Dockerfile"),
        "FROM python:3.11-slim\nCOPY verify_b.py key.json test.sh /tests/\n")
    os.makedirs(os.path.join(OUT, "solution"))
    for f in ("bounds_b.py", "truth_b.py", "ref_solve_b.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(OUT, "solution", f))
    H.w(os.path.join(OUT, "solution", "solve.sh"),
        "#!/bin/bash\nset -eu\npython3 /solution/ref_solve_b.py /app\n", mode=0o755)
    os.makedirs(os.path.join(OUT, "authoring"))
    json.dump({"accepted": ok, "n_items": n_items, "gates": cert},
              open(os.path.join(OUT, "authoring", "certificate.json"), "w"), indent=1)

    print("items", {k: len(v) for k, v in queries.items()}, "total", n_items)
    print("decision labels", cert["G6_coverage"]["decision_labels"],
          "flipped by point estimate", len(cert["G6_coverage"]["flipped_by_point_estimate"]))
    for n, v in sorted(sc.items(), key=lambda x: x[1]["n_fail"]):
        print("  %-48s n_fail=%-3d worst=%.4f %s" % (n, v["n_fail"], v["worst_endpoint_error"],
                                                     v["items"][:4]))
    print("underkilled:", short)
    print("accepted:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
