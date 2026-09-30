"""Build, certify and export `d-design-open`: the arm where the evidence stops short of the question.

`d-design-sealed` took the specification away from the input and made the agent infer, from a log of past
recovery requests, what a request comes back with.  Fable 5.1 read the log, cross-tabulated the outcome
column against the reason the archive lost the run, announced the three-way rule in its third message, and
scored 31/31; GPT-6 Astra inferred the same rule correctly and lost five items to over-sharp intervals
rather than to the semantics.  That is the same lesson `b-bounds-sealed` taught at 34/34: an undocumented
mechanism whose signature is a clean partition of an observed column is a contingency table, and frontier
models build contingency tables reliably.

The only arm that has ever broken a model - `s-censored-a` - is the one where the evidence *under*
determines the answer and the agent has to notice that and fall back on the worst case.  This arm is that
move, applied to the sealed world with the smallest possible edit:

  * the truth about the service is **exactly the sealed arm's**, so the shipped key is the same function
    of the same world.  Nothing about the estimand, the archive or the arithmetic changes;
  * the recovery log no longer mentions a single run the sink never delivered a row for.  The log pins
    what an intact bundle and a redacted bundle come back with; about the runs the outage took, it is
    silent;
  * `recovery_service.md` closes the outcome space (a request comes back in exactly one of three ways) and
    pins the granularity (what you get depends only on what became of that run's record - not the worker,
    not the date, not the request), and then says, truthfully, that the log is a sweep the on-call team
    happened to make and nothing arranged for it to cover every state.

So one cell of the table is empty, three fillings of it are consistent with `/app`, and the instruction's
definition - the largest width over every outcome that *contradicts nothing in `/app`* - forces the worst
of the three.  The agent that completes the table by elimination ("not redacted, not intact-with-a-value,
so it must be there") gets a strictly narrower guarantee and is wrong.  And the trap is not neutral: the
runs in the open cell are the widest boxes in the sweep, so under the elimination reading they are the
best buys in every planning item, and under the truth they are worth nothing at all.

Because the truth and the key are the sealed arm's, the two arms are a controlled pair: same world, same
estimand, same reference solver, differing only in whether the evidence covers the case the items turn on.
Any score difference between them is attributable to the underdetermination step and to nothing else.

Gates.  G2-G8 and G10-G12 are the sealed arm's, with `RIVAL` pointed at the elimination reading.  G9 is
inverted and G13 is new:

  G9'  *the archive leaves exactly the intended slack* - the readings consistent with the log are exactly
       the three completions of the open cell, so the items are underdetermined in one specific way and
       in no other;
  G13  *the open cell is real, worst-case-forced and load-bearing* - no run of the open class appears in
       the log, the three survivors agree everywhere except on that class, the shipped key is their
       pointwise maximum (so the published answer is the guarantee, not a guess), and shipped items name
       open-class runs in every graded section.
"""
import itertools
import json
import os
import random
import shutil
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
import design_d as D                                                          # noqa: E402
import gen_d as G                                                             # noqa: E402
import gen_ds as GS                                                           # noqa: E402
import harbor as H                                                            # noqa: E402
import world_d as WD                                                          # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/d-design-open"
OPEN_CLASS = "never_written"
ZERO = Fraction(0)

# The three ways the open cell can be filled.  The truth - the sealed arm's reading - is the one that says
# the sink that never delivered the row never finalised the bundle either; the other two are the readings
# an agent reaches by completing the table from the three cells the log does show.  All three are
# consistent with every row of the log, which is the point of the arm.
COMPLETIONS = {
    # nothing comes back: the sealed arm's truth, and the worst of the three.
    GS.TRUTH_RULE: GS.RECOV_RULES[GS.TRUTH_RULE],
    # the bundle is intact and the value comes back - cold storage as the backup for what the archive lost.
    "the_outage_stopped_at_the_archive_and_the_bundle_is_intact":
        lambda c, r: GS._screen(c, r),
    # the bundle is there but redacted to the screen's audit line, like the runs the screen removed.
    "the_outage_left_only_an_audit_line_for_those_runs":
        lambda c, r: (c.band(r) if c.why[r] in (OPEN_CLASS, "below_retention_floor") else ZERO),
}
GS.RECOV_RULES.update(COMPLETIONS)
# The reading G11 punishes: the seductive completion, under which the widest boxes in the sweep - the runs
# the archive has nothing at all for - become the most valuable recoveries instead of the worthless ones.
GS.RIVAL = "the_outage_stopped_at_the_archive_and_the_bundle_is_intact"
SPARE = tuple(n for n in COMPLETIONS if n != GS.TRUTH_RULE)
TOL = GS.TOL

RECOVERY_OPEN = """# Cold-storage recovery requests

The evaluation fleet writes a per-run **trace bundle** to cold storage.  A **recovery request** names a
set of launched runs and is answered out of those bundles.  Requests are slow and are charged to the
team's quarterly budget, which is why the planning questions in `queries.json` ask what a request would be
worth *before* it is issued.

A request is issued blind: you commit to the whole set and pay for it, and you do not get to see the first
answer before choosing the second.

## What a request can come back with

For each run in the set, a request comes back in exactly one of these three ways, and in no other:

  1. **the bundle is intact** - you get that run's `pass_rate`;
  2. **the bundle was redacted** - the trace is gone, and what survives is the quality screen's audit
     line, which kept the score only to the resolution its suite is bucketed at.  You get a range of that
     width containing the value.  The service does not promise the value sits anywhere in particular
     inside the range, and where the range falls is not something this archive determines;
  3. **there is no bundle** - nothing comes back for that run.

What comes back is never wrong and never contradicts this archive.

## What it depends on

Cold storage does not decide case by case.  Which of the three you get for a run depends only on **what
became of that run's result record**: whether the archive still has its number and, if not, which of the
incidents in `incidents.md` is the reason it does not.  Two runs the incidents left in the same state come
back the same way.  It does not depend on the worker, on when the run finished, on when you ask, on what
else is in the same request, or on the run's knob settings - except that the *width* of a redacted answer
follows the run's own suite, since the suite is what sets the audit resolution.

Which state comes back which way is not written down anywhere.  Cold storage is a different system from
the archive - written by the evaluation fleet, not by the archiver - and the incidents did not all reach
the two systems the same way.  Nothing here, and nothing anywhere else in `/app`, says which is which.

## The log

`recovery_log.csv` is the record of the recovery requests the on-call team issued during the incident
response: the run, the outcome, and, where anything came back, the range of `pass_rate` that answer left
the run in (`returned_lo` and `returned_hi`, equal to each other when the answer was a single number).

It is what it is - a sweep made under time pressure while some of the incidents were still open, on the
runs somebody happened to need at the time.  It is a sample, not a survey.  Most launched runs were never
requested, none of the runs `queries.json` asks about were, it was not collected to answer your questions,
and nothing arranged for it to cover every state the fleet's runs ended up in.
"""


# The sweep's incident notes say of the sink outage that "the number never left the worker".  In the
# sealed arm that is harmless colour - the log settles what a recovery returns, so nothing hangs on it.
# Here it would be a second, unintended route to the answer: a reader could argue the open cell is not
# open at all because nothing could have reached cold storage either.  It is also not quite true, since
# the same notes go on to describe values scraped out of those workers' stdout tails.  The sentence is
# therefore narrowed to the claim the archive actually supports, so that the cell is open on the evidence
# and the guarantee has to be taken as a worst case rather than argued from physics.
INCIDENT_FIX = ("never got a row into the archive at all - the number never\nleft the worker.",
                "never got a row into the archive at all - the writer never\ndelivered them.")


def emit(app, w, log_rows):
    G.emit(app, w)
    p = os.path.join(app, "incidents.md")
    txt = open(p).read()
    assert txt.count(INCIDENT_FIX[0]) == 1, "the incident note this arm narrows has moved"
    H.w(p, txt.replace(*INCIDENT_FIX))
    H.w(os.path.join(app, "recovery_service.md"), RECOVERY_OPEN)
    H.csv_rows(os.path.join(app, "recovery_log.csv"),
               ["requested_at", "run_id", "outcome", "returned_lo", "returned_hi"], log_rows)


def plan_items_open(b, live):
    """The sealed arm's plan items, plus items the elimination reading actually gets wrong.

    The sealed arm padded every candidate list with the widest boxes it could find, on the theory that a
    reading which thinks they are recoverable will spend the budget on them.  Measured against this arm's
    rival that turns out to be false: a wide box on a day whose surplus is already the binding constraint
    is worth nothing *even when it comes back exact*, so making those runs recoverable does not move the
    optimum and the rival answers the plan items correctly by accident.

    So the bait here is selected by the property that matters rather than by width: a run is worth padding
    a list with only if recovering it would genuinely shrink the guarantee **under the rival's reading**.
    Then a target exists between what the rival thinks it can buy and what the archive can actually
    guarantee, and the two readings answer with different sets or different sizes.  This is the same
    discipline the rest of the factory uses - never assume a decoy is attractive, check that it is.
    """
    dd = b.ref
    riv = b.alt["recovery__" + GS.RIVAL]
    out = list(GS.plan_items(b, live))
    for sa, sc, days in live:
        rel = sorted({r for d in days for r in dd.T.free[d]})
        if not 4 <= len(rel) <= 240:
            continue
        w0, w0r = dd.width(sa, sc, set()), riv.width(sa, sc, set())
        bait = sorted((r for r in rel if GS.dead(dd, r) and riv.width(sa, sc, {r}) < w0r),
                      key=lambda r: (riv.width(sa, sc, {r}), r))[:5]
        good = sorted((r for r in rel if not GS.dead(dd, r) and dd.width(sa, sc, {r}) < w0),
                      key=lambda r: (dd.width(sa, sc, {r}), r))[:7]
        if not bait or not good:
            continue
        far = sorted(r for d in sorted(dd.T.free) if d not in set(days) for r in dd.T.free[d])
        cands = sorted(set(good + bait + far[::max(1, len(far) // 4)][:3]))
        if not 9 <= len(cands) <= 17:
            continue
        W, R = {}, {}
        for k in range(0, GS.MAX_K + 1):
            for sub in itertools.combinations(cands, k):
                W[sub], R[sub] = dd.width(sa, sc, set(sub)), riv.width(sa, sc, set(sub))
        seq = G.greedy_sequence(dd, sa, sc, cands, GS.MAX_K)
        gw = [x for _, x in seq]
        for k in (1, 2, 3):
            for tgt in sorted({v for s, v in W.items() if len(s) == k}):
                kmin = min(len(s) for s, v in W.items() if v <= tgt)
                if kmin != k:
                    continue
                tgt = G.publishable_target(W, kmin, tgt, GS.TOL, extra=R.values())
                if tgt is None:
                    continue
                fs = sorted(sorted(s) for s, v in W.items() if len(s) == kmin and v <= tgt)
                rk = min(len(s) for s, v in R.items() if v <= tgt)
                rs = sorted(sorted(s) for s, v in R.items() if len(s) == rk and v <= tgt)
                if rk == kmin and rs[0] in fs:
                    continue                              # the rival would be graded correct here
                gk = next((i + 1 for i, x in enumerate(gw) if x <= tgt), None)
                out.append({"id": None, "from": sa, "to": sc, "candidates": cands, "target": tgt,
                            "_k": kmin, "_sets": fs, "_greedy": gk,
                            "_greedy_seq": [r for r, _ in seq],
                            "_bait": [r for r in cands if GS.dead(dd, r)]})
                break
            else:
                continue
            break
    return out



def open_gate(b, ctx, log_rows, queries, cert):
    """G13 - the open cell is real, the key is its worst case, and the items turn on it."""
    dd = b.ref
    seen = GS.read_back(ctx, log_rows)
    openrun = sorted(r for r in dd.unknown if ctx.why[r] == OPEN_CLASS)
    assert openrun, "no run of the open class survived into the unrecorded set"
    surv = cert["G9_identifiability"]["readings_consistent_with_the_log"]
    others = sorted(dd.unknown - set(openrun))

    agree = all(GS.observed(ctx, r, n) == GS.observed(ctx, r, GS.TRUTH_RULE)
                for n in surv for r in others)
    dominates = all(GS.observed(ctx, r, n) <= GS.observed(ctx, r, GS.TRUTH_RULE)
                    for n in surv for r in openrun)
    strict = sorted({n for n in surv if any(GS.observed(ctx, r, n)
                                            < GS.observed(ctx, r, GS.TRUTH_RULE) for r in openrun)})
    named = {"recovery": sorted({q["run_id"] for q in queries["recovery"] if q["run_id"] in openrun}),
             "values": sorted({q["run_id"] for q in queries["values"] if q["run_id"] in openrun}),
             "widths": sorted({q["id"] for q in queries["widths"] if set(q["recover"]) & set(openrun)}),
             "plans": sorted({q["id"] for q in queries["plans"]
                              if set(q["candidates"]) & set(openrun)})}
    cert["G13_open_cell"] = {
        "open_class": OPEN_CLASS, "unrecorded_runs_in_it": len(openrun),
        "rows_of_it_in_the_log": sorted(r for r in seen if ctx.why[r] == OPEN_CLASS),
        "states_the_log_exercises": sorted({ctx.why[r] for r in seen}),
        "survivors_agree_on_every_other_run": agree,
        "key_is_the_pointwise_worst_case": dominates,
        "survivors_strictly_below_it_on_the_open_cell": strict,
        "widest_box_in_the_sweep_is_open_class":
            max(dd.box[r][1] - dd.box[r][0] for r in dd.unknown)
            == max(dd.box[r][1] - dd.box[r][0] for r in openrun),
        "shipped_items_that_touch_it": named}
    assert not cert["G13_open_cell"]["rows_of_it_in_the_log"]
    assert agree and dominates, (agree, dominates)
    assert sorted(strict) == sorted(SPARE), strict
    assert all(named[k] for k in named), named
    return cert


def main():
    w = WD.build(0)
    app = os.path.join(OUT, "environment", "app")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(app)
    G.emit(app, w)

    base = D.Design(app)
    ctx = GS.Ctx(app, base)
    runs = sorted(base.unknown)
    b = GS.BenchS(app, ctx, runs)
    openrun = [r for r in runs if ctx.why[r] == OPEN_CLASS]
    print("pool: readings=%d unrecorded=%d open_class=%d" % (len(b.alt) + 1, len(runs), len(openrun)))

    items, live = GS.build_items(b)
    plans = plan_items_open(b, live)
    assert plans, "no plan item survived the search"
    # The open cell has to be graded in every section, or the arm measures nothing it was built for.
    # `select` already seeds one recovery and one value item on an unrecoverable run; these make it two.
    seeds = [("recovery", lambda q: ctx.why[q["run_id"]] == OPEN_CLASS),
             ("values", lambda q: ctx.why[q["run_id"]] == OPEN_CLASS),
             ("widths", lambda q: any(ctx.why[r] == OPEN_CLASS for r in q["recover"])),
             ("plans", lambda q: q["_greedy"] is None or q["_greedy"] > q["_k"])]
    queries, need, killable, ans = GS.select(b, items, plans, seeds)
    pub, meta = G.publish(queries)
    plans_meta = {q["id"]: dict(meta[q["id"]]) for q in queries["plans"]}
    for q in pub["plans"]:
        k, sets = b.ref.plan(q["from"], q["to"], q["candidates"], Fraction(str(q["target"])))
        assert k == plans_meta[q["id"]]["_k"], (q["id"], k, plans_meta[q["id"]]["_k"])
        assert sorted(sets) == sorted(plans_meta[q["id"]]["_sets"]), q["id"]

    hidden = {k: Fraction(str(v)) for k, v in w["hidden"].items()}
    avoid = {q["run_id"] for q in pub["recovery"] + pub["values"]} | set(openrun)
    for q in pub["widths"]:
        avoid |= set(q["recover"])
    for q in pub["plans"]:
        avoid |= set(q["candidates"])
    log_rows = GS.build_log(ctx, b.ref, hidden, random.Random(5), avoid, spare=SPARE)
    emit(app, w, log_rows)

    expect = sorted(COMPLETIONS)
    cert = GS.certify(b, ctx, hidden, log_rows, pub, plans_meta, expect=expect)
    cert.update(GS.coverage(b, ctx, log_rows, pub, plans_meta, need, killable, ans))
    open_gate(b, ctx, log_rows, pub, cert)
    n_items = sum(len(v) for v in pub.values())
    ok = (not any(v > 0 for v in need.values())) and cert["G3_attainment"]["all_attained"]

    json.dump(pub, open(os.path.join(app, "queries.json"), "w"), indent=1)
    ref = ans["reference"]
    key = {"tol": TOL, "contrasts": ref["contrasts"], "recovery": ref["recovery"],
           "values": ref["values"], "widths": ref["widths"],
           "plans": {q["id"]: {"k": plans_meta[q["id"]]["_k"],
                               "sets": [sorted(s) for s in plans_meta[q["id"]]["_sets"]]}
                     for q in pub["plans"]}}

    H.w(os.path.join(OUT, "instruction.md"),
        GS.INSTRUCTION % {"tol": TOL, "n": n_items} + H.SUFFIX_T.format(t=5400))
    H.w(os.path.join(OUT, "task.toml"),
        H.task_toml(artifacts=["answers.json"], family="d-design-open",
                    tags=["value-of-information", "partial-identification", "experiment-design",
                          "underdetermined-evidence", "worst-case-reasoning", "ai4ai-eval"],
                    agent_timeout=5400, expert_hours=6.0, verifier_timeout=300))
    os.makedirs(os.path.join(OUT, "tests"))
    H.w(os.path.join(OUT, "tests", "verify_ds.py"), GS.VERIFY)
    shutil.copy(os.path.join(ROOT, "core", "harbor.py"), os.path.join(OUT, "tests", "harbor.py"))
    json.dump(key, open(os.path.join(OUT, "tests", "key.json"), "w"), indent=1)
    H.w(os.path.join(OUT, "tests", "test.sh"), "#!/bin/bash\nset -eu\npython3 /tests/verify_ds.py\n",
        mode=0o755)
    H.w(os.path.join(OUT, "tests", "Dockerfile"),
        "FROM python:3.11-slim\nCOPY verify_ds.py harbor.py key.json test.sh /tests/\n")
    os.makedirs(os.path.join(OUT, "solution"))
    H.w(os.path.join(OUT, "solution", "ref_solve_ds.py"), GS.SOLVE)
    json.dump({r: str(b.ref.grain[r]) for r in sorted(b.ref.grain)},
              open(os.path.join(OUT, "solution", "grain.json"), "w"), indent=1)
    for f in ("design_d.py", "truth_b.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(OUT, "solution", f))
    H.w(os.path.join(OUT, "solution", "solve.sh"),
        "#!/bin/bash\nset -eu\npython3 /solution/ref_solve_ds.py /app\n", mode=0o755)
    os.makedirs(os.path.join(OUT, "authoring"))
    json.dump({"accepted": ok, "n_items": n_items, "gates": cert},
              open(os.path.join(OUT, "authoring", "certificate.json"), "w"), indent=1)

    print("items", {k: len(v) for k, v in pub.items()}, "total", n_items)
    print("G2", cert["G2_validity"])
    print("G3", {k: v for k, v in cert["G3_attainment"].items() if k != "sample"})
    print("G4", cert["G4_soundness_sampling"])
    print("G9'", cert["G9_identifiability"])
    print("G10", cert["G10_no_lookup"])
    print("G11", cert["G11_inversion_is_load_bearing"]["items_it_gets_wrong"])
    print("G12", cert["G12_resolutions"])
    print("G13", {k: v for k, v in cert["G13_open_cell"].items() if k != "shipped_items_that_touch_it"})
    print("G13 items", cert["G13_open_cell"]["shipped_items_that_touch_it"])
    print("G6", cert["G6_coverage"])
    print("G7 bait:", cert["G7_search"]["with_unrecoverable_bait"],
          "greedy-beats:", cert["G7_search"]["with_greedy_strictly_worse"])
    print("G8 regimes:", sorted(cert["G8_regimes"]))
    for n, v in sorted(cert["G5_decoys"]["items_each_reading_gets_wrong"].items(), key=lambda x: x[1]):
        print("  %-56s wrong_on=%-3d could_lose_in_pool=%d" % (n, v, killable[n]))
    print("underkilled:", cert["G5_decoys"]["still_needed"])
    print("accepted:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
