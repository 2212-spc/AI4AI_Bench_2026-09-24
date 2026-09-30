"""Build, certify and export `d-design-a`: which measurement is worth taking.

The archive is family B's, extended by `world_d.py` with a late campaign whose days are shaped so that the
easy rule learned from ordinary days is wrong.  What is graded is new: not what the records pin down, but
how much a *future* recovery would narrow it.

Gates, in the order they run:

  G1  emit the archive and re-derive every published quantity from the emitted files only;
  G2  *validity* - every hidden value inside the support its own archive implies, every day's hidden
      values adding up to the published digest, and the true contrast inside every published interval;
  G3  *attainment* - for every published width, build the recovery outcome that attains it, hand those
      values to `truth_b.py` as if they had really been recovered, and check the interval it derives has
      exactly that width.  This is a third implementation path and it is the one that can catch a wrong
      *definition*, not just wrong arithmetic;
  G4  *soundness by sampling* - a few hundred random feasible recovery outcomes per item, none of which
      may produce a width above the published guarantee;
  G5  *item selection* - every competing procedure in `design_d.RULES` must be wrong on at least MIN_FAIL
      shipped items, and items are picked greedily by how much scarce decoy weight they kill;
  G6  *coverage* - the value items must contain both labels; at least two `yes` items must be runs in
      neither cell, at least two `no` items must be the widest box on their day, and at least one plan's
      unique optimum must contain no member of either cell;
  G7  *search* - at least one plan item on which the greedy ranking of candidates needs strictly more
      recoveries than the optimum, so the item cannot be passed by ranking;
  G8  *regimes present* - the three day shapes the family depends on (slack, absorbing, saturated) must
      all be present among the shipped items, and each must be checkable by the agent from the published
      files alone.
"""
import itertools
import json
import math
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
import harbor as H                                                            # noqa: E402
import truth_b as T                                                           # noqa: E402
import world_b as WB                                                          # noqa: E402
import world_d as WD                                                          # noqa: E402

TOL = 5e-4
MIN_FAIL = 2
N_CONTRAST, N_VALUE, N_WIDTH, N_PLAN = 4, 12, 8, 3
MAX_K = 3
N_SAMPLE = 60
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/d-design-a"
KNOBS = D.KNOBS
ZERO = Fraction(0)

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

**INC-4530 - sink outage on the late campaign (worker %(late_worker)s).**  After the main sweep closed,
three short follow-up campaigns were run on %(late_worker)s between %(late_from)s and %(late_to)s to try
two curriculum settings that the sweep had not covered.  %(late_worker)s had been rebuilt during the
storage migration and came back without its results-writer credentials, which nobody noticed until the
campaigns were over: none of its rows reached the archive.  As with INC-4471 the failure is in the
writer, so which runs lost their rows has nothing to do with what they scored, and no scrape was
attempted - the worker was reimaged.  Runs from other workers that finished on the same days were written
and screened normally.

**Scheduler accounting.**  The scheduler keeps its own ledger, on a different host from the results
writer and untouched by any of these incidents: `runtime.csv` records when each launched run finished, and
`daily_digest.csv` records, for each UTC day, how many runs finished that day and what their `pass_rate`
values added up to.  The digest is written at 23:59 from the scores the scheduler received directly from
each worker, so its totals still cover runs whose rows the archive screened out, lost, or never received.
`daily_digest.csv` is exact to the four decimals it prints.

**Not affected.**  `plan.csv`, `launch_log.csv`, `retention_log.csv`, `runtime.csv` and
`daily_digest.csv` are complete for the whole sweep, including the late campaigns.
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

RECOVERY = """# Cold-storage recovery requests

The evaluation harness writes a per-run trace bundle to cold storage before it writes anything to the
archive, and the bundles are still there.  A **recovery request** names one launched run and returns its
exact `pass_rate`, the same number the archive should have kept.  Recoveries are slow and are charged to
the team's quarterly budget, which is why the planning questions in `queries.json` ask how much a given
request would be worth *before* it is issued.

Two properties of the service matter for those questions:

  * a request is issued blind.  You commit to a set of runs and pay for it; you do not get to look at the
    first value before choosing the second.
  * a recovery returns the value that run actually had.  It is not an estimate and it cannot come back
    with a number that contradicts anything already in this archive - but which number it will be is not
    something the archive determines, except where the archive already pins it.
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
               "versions": [{"version": p["version"], "from": WB.ts(p["from"]), "to": WB.ts(p["to"]),
                             "floor": p["floor"]} for p in WB.POLICY]},
              open(os.path.join(app, "retention_policy.json"), "w"), indent=1)
    json.dump(WB.SUITES, open(os.path.join(app, "suites.json"), "w"), indent=1)
    H.csv_rows(os.path.join(app, "runtime.csv"), ["run_id", "finished_at"], w["runtime"])
    H.csv_rows(os.path.join(app, "daily_digest.csv"), ["date", "n_runs", "sum_pass_rate"], w["digest"])
    H.w(os.path.join(app, "incidents.md"),
        INCIDENTS % {"worker": w["shard"]["worker"], "date": w["shard"]["date"],
                     "late_worker": w["late"]["worker"], "late_from": w["late"]["from"],
                     "late_to": w["late"]["to"]})
    H.w(os.path.join(app, "metric_card.md"), METRIC)
    H.w(os.path.join(app, "recovery_service.md"), RECOVERY)


# ---------------------------------------------------------------------------------------------------
# items


def as_spec(c):
    return dict(zip(KNOBS, c))


def cells_with_runs(dd):
    out = {}
    for rid, r in dd.T.plan.items():
        if rid in dd.T.started:
            c = tuple(r[k] for k in KNOBS)
            out[c] = out.get(c, 0) + 1
    return out


def pairs_of(dd):
    cells = sorted(cells_with_runs(dd))
    out = []
    for a in cells:
        for b in cells:
            if a < b and a[0] == b[0] and sum(1 for i in range(4) if a[i] != b[i]) == 1:
                out.append((a, b))
    return out


def fmt(x):
    return float(round(Fraction(x), 12) if isinstance(x, Fraction) else x)


class Bench(object):
    """One archive, evaluated under every rule in `design_d.RULES` as well as the reference."""

    def __init__(self, app):
        self.app = app
        self.ref = D.Design(app)
        self.alt = {n: D.Design(app, r) for n, r in D.RULES.items()}

    def each(self):
        yield "reference", self.ref
        for n in sorted(self.alt):
            yield n, self.alt[n]

    def answer(self, dd, queries):
        out = {"contrasts": {}, "values": {}, "widths": {}, "plans": {}}
        for q in queries["contrasts"]:
            lo, hi = dd.T.diff(q["from"], q["to"])
            out["contrasts"][q["id"]] = {"lo": fmt(lo), "hi": fmt(hi)}
        for q in queries["values"]:
            out["values"][q["id"]] = "yes" if dd.voi(q["from"], q["to"], q["run_id"]) else "no"
        for q in queries["widths"]:
            out["widths"][q["id"]] = fmt(dd.width(q["from"], q["to"], set(q["recover"])))
        for q in queries["plans"]:
            k, sets = dd.plan(q["from"], q["to"], q["candidates"], Fraction(str(q["target"])))
            out["plans"][q["id"]] = {"k": k, "runs": sets[0] if sets else []}
        return out


def build_items(b):
    """Every item worth considering, before selection.

    The full grid is far larger than the selector needs - two hundred contrasts, twenty thousand value
    items - and evaluating all of it under seven procedures would cost more than it buys.  The pool is
    therefore cut to the part of the grid where the family's question has any content: every contrast that
    touches a late campaign, plus a handful of ordinary contrasts kept as controls, because an agent that
    over-corrects into "recovering a cell member never helps" has to be caught too.
    """
    dd = b.ref
    items = {"contrasts": [], "values": [], "widths": [], "plans": []}
    late_days = set(WD.CAMPAIGN_DAYS)
    live, control = [], []
    for a, c in pairs_of(dd):
        sa, sc = as_spec(a), as_spec(c)
        days = dd.relevant_days(sa, sc)
        items["contrasts"].append({"id": None, "from": sa, "to": sc})
        if not days:
            continue
        if a[1] in WD.LATE_CURRICULA or c[1] in WD.LATE_CURRICULA:
            live.append((sa, sc, days))
        else:
            n = sum(len(dd.T.free[d]) for d in days)
            control.append(((len(days), n, a, c), (sa, sc, days)))
    control.sort(key=lambda x: (abs(x[0][0] - 3), x[0][1], x[0][2], x[0][3]))
    live += [v for _, v in control[:6]]

    rng = random.Random(11)
    for sa, sc, days in live:
        runs = [r for d in days if d in late_days for r in dd.T.free[d]]
        other = sorted(r for d in days if d not in late_days for r in dd.T.free[d])
        runs += other[:12]
        for rid in runs:
            items["values"].append({"id": None, "from": sa, "to": sc, "run_id": rid})
        items["widths"].append({"id": None, "from": sa, "to": sc, "recover": []})
        for rid in runs:
            items["widths"].append({"id": None, "from": sa, "to": sc, "recover": [rid]})
        for _ in range(12):
            if len(runs) >= 2:
                s = sorted(rng.sample(runs, 2))
                items["widths"].append({"id": None, "from": sa, "to": sc, "recover": s})
    return items, live


def greedy_sequence(dd, a, b, cands, maxk):
    """The order a marginal-gain ranking would recover in, with the width after each step."""
    chosen, seq = [], []
    cur = dd.width(a, b)
    pool = sorted(cands)
    while pool and len(chosen) < maxk:
        scored = []
        for rid in pool:
            w = dd.width(a, b, set(chosen + [rid]))
            scored.append((w, -(dd.box[rid][1] - dd.box[rid][0]), rid))
        scored.sort()
        w, _, rid = scored[0]
        chosen.append(rid)
        pool.remove(rid)
        cur = w
        seq.append((rid, cur))
    return seq


def publishable_target(widths, k, raw, tol, slack=2, extra=()):
    """Move a raw achievable width up to a six-decimal target that no float solver can straddle.

    The first version of this rounded the achievable width *up* to the publishable grid, which put the
    target between 1e-7 and 1e-6 above `w(k)`.  That is arithmetically correct and practically a trap: an
    agent computing widths in floating point is off by more than that routinely, decides k recoveries miss
    the target, and answers k+1.  The item then measures whether the solver used exact rationals, which is
    not the capability family D exists to measure, and the defect is invisible to a reference solver that
    uses `Fraction` throughout - which is exactly why it survived two frontier models answering correctly.

    The repair has to keep the sweep alive, though: which sets are optimal, and how badly a greedy ranking
    does, both depend on *where* in the interval the threshold sits, so collapsing every threshold to the
    interval's midpoint - the first repair I wrote - silently deleted every item on which greedy is worse
    than the optimum, and the search gate caught it.  So the target stays attached to its raw value and is
    only nudged: up to the next grid point above `raw`, and accepted only if the gap to the next *distinct*
    achievable width above it leaves `slack x tol` of room on both sides.  Inside such a gap the answer -
    the count, the optimal sets, and the greedy comparison alike - is constant, so a solver whose arithmetic
    is off by less than the slack still lands on it.

    `slack` is 2 because the bar comes from the audit gate that polices this: a threshold item fails when a
    perturbation of one tolerance moves the graded answer, so two tolerances of room is the first setting
    that clears it with a doubling to spare.  It is not free to ask for more - the gaps between achievable
    widths are what the slack is spent out of, and at `slack=10` the search returns only items on which a
    greedy ranking happens to be optimal, which is to say it deletes the discriminating ones.
    """
    wk = min((w for s, w in widths.items() if len(s) <= k), default=None)
    wk1 = min((w for s, w in widths.items() if len(s) <= k - 1), default=None)
    if wk is None or wk1 is None or not (wk <= raw < wk1):
        return None
    eps = slack * Fraction(str(tol))
    other = set(widths.values()) | set(extra)          # `extra`: widths under a rival reading, which the
    hi = min([w for w in other if w > raw] + [wk1])    # threshold must not straddle either
    t = Fraction(math.ceil((raw + eps) * 1000000), 1000000)          # first grid point clear of `raw`
    return t if t + eps <= hi else None


def plan_items(b, live):
    """Search for plan items worth shipping.

    The candidate list is part of the question - in the world it is the set of runs whose trace bundles
    are still in cold storage - so the generator is free to choose it, and it chooses it to make the item
    discriminating.  Two things are deliberately put in every list: runs from slack days, where recovering
    a wide box is the intuitive move and buys almost nothing per unit width, and runs from days on which
    neither cell has an unrecorded member, which cannot possibly matter however wide they are.  Lists are
    tried from tight to loose, because it is the wide-box slack-day runs that make a greedy ranking
    accidentally optimal and so make the item undiscriminating.

    For each list the search keeps at most two items: the cheapest target on which greedy needs strictly
    more recoveries than the optimum (the one G7 cares about), and, if there is one, a target that cannot
    be met with fewer than three recoveries.
    """
    dd = b.ref
    out = []
    for sa, sc, days in live:
        if sa["curriculum"] not in WD.LATE_CURRICULA and sc["curriculum"] not in WD.LATE_CURRICULA:
            continue
        rel = sorted({r for d in days for r in dd.T.free[d]})
        far = sorted(r for d in sorted(dd.T.free) if d not in set(days) for r in dd.T.free[d])
        if not 4 <= len(rel) <= 40:
            continue
        got = []
        for cap in (Fraction(12, 100), Fraction(20, 100), Fraction(37, 100), Fraction(50, 100),
                    Fraction(76, 100)):
            cands = [r for r in rel if dd.box[r][1] - dd.box[r][0] <= cap]
            cands = sorted(cands, key=lambda r: (dd.box[r][1] - dd.box[r][0], r))
            pad = far[::max(1, len(far) // 9)][:14 - len(cands)]
            cands = sorted(cands[:14] + pad)
            if len(cands) < 10:
                continue
            widths = {}
            for k in range(0, MAX_K + 1):
                for sub in itertools.combinations(cands, k):
                    widths[sub] = dd.width(sa, sc, set(sub))
            seq = greedy_sequence(dd, sa, sc, cands, MAX_K)
            gw = [w for _, w in seq]
            take = {}
            for k in (2, 3):
                for tgt in sorted({w for s, w in widths.items() if len(s) == k}):
                    kmin = min(len(s) for s, w in widths.items() if w <= tgt)
                    if kmin != k:
                        continue
                    tgt = publishable_target(widths, kmin, tgt, TOL)
                    if tgt is None:
                        continue
                    gk = next((i + 1 for i, w in enumerate(gw) if w <= tgt), None)
                    lab = "search" if (gk is None or gk > kmin) else ("deep" if kmin >= 3 else None)
                    if lab and lab not in take:
                        sets = sorted(sorted(s) for s, w in widths.items()
                                      if len(s) == kmin and w <= tgt)
                        take[lab] = {"id": None, "from": sa, "to": sc, "candidates": cands,
                                     "target": tgt, "_k": kmin, "_sets": sets, "_greedy": gk,
                                     "_greedy_seq": [r for r, _ in seq], "_cap": str(cap)}
                    if len(take) == 2:
                        break
                if len(take) == 2:
                    break
            got += [take[k] for k in sorted(take)]
            if got:
                break
        out += got[:2]
    return out


def differs(kind, ref, got):
    if got is None:
        return True
    if kind == "contrasts":
        return abs(ref["lo"] - got["lo"]) > TOL or abs(ref["hi"] - got["hi"]) > TOL
    if kind == "values":
        return ref != got
    if kind == "widths":
        return abs(ref - got) > TOL
    return ref["k"] != got["k"]


def gap(kind, ref, got):
    if kind == "widths":
        return abs(ref - got)
    if kind == "contrasts":
        return max(abs(ref["lo"] - got["lo"]), abs(ref["hi"] - got["hi"]))
    return 1.0


def select(b, items, plans):
    """Greedy set cover over the competing procedures, weighted so scarcity beats volume."""
    dd = b.ref
    qall = {k: [dict(q, id="x%03d" % i) for i, q in enumerate(items[k])] for k in items}
    qall["plans"] = [dict(q, id="x%03d" % i) for i, q in enumerate(plans)]
    ans = {}
    for name, d in b.each():
        ans[name] = b.answer(d, qall)
    ref = ans["reference"]
    killable = {}
    for n in D.RULES:
        killable[n] = sum(1 for k in qall for q in qall[k]
                          if differs(k, ref[k][q["id"]], ans[n][k].get(q["id"])))
    weight = {n: 1.0 / max(1, killable[n]) for n in D.RULES}
    need = {n: MIN_FAIL for n in D.RULES}
    cap = {"contrasts": N_CONTRAST, "values": N_VALUE, "widths": N_WIDTH, "plans": N_PLAN}
    chosen = {k: [] for k in cap}
    rng = random.Random(5)
    seen = set()

    def sig(k, q):
        s = (k, tuple(sorted(q["from"].items())), tuple(sorted(q["to"].items())))
        if k == "values":
            return s + (q["run_id"],)
        if k == "widths":
            return s + (tuple(q["recover"]),)
        if k == "plans":
            return s + (tuple(q["candidates"]), str(q["target"]))
        return s

    # seed: both labels in `values`, and the empty recovery in `widths` (the family-B layer must be
    # graded too, so a wrong answer there is distinguishable from a wrong answer about recovery).
    # G14 adds the rest: both labels inside *every* reason a value can be missing, so that no single
    # published column about a missing run - retention-log membership, `reason`, `dropped_at` - is on its
    # own enough to reproduce the label column.
    def _why(rid):
        if rid in dd.T.value:
            return "recorded"
        return dd.T.dropped[rid][0] if rid in dd.T.dropped else "never_written"

    seed_preds = [("values", lambda q: ref["values"][q["id"]] == "yes"),
                  ("values", lambda q: ref["values"][q["id"]] == "no"),
                  ("widths", lambda q: not q["recover"])]
    for w in sorted({_why(q["run_id"]) for q in qall["values"]}):
        for lab in ("yes", "no"):
            seed_preds.append(("values", lambda q, w=w, l=lab:
                               _why(q["run_id"]) == w and ref["values"][q["id"]] == l))
    for k, pred in seed_preds:
        pick = None
        for q in qall[k]:
            if not pred(q) or sig(k, q) in seen:
                continue
            kills = [n for n in need if need[n] > 0 and differs(k, ref[k][q["id"]], ans[n][k][q["id"]])]
            sc = (sum(weight[n] for n in kills), len(kills), rng.random())
            if pick is None or sc > pick[0]:
                pick = (sc, q, kills)
        if pick:
            _, q, kills = pick
            chosen[k].append(q)
            seen.add(sig(k, q))
            for n in kills:
                need[n] -= 1

    while sum(len(v) for v in chosen.values()) < sum(cap.values()):
        best = None
        for k in cap:
            if len(chosen[k]) >= cap[k]:
                continue
            for q in qall[k]:
                if sig(k, q) in seen:
                    continue
                kills = [n for n in need if need[n] > 0
                         and differs(k, ref[k][q["id"]], ans[n][k][q["id"]])]
                score = (sum(weight[n] for n in kills), len(kills),
                         min([gap(k, ref[k][q["id"]], ans[n][k][q["id"]]) for n in kills] or [0]),
                         rng.random())
                if best is None or score > best[0]:
                    best = (score, k, q, kills)
        if best is None:
            break
        _, k, q, kills = best
        chosen[k].append(q)
        seen.add(sig(k, q))
        for n in kills:
            need[n] -= 1

    out, remap = {}, {}
    pre = {"contrasts": "k%02d", "values": "v%02d", "widths": "w%02d", "plans": "p%02d"}
    for k in ("contrasts", "values", "widths", "plans"):
        out[k] = []
        for i, q in enumerate(sorted(chosen[k], key=lambda x: x["id"])):
            q = dict(q)
            remap[(k, pre[k] % (i + 1))] = q["id"]
            q["id"] = pre[k] % (i + 1)
            out[k].append(q)
    # Carry the answers the selector already computed over to the published ids rather than recomputing
    # them: it is the same archive and the same items, and recomputing costs more than the whole rest of
    # the build.  `main` re-derives the reference answers for the plan items anyway, as a cross-check.
    carried = {n: {k: {q["id"]: ans[n][k][remap[(k, q["id"])]] for q in out[k]} for k in out}
               for n in ans}
    return out, need, killable, carried


# ---------------------------------------------------------------------------------------------------
# certification


def day_parts_all(dd, a, b, day, S):
    """The day's decomposition with the *full* surplus, i.e. before S is taken out of the groups."""
    return dd.parts(a, b, day, S)


def witness_values(dd, a, b, S):
    """Concrete recovered values that attain the published guarantee.

    For each day, `argmax_surplus` says what the remaining surplus has to be for that day's width to be
    the worst case; the recovered runs on that day therefore have to absorb exactly the difference, and
    any split of it across their boxes will do.  Filling them in order is the simplest such split, and
    because every box endpoint and every day total is a multiple of 1e-4 the values it produces are legal
    `pass_rate` numbers rather than arithmetic conveniences - which the caller checks.
    """
    S = set(S)
    arg = dd.argmax_surplus(a, b, S)
    vals = {}
    for day in sorted(dd.T.free):
        here = [r for r in dd.T.free[day] if r in S]
        if not here:
            continue
        T = dd.T.resid[day] - sum((dd.box[r][0] for r in dd.T.free[day]), ZERO)
        left = T - arg[day][0]
        assert left >= 0, (day, T, arg[day][0])
        for r in sorted(here):
            lo, hi = dd.box[r]
            take = min(hi - lo, left)
            vals[r] = lo + take
            left -= take
        assert left == 0, ("surplus left over after filling the recovered boxes", day, left)
    return vals


def inject(dd, vals):
    """A `truth_b.Truth` that has been told the recovered values, built by hand rather than by rerunning
    the reader, so the check does not go back through any of the machinery being checked."""
    import copy
    t2 = copy.deepcopy(dd.T)
    for rid, v in vals.items():
        day = t2.day[rid]
        t2.value[rid] = v
        t2.resid[day] -= v
        t2.free[day] = [r for r in t2.free[day] if r != rid]
        if not t2.free[day]:
            del t2.free[day]
    return t2


def grid_ok(v):
    return (v * 10000).denominator == 1


def certify(b, w, queries, plans_meta):
    dd = b.ref
    cert = {}
    hidden = {k: Fraction(str(v)) for k, v in w["hidden"].items()}

    # G2 - the published archive is true of the world that produced it.
    worst = ZERO
    for rid in sorted(dd.T.started):
        if rid in dd.unknown:
            lo, hi = dd.box[rid]
            y = hidden[rid]
            assert lo <= y <= hi, ("run outside its published support", rid, float(lo), float(y),
                                   float(hi))
            worst = max(worst, min(y - lo, hi - y))
    day_err = ZERO
    for day, rids in dd.T.free.items():
        day_err = max(day_err, abs(sum((hidden[r] for r in rids), ZERO) - dd.T.resid[day]))
    assert day_err == 0, float(day_err)
    inside = []
    for q in queries["contrasts"]:
        lo, hi = dd.T.diff(q["from"], q["to"])
        pa, pb = dd._pop(q["from"]), dd._pop(q["to"])
        true = (sum((hidden[r] for r in pb), ZERO) / len(pb)
                - sum((hidden[r] for r in pa), ZERO) / len(pa))
        assert lo <= true <= hi, (q["id"], float(lo), float(true), float(hi))
        inside.append(float(min(true - lo, hi - true)))
    cert["G2_validity"] = {"unrecorded_runs": len(dd.unknown), "worst_day_total_error": float(day_err),
                           "tightest_run_slack": float(worst),
                           "tightest_contrast_slack": round(min(inside), 8)}

    # G3 - every published width is attained by a legal recovery outcome, checked through truth_b.
    att = []
    todo = [(q["id"], q["from"], q["to"], list(q["recover"])) for q in queries["widths"]]
    for q in queries["plans"]:
        todo.append((q["id"], q["from"], q["to"], plans_meta[q["id"]]["_sets"][0]))
    for qid, a, c, S in todo:
        want = dd.width(a, c, set(S))
        vals = witness_values(dd, a, c, S)
        assert all(grid_ok(v) for v in vals.values()), ("witness off the 1e-4 grid", qid, vals)
        for rid, v in vals.items():
            lo, hi = dd.box[rid]
            assert lo <= v <= hi, ("witness outside the box", qid, rid)
        t2 = inject(dd, vals)
        lo, hi = t2.diff(a, c)
        att.append({"item": qid, "k": len(S), "width": fmt(want), "witness_width": fmt(hi - lo),
                    "attained": (hi - lo) == want})
        assert (hi - lo) == want, ("published width not attained", qid, float(want), float(hi - lo))
    cert["G3_attainment"] = {"items": len(att), "all_attained": all(x["attained"] for x in att),
                             "sample": att[:4]}

    # G4 - and no legal recovery outcome does worse than the guarantee.  Sampling, not proof, but it is
    # an independent path: random outcome, inject, re-derive through truth_b, compare.
    rng = random.Random(23)
    checked, tight = 0, []
    for qid, a, c, S in todo:
        if not S:
            continue
        want = dd.width(a, c, set(S))
        best = ZERO
        for _ in range(N_SAMPLE):
            vals = {}
            for day in sorted(dd.T.free):
                here = sorted(r for r in dd.T.free[day] if r in set(S))
                if not here:
                    continue
                T = dd.T.resid[day] - sum((dd.box[r][0] for r in dd.T.free[day]), ZERO)
                rest = sum((dd.box[r][1] - dd.box[r][0] for r in dd.T.free[day] if r not in set(S)),
                           ZERO)
                cs = sum((dd.box[r][1] - dd.box[r][0] for r in here), ZERO)
                lo_s, hi_s = max(ZERO, T - rest), min(T, cs)
                steps = int((hi_s - lo_s) * 10000)
                sigma = lo_s + Fraction(rng.randint(0, steps), 10000)
                order = here[:]
                rng.shuffle(order)
                left = sigma
                for r in order:
                    lo, hi = dd.box[r]
                    take = min(hi - lo, left)
                    vals[r] = lo + take
                    left -= take
                assert left == 0
            t2 = inject(dd, vals)
            lo, hi = t2.diff(a, c)
            assert hi - lo <= want, ("a legal recovery outcome beat the guarantee", qid, float(hi - lo))
            best = max(best, hi - lo)
            checked += 1
        tight.append(best == want)
    cert["G4_soundness_sampling"] = {"outcomes_checked": checked,
                                     "items_where_sampling_also_hit_the_max": sum(tight),
                                     "items": len(tight)}
    return cert


def coverage(b, queries, plans_meta, need, killable, ans):
    """G5-G8: the shipped item set discriminates, and it covers all three day regimes."""
    dd = b.ref
    cert = {}
    cert["G5_decoys"] = {"min_fail": MIN_FAIL,
                         "still_needed": {n: v for n, v in need.items() if v > 0},
                         "items_each_rule_gets_wrong":
                             {n: sum(1 for k in queries for q in queries[k]
                                     if differs(k, ans["reference"][k][q["id"]],
                                                ans[n][k].get(q["id"])))
                              for n in D.RULES},
                         "items_in_pool_each_rule_could_lose": killable}
    assert not any(v > 0 for v in need.values()), need

    labels = sorted({ans["reference"]["values"][q["id"]] for q in queries["values"]})
    yes_o, no_widest = [], []
    for q in queries["values"]:
        rid = q["run_id"]
        g = dd.group(q["from"], q["to"], rid)
        lab = ans["reference"]["values"][q["id"]]
        day = dd.T.day[rid]
        widest = max(dd.box[r][1] - dd.box[r][0] for r in dd.T.free[day])
        if lab == "yes" and g == "O":
            yes_o.append(q["id"])
        if lab == "no" and (dd.box[rid][1] - dd.box[rid][0]) == widest:
            no_widest.append(q["id"])
    plan_no_member = [q["id"] for q in queries["plans"]
                      if all(dd.group(q["from"], q["to"], r) == "O"
                             for r in plans_meta[q["id"]]["_sets"][0])]
    cert["G6_coverage"] = {"value_labels": labels,
                           "yes_for_a_run_in_neither_cell": yes_o,
                           "no_for_the_widest_box_on_its_day": no_widest,
                           "plans_whose_optimum_is_all_non_members": plan_no_member}
    assert labels == ["no", "yes"], labels
    assert len(yes_o) >= 2 and len(no_widest) >= 2 and plan_no_member, cert["G6_coverage"]

    search = [{"item": q["id"], "k": plans_meta[q["id"]]["_k"],
               "greedy_needs": plans_meta[q["id"]]["_greedy"],
               "greedy_order": plans_meta[q["id"]]["_greedy_seq"],
               "optimal_sets": plans_meta[q["id"]]["_sets"]}
              for q in queries["plans"]]
    cert["G7_search"] = {"plans": search,
                         "with_greedy_strictly_worse":
                             [s["item"] for s in search
                              if s["greedy_needs"] is None or s["greedy_needs"] > s["k"]]}
    assert cert["G7_search"]["with_greedy_strictly_worse"], search

    # G8 - the three day shapes, classified by what a single recovery does on that day rather than by
    # the parameters that produce them, so the classification cannot drift away from the behaviour.
    seen = {}
    for q in queries["widths"] + queries["values"]:
        a, c = q["from"], q["to"]
        for day in dd.relevant_days(a, c):
            na, nb = dd.n(a), dd.n(c)
            w0 = dd._day_width(a, c, day, set(), na, nb)
            helps = {g: 0 for g in "PMO"}
            for rid in dd.T.free[day]:
                if dd._day_width(a, c, day, {rid}, na, nb) < w0:
                    helps[dd.group(a, c, rid)] += 1
            Lp, cP, Lm, cM, Lo, cO, T, cs = dd.parts(a, c, day, set())
            forced = any(max(ZERO, T - (cP + cM + cO - (dd.box[r][1] - dd.box[r][0]))) > 0
                         for r in dd.T.free[day])
            if helps["O"] == 0 and helps["P"] + helps["M"] > 0:
                lab = "members_only"
            elif helps["O"] > 0 and helps["P"] + helps["M"] == 0:
                lab = "non_members_only"
            elif helps["O"] + helps["P"] + helps["M"] == 0:
                lab = "nothing_helps"
            else:
                lab = "mixed"
            key = (lab, forced)
            seen.setdefault(key, []).append("%s@%s" % (day, q["id"]))
    cert["G8_regimes"] = {"%s%s" % (k[0], "_surplus_forced_down" if k[1] else ""): v[:3]
                          for k, v in sorted(seen.items())}
    labs = {k[0] for k in seen}
    assert {"members_only", "non_members_only"} <= labs, sorted(labs)
    assert any(k[1] for k in seen), "no shipped day forces the surplus down"
    return cert


INSTRUCTION = """You are the on-call analyst for a model-evaluation fleet.  A six-week sweep and three
short follow-up campaigns have finished; the archive in `/app` is what survived of them.

Every planned run appears in `plan.csv` with its four configuration knobs (`suite`, `curriculum`,
`retrieval`, `decoder`).  `launch_log.csv` says which of them actually started.  Each launched run
produced a single number, its `pass_rate` on the suite it was assigned - but the archive lost some of
those numbers, in several different ways.  `incidents.md` is the on-call write-up of what happened,
`metric_card.md` says what the number means, and `recovery_service.md` describes the cold-storage
recovery service.  Read all three, and read the logs: how much each surviving record still constrains a
missing value is for you to work out.  Nothing in `/app` is a decoy: every file is exactly what it claims
to be, and no file is wrong.

## The quantity

Fix a configuration `c` (one value for each of the four knobs).  `L(c)` is the set of run ids that appear
in `launch_log.csv` *and* whose `plan.csv` row has exactly those four knob values; count each run id once,
however many rows mention it.  `N(c) = |L(c)|`, and `m(c)` is the plain average of the true `pass_rate` of
all `N(c)` of them, including the ones whose value the archive lost.

Call an assignment of a value to every launched run whose `pass_rate` the archive did not keep
*consistent* if it contradicts nothing in `/app`.  For a contrast `m(to) - m(from)`, its **sharp
interval** is the smallest closed interval containing its value under every consistent assignment.

## Recoveries

A recovery request names a set `S` of launched runs and returns the true `pass_rate` of each.  You must
choose `S` before you see any of the values.  Once the values in `S` are known, the sharp interval of a
contrast is recomputed with those values fixed, and it can only get narrower or stay the same.

Which values come back is not yours to choose and not determined by `/app` - so the only thing you can
promise your budget committee before paying is the **guaranteed width**:

    guaranteed_width(contrast, S)
        = the largest width the contrast's sharp interval can have, taken over every consistent
          assignment of values to the runs in S.

With `S` empty this is just the width of the sharp interval today.  Recovering a run is **worth taking**
for a contrast when adding it to an empty `S` makes the guaranteed width strictly smaller.

## What to report

For each item in `/app/queries.json`:

  * `contrasts`: the sharp interval for `m(to) - m(from)` today, as `lo` and `hi`.
  * `values`: is recovering the single run `run_id` worth taking for that contrast?  `"yes"` or `"no"`.
  * `widths`: `guaranteed_width` for that contrast after recovering the runs listed in `recover`
    (possibly the empty list).
  * `plans`: the smallest number `k` of runs from `candidates` whose recovery guarantees a width of at
    most `target`, together with one set of that size that achieves it.  Every run you name must come
    from that item's `candidates` list.  If several sets of size `k` work, any one of them is accepted.

Write `/app/answers.json`:

```json
{
  "contrasts": {"k01": {"lo": -0.0182, "hi": 0.0413}},
  "values":    {"v01": "yes"},
  "widths":    {"w01": 0.135294},
  "plans":     {"p01": {"k": 2, "runs": ["r01234", "r02345"]}}
}
```

Every id in `queries.json` must appear.  Numbers are graded to an absolute tolerance of %(tol)s, so give
at least six decimals; the answers are exact arithmetic, not estimates.  Scoring is all-or-nothing over
the %(n)d items.  Do not edit anything under `/app` except `answers.json`.
"""

VERIFY = '''"""Grade `d-design-a`.  Compares against the key only - no archive reading, no re-derivation."""
import json, os, sys

KEY = json.load(open(os.path.join(os.environ.get("TESTS", "/tests"), "key.json")))
APP = os.environ.get("APP", "/app")
sys.path.insert(0, os.environ.get("TESTS", "/tests"))
import harbor as H


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    failed, by = [], {}
    try:
        a = json.load(open(os.path.join(APP, "answers.json")))
        assert isinstance(a, dict)
    except Exception as e:
        return H.reward(False, {"error": "answers.json unreadable: %s" % type(e).__name__})
    tol = KEY["tol"]
    for sec in ("contrasts", "values", "widths", "plans"):
        got = a.get(sec)
        by[sec] = {"n": len(KEY[sec]), "failed": 0}
        if not isinstance(got, dict):
            failed.append("%s:missing_section" % sec)
            by[sec]["failed"] = len(KEY[sec])
            continue
        for qid, want in sorted(KEY[sec].items()):
            g = got.get(qid)
            bad = None
            if sec == "contrasts":
                if not isinstance(g, dict) or not num(g.get("lo")) or not num(g.get("hi")):
                    bad = "shape"
                elif abs(g["lo"] - want["lo"]) > tol or abs(g["hi"] - want["hi"]) > tol:
                    bad = "value"
            elif sec == "values":
                if not isinstance(g, str):
                    bad = "shape"
                elif g.strip().lower() != want:
                    bad = "value"
            elif sec == "widths":
                if not num(g):
                    bad = "shape"
                elif abs(g - want) > tol:
                    bad = "value"
            else:
                if not isinstance(g, dict) or not isinstance(g.get("k"), int) \\
                        or not isinstance(g.get("runs"), list):
                    bad = "shape"
                elif g["k"] != want["k"]:
                    bad = "k"
                elif len(g["runs"]) != want["k"] or len(set(g["runs"])) != want["k"]:
                    bad = "set_size"
                elif sorted(str(x) for x in g["runs"]) not in want["sets"]:
                    bad = "set"
            if bad:
                failed.append("%s:%s:%s" % (sec, qid, bad))
                by[sec]["failed"] += 1
    return H.reward(not failed, {"failed": failed[:40], "n_failed": len(failed), "by_section": by})


if __name__ == "__main__":
    sys.exit(main())
'''

SOLVE = '''"""Reference solution for `d-design-a`."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_d as D

APP = sys.argv[1] if len(sys.argv) > 1 else "/app"
from fractions import Fraction

dd = D.Design(APP)
q = json.load(open(os.path.join(APP, "queries.json")))
out = {"contrasts": {}, "values": {}, "widths": {}, "plans": {}}
for it in q["contrasts"]:
    lo, hi = dd.T.diff(it["from"], it["to"])
    out["contrasts"][it["id"]] = {"lo": float(lo), "hi": float(hi)}
for it in q["values"]:
    out["values"][it["id"]] = "yes" if dd.voi(it["from"], it["to"], it["run_id"]) else "no"
for it in q["widths"]:
    out["widths"][it["id"]] = float(dd.width(it["from"], it["to"], set(it["recover"])))
for it in q["plans"]:
    k, sets = dd.plan(it["from"], it["to"], it["candidates"], Fraction(str(it["target"])))
    out["plans"][it["id"]] = {"k": k, "runs": sets[0]}
json.dump(out, open(os.path.join(APP, "answers.json"), "w"), indent=1)
print(json.dumps({k: len(v) for k, v in out.items()}))
'''


def meta_target(qid, meta):
    return meta[qid]["_target"]


def publish(queries):
    """Strip the generator's bookkeeping off the plan items, and keep the meta separately."""
    meta = {q["id"]: dict({k: v for k, v in q.items() if k.startswith("_")},
                          _target=q["target"]) for q in queries["plans"]}
    out = {k: [dict(q) for q in queries[k]] for k in queries}
    for q in out["plans"]:
        for k in [k for k in q if k.startswith("_")]:
            q.pop(k)
        q["target"] = float(q["target"])
        assert Fraction(str(q["target"])) == meta_target(q["id"], meta), q["id"]
        q["candidates"] = sorted(q["candidates"])
    return out, meta


def main():
    w = WD.build(0)
    app = os.path.join(OUT, "environment", "app")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(app)
    emit(app, w)

    b = Bench(app)
    items, live = build_items(b)
    plans = plan_items(b, live)
    assert plans, "no plan item survived the search"
    queries, need, killable, ans = select(b, items, plans)
    pub, meta = publish(queries)
    plans_meta = {q["id"]: dict(meta[q["id"]]) for q in queries["plans"]}

    # The plan items are re-derived from the published `pub` here, so a discrepancy between what the
    # selector scored and what the agent will see shows up as an assertion rather than silently.
    for q in pub["plans"]:
        k, sets = b.ref.plan(q["from"], q["to"], q["candidates"], Fraction(str(q["target"])))
        assert k == plans_meta[q["id"]]["_k"], (q["id"], k, plans_meta[q["id"]]["_k"])
        assert sorted(sets) == sorted(plans_meta[q["id"]]["_sets"]), q["id"]

    cert = certify(b, w, pub, plans_meta)
    cert.update(coverage(b, pub, plans_meta, need, killable, ans))
    n_items = sum(len(v) for v in pub.values())
    ok = (not any(v > 0 for v in need.values())) and cert["G3_attainment"]["all_attained"]

    json.dump(pub, open(os.path.join(app, "queries.json"), "w"), indent=1)
    ref = ans["reference"]
    key = {"tol": TOL,
           "contrasts": ref["contrasts"], "values": ref["values"], "widths": ref["widths"],
           "plans": {q["id"]: {"k": plans_meta[q["id"]]["_k"],
                               "sets": [sorted(s) for s in plans_meta[q["id"]]["_sets"]]}
                     for q in pub["plans"]}}

    H.w(os.path.join(OUT, "instruction.md"),
        INSTRUCTION % {"tol": TOL, "n": n_items} + H.SUFFIX_T.format(t=5400))
    H.w(os.path.join(OUT, "task.toml"),
        H.task_toml(artifacts=["answers.json"], family="d-design-a",
                    tags=["value-of-information", "partial-identification", "experiment-design",
                          "combinatorial-search", "ai4ai-eval"],
                    agent_timeout=5400, expert_hours=4.0, verifier_timeout=300))
    os.makedirs(os.path.join(OUT, "tests"))
    H.w(os.path.join(OUT, "tests", "verify_d.py"), VERIFY)
    shutil.copy(os.path.join(ROOT, "core", "harbor.py"), os.path.join(OUT, "tests", "harbor.py"))
    json.dump(key, open(os.path.join(OUT, "tests", "key.json"), "w"), indent=1)
    H.w(os.path.join(OUT, "tests", "test.sh"), "#!/bin/bash\nset -eu\npython3 /tests/verify_d.py\n",
        mode=0o755)
    H.w(os.path.join(OUT, "tests", "Dockerfile"),
        "FROM python:3.11-slim\nCOPY verify_d.py harbor.py key.json test.sh /tests/\n")
    os.makedirs(os.path.join(OUT, "solution"))
    H.w(os.path.join(OUT, "solution", "ref_solve_d.py"), SOLVE)
    for f in ("design_d.py", "truth_b.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(OUT, "solution", f))
    H.w(os.path.join(OUT, "solution", "solve.sh"),
        "#!/bin/bash\nset -eu\npython3 /solution/ref_solve_d.py /app\n", mode=0o755)
    os.makedirs(os.path.join(OUT, "authoring"))
    json.dump({"accepted": ok, "n_items": n_items, "gates": cert},
              open(os.path.join(OUT, "authoring", "certificate.json"), "w"), indent=1)

    print("items", {k: len(v) for k, v in pub.items()}, "total", n_items)
    print("G2", cert["G2_validity"])
    print("G3", {k: v for k, v in cert["G3_attainment"].items() if k != "sample"})
    print("G4", cert["G4_soundness_sampling"])
    print("G6", {k: (v if isinstance(v, list) and len(v) < 4 else len(v))
                 for k, v in cert["G6_coverage"].items()})
    print("G7 greedy-beats:", cert["G7_search"]["with_greedy_strictly_worse"])
    print("G8 regimes:", sorted(cert["G8_regimes"]))
    for n, v in sorted(cert["G5_decoys"]["items_each_rule_gets_wrong"].items(), key=lambda x: x[1]):
        print("  %-38s wrong_on=%-3d could_lose_in_pool=%d" % (n, v, killable[n]))
    print("underkilled:", cert["G5_decoys"]["still_needed"])
    print("accepted:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
