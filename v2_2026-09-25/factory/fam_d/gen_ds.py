"""Build, certify and export `d-design-sealed`: the arm where the *input* is not specified.

Why this arm exists.  `d-design-a` shipped a brand-new estimand - the guaranteed width of a contrast
after a recovery you have not made yet - with three planted day shapes, a double decoy, and a plan item
whose optimum a marginal-gain ranking cannot reach.  Both frontier models scored 27/27 in about twelve
minutes.  Neither did any of the reasoning the decoys were built to punish: they read the instruction's
definition, wrote an exact solver (rationals, a continuous knapsack per day, brute force over subsets),
sampled to check it, and submitted.  The lesson is that a *precisely formalised* question, however novel,
is a coding task, and these models code correctly.

So this arm keeps the target formalised - it has to be, or grading becomes a matter of opinion - and
takes the specification away from the *input* instead.  The instruction says what a guaranteed width is.
It does not say what a recovery request returns, and the answer is not the same for every run:

  * for most runs the bundle is intact and the exact value comes back;
  * for a run the quality screen removed, the screen also redacted the bundle down to its own audit line,
    so what comes back is a *band* - an interval containing the value, of the resolution that suite is
    bucketed at, and the resolution differs by suite;
  * for a run whose row the results sink never delivered, there is no bundle at all and the request comes
    back empty.

That last one is the inversion the arm turns on.  Cold storage reads like a backup of what the archive
lost, so the runs with nothing in the archive look like the ones worth paying to recover; in fact the
bundle is finalised by the same sink, so those are exactly the runs that cannot be recovered - and they
are the widest boxes in the sweep.  The only evidence is `recovery_log.csv`, a sample of requests the
on-call team issued during the incident response and what each came back with.  It is not arranged for
convenience, it never mentions a run any shipped item asks about, and the rule has to be carried from it
to runs it does not mention.

A solver cannot be written from the instruction here, because the instruction does not say what the
solver's input is.

Gates.  G2-G8 are family D's, re-run against this arm (G3 and G4 are rewritten, because a witness now has
to place *windows* rather than fix values).  Added:

  G9   *identifiability* - exactly one reading in `RECOV_RULES` is consistent with every row of
       `recovery_log.csv`, and it is the one the world was built with.  Two survivors would leave the
       items with no defensible answer;
  G10  *no lookup* - no run named in any shipped item appears in `recovery_log.csv`, so the log can never
       be read off instead of generalised from;
  G11  *the inversion is load-bearing* - the `d-design-a` reading, in which every request returns an
       exact value, must be wrong on at least MIN_FAIL shipped items in each of `recovery`, `values` and
       `widths`, and on at least one plan.  An agent that loses the inference cannot keep the score;
  G12  *the resolutions are pinned and are not constant* - every suite a shipped item depends on has at
       least two banded rows in the log, and no two suites share a resolution, so "one resolution for the
       fleet" is refuted rather than merely unsupported.
"""
import copy
import itertools
import json
import os
import random
import shutil
import sys
import types
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
import design_d as D                                                          # noqa: E402
import gen_d as G                                                             # noqa: E402
import harbor as H                                                            # noqa: E402
import truth_b as TB                                                          # noqa: E402
import world_d as WD                                                          # noqa: E402

TOL = G.TOL
MIN_FAIL = 2
N_CONTRAST, N_RECOV, N_VALUE, N_WIDTH, N_PLAN = 3, 8, 10, 7, 3
MAX_K = 3
N_SAMPLE = 40
N_LOG_FILLER = 12
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/d-design-sealed"
ZERO = Fraction(0)
GONE = Fraction(10)          # any residual at or above a box width means "the request came back empty"

# The screen's audit resolution, per suite.  Deliberately different for all three, so that a reading in
# which the fleet has one resolution is refuted by the log rather than merely unsupported (G12), and large
# enough that a band and an exact value are told apart well outside the grading tolerance.
BANDS = {"arc_lite": Fraction(400, 10000),
         "gsm_plus": Fraction(200, 10000),
         "code_ms": Fraction(250, 10000)}


# ---------------------------------------------------------------------------------------------------
# what a recovery returns: the thing the agent has to infer


class Ctx(object):
    """Everything a candidate reading is allowed to look at - all of it readable from `/app`."""

    def __init__(self, app, dd):
        self.T = dd.T
        self.box = dd.box
        self.worker = {r["run_id"]: r["worker"] for r in TB.rows(os.path.join(app, "launch_log.csv"))}
        self.suite = {rid: r["suite"] for rid, r in dd.T.plan.items()}
        self.guess = {k: Fraction(str(v["random_guess"]))
                      for k, v in json.load(open(os.path.join(app, "suites.json"))).items()}
        self.why = {}
        for rid in sorted(dd.T.started):
            if rid in dd.T.value:
                self.why[rid] = "recorded"
            elif rid in dd.T.dropped:
                self.why[rid] = dd.T.dropped[rid][0]
            else:
                self.why[rid] = "never_written"
        fins = [dd.T.fin[r] for r in dd.T.started
                if self.why[r] == "never_written" and self.worker[r] != WD.LATE_WORKER]
        self.outage = (min(fins), max(fins))

    def band(self, rid):
        return BANDS[self.suite[rid]]


def _gone_if(cond, otherwise=ZERO):
    return GONE if cond else otherwise


def _screen(c, r):
    return c.band(r) if c.why[r] == "below_retention_floor" else ZERO


# Candidate readings of the recovery service.  Each maps a run to the *residual width* a request for it
# leaves behind: 0 for an exact value, a positive width for a band, GONE for an empty request.  These are
# readings an analyst could actually hold after reading `/app` - the literal one, the intuitive inversion,
# the per-worker and per-window ones, the two ways of getting the resolution wrong, and the true one.
# G9 requires the log to leave exactly one standing.
RECOV_RULES = {
    "every_request_returns_the_exact_value":
        lambda c, r: ZERO,
    "cold_storage_is_the_backup_for_whatever_the_archive_lost":
        lambda c, r: _gone_if(c.why[r] == "recorded"),
    "nothing_the_archive_lost_can_be_recovered":
        lambda c, r: _gone_if(c.why[r] != "recorded"),
    "the_sink_outage_took_the_bundles_and_the_screen_left_them_alone":
        lambda c, r: _gone_if(c.why[r] == "never_written"),
    "the_screen_deleted_the_bundles_and_the_sink_left_them_alone":
        lambda c, r: _gone_if(c.why[r] == "below_retention_floor"),
    "the_screen_redacted_what_the_sink_delivered":
        lambda c, r: (c.band(r) if c.why[r] == "never_written" else ZERO),
    "one_audit_resolution_for_the_whole_fleet":
        lambda c, r: (GONE if c.why[r] == "never_written"
                      else (BANDS["arc_lite"] if c.why[r] == "below_retention_floor" else ZERO)),
    "the_audit_resolution_is_the_suites_chance_rate":
        lambda c, r: (GONE if c.why[r] == "never_written"
                      else (c.guess[c.suite[r]] if c.why[r] == "below_retention_floor" else ZERO)),
    "the_two_outage_workers_lost_everything":
        lambda c, r: (GONE if c.worker[r] in ("w03", "w07") else _screen(c, r)),
    "the_three_sink_workers_lost_everything":
        lambda c, r: (GONE if c.worker[r] in ("w03", "w07", WD.LATE_WORKER) else _screen(c, r)),
    "nothing_that_finished_during_the_outage_survived":
        lambda c, r: (GONE if c.outage[0] <= c.T.fin[r] <= c.outage[1] else _screen(c, r)),
    "the_lost_shard_took_its_bundles_too":
        lambda c, r: (GONE if c.why[r] in ("never_written", "shard_file_lost") else _screen(c, r)),
    # The truth.  The bundle's closing record is written by the results sink, so a run whose row the sink
    # never delivered has no bundle at all - whichever incident stopped the sink, on whichever worker.
    # The screen does not delete a bundle; it redacts it to its own audit line, which keeps the score only
    # to the resolution that suite is bucketed at.
    "the_sink_finalises_the_bundle_and_the_screen_redacts_it":
        lambda c, r: (GONE if c.why[r] == "never_written" else _screen(c, r)),
}
TRUTH_RULE = "the_sink_finalises_the_bundle_and_the_screen_redacts_it"
NAIVE_RULE = "every_request_returns_the_exact_value"
# The reading G11 is built to punish.  For the sealed arm it is the `d-design-a` one; an arm that turns on
# a different confusion points this at whichever reading a competent-but-overconfident agent would land on.
RIVAL = NAIVE_RULE


def grain_of(ctx, name, runs):
    f = RECOV_RULES[name]
    return {r: f(ctx, r) for r in runs}


def observed(ctx, rid, name):
    """What a request for `rid` would look like in the log under this reading: the residual width,
    clamped to the run's own box, which is all an observer could tell apart."""
    lo, hi = ctx.box[rid]
    u = RECOV_RULES[name](ctx, rid)
    return (hi - lo) if u > hi - lo else u


def build_log(ctx, dd, hidden, rng, avoid, spare=()):
    """A sample of past recovery requests that pins the reading down and nothing else.

    Construction is adversarial against the *other* readings rather than in favour of the true one: for
    every candidate that is not the truth, the log must carry at least two runs on which that candidate
    predicts a different outcome from the truth.  Runs named in shipped items are excluded, so the log can
    never be read as a lookup table.  Band placements are deliberately off the resolution grid, because a
    reader who assumed bands were aligned would infer a narrower reveal than the service actually gives.
    """
    pool = sorted(r for r in dd.unknown if r not in avoid)
    chosen = []
    for name in sorted(RECOV_RULES):
        if name == TRUTH_RULE or name in spare:
            continue
        wrong = [r for r in pool
                 if observed(ctx, r, name) != observed(ctx, r, TRUTH_RULE) and r not in chosen]
        rng.shuffle(wrong)
        kinds = {}
        for r in wrong:
            kinds.setdefault(observed(ctx, r, TRUTH_RULE) == 0, []).append(r)
        take = [v[0] for v in kinds.values()][:2]
        if len(take) < 2:
            take = (take + [r for r in wrong if r not in take])[:2]
        assert len(take) >= 2, ("the log cannot witness against %s" % name)
        chosen += take
    # Every suite a band is possible for has to be pinned twice over (G12), then filler for realism.
    # A run whose own support is narrower than its suite's resolution shows the same thing as an empty
    # request and so pins nothing; those are no use as witnesses.
    def banded(r):
        return 0 < observed(ctx, r, TRUTH_RULE) < dd.box[r][1] - dd.box[r][0]

    for suite in sorted(BANDS):
        have = [r for r in chosen if ctx.suite[r] == suite and banded(r)]
        more = [r for r in pool if r not in chosen and ctx.suite[r] == suite and banded(r)]
        rng.shuffle(more)
        chosen += more[:max(0, 2 - len(have))]
    filler = [r for r in pool if r not in chosen]
    rng.shuffle(filler)
    chosen += filler[:N_LOG_FILLER]

    rows = []
    for rid in sorted(set(chosen)):
        lo, hi = dd.box[rid]
        u = observed(ctx, rid, TRUTH_RULE)
        v = hidden[rid]
        if u >= hi - lo:
            rows.append({"run_id": rid, "outcome": "no_bundle", "returned_lo": "", "returned_hi": ""})
            continue
        if u == 0:
            rows.append({"run_id": rid, "outcome": "value",
                         "returned_lo": "%.4f" % float(v), "returned_hi": "%.4f" % float(v)})
            continue
        off = Fraction(rng.randrange(1, int(u * 10000)), 10000)      # never aligned to the resolution
        p = max(lo, min(v - off, hi - u))
        assert p <= v <= p + u and lo <= p and p + u <= hi, (rid, float(p), float(v), float(u))
        rows.append({"run_id": rid, "outcome": "range",
                     "returned_lo": "%.4f" % float(p), "returned_hi": "%.4f" % float(p + u)})
    for i, r in enumerate(rows):
        r["requested_at"] = "2026-07-%02dT%02d:%02d:00Z" % (16 + i // 24, 6 + (i // 4) % 12, (i * 7) % 60)
    return rows


def read_back(ctx, rows):
    """The log as an observer reads it: run id -> the residual width that row shows."""
    out = {}
    for r in rows:
        if r["outcome"] == "no_bundle":
            lo, hi = ctx.box[r["run_id"]]
            out[r["run_id"]] = hi - lo
        else:
            out[r["run_id"]] = Fraction(r["returned_hi"]) - Fraction(r["returned_lo"])
    return out


# ---------------------------------------------------------------------------------------------------
# archive


RECOVERY_SEALED = """# Cold-storage recovery requests

The evaluation fleet writes a per-run **trace bundle** to cold storage.  A **recovery request** names a
set of launched runs and is answered out of those bundles.  Requests are slow and are charged to the
team's quarterly budget, which is why the planning questions in `queries.json` ask what a request would be
worth *before* it is issued.

Three things about the service matter for those questions:

  * a request is issued blind.  You commit to the whole set and pay for it; you do not get to see the
    first answer before choosing the second.
  * what comes back for a run is never wrong and never contradicts this archive, but it is **not the same
    for every run**.  Cold storage is a different system from the archive - written by the evaluation
    fleet, not by the archiver - and the incidents in `incidents.md` did not all reach the two systems the
    same way.  Nothing here, and nothing anywhere else in `/app`, states which runs answer which way:
    establishing that is part of the work.
  * the only record of how the service behaves is `recovery_log.csv`: every recovery request the on-call
    team issued during the incident response, with what came back.  Each row gives the run, the outcome,
    and, where anything came back, the range of `pass_rate` that answer left the run in (`returned_lo`
    and `returned_hi`, equal to each other when the answer was a single number).  It is a sample - most
    launched runs were never requested, and none of the runs `queries.json` asks about were.  It was not
    collected to answer your questions and nothing in it is arranged for your convenience.  What it shows
    about the service holds for the whole fleet, including runs it does not mention.

Where an answer leaves a run in a range rather than at a number, that range is a genuine constraint on the
run's `pass_rate` and nothing more: the service does not promise the value sits anywhere in particular
inside it, and where the range falls is not something this archive determines.
"""


def emit(app, w, log_rows):
    G.emit(app, w)
    H.w(os.path.join(app, "recovery_service.md"), RECOVERY_SEALED)
    H.csv_rows(os.path.join(app, "recovery_log.csv"),
               ["requested_at", "run_id", "outcome", "returned_lo", "returned_hi"], log_rows)


# ---------------------------------------------------------------------------------------------------
# the bench


class BenchS(G.Bench):
    """The archive under every reading that has to be told apart.

    The decoy set is deliberately not the full cross product of the two layers.  Separating an agent that
    gets both layers wrong from one that gets only one wrong buys nothing, and pricing a hundred and
    thirty readings costs more than it buys.  What has to be covered is each layer *alone*: the true
    recovery semantics under every wrong analysis, and the true analysis under every wrong semantics.
    """

    def __init__(self, app, ctx, runs):
        self.app = app
        self.ctx = ctx
        truth = grain_of(ctx, TRUTH_RULE, runs)
        self.ref = D.Design(app, None, None, truth)
        self.alt = {}
        for name in sorted(RECOV_RULES):
            if name != TRUTH_RULE:
                self.alt["recovery__" + name] = D.Design(app, None, None, grain_of(ctx, name, runs))
        for name, rule in sorted(D.RULES.items()):
            self.alt["analysis__" + name] = D.Design(app, rule, None, truth)

    def answer(self, dd, queries):
        out = G.Bench.answer(self, dd, queries)
        out["recovery"] = {q["id"]: G.fmt(dd.u(q["run_id"])) for q in queries["recovery"]}
        return out


def differs(kind, ref, got, sets=None):
    """Would this reading be graded wrong on this item?

    Family D compares plan answers by `k` alone, which is right there because every reading that gets `k`
    right gets the set right too.  Here it is not: a reading that thinks an unrecoverable run answers
    exactly will often name the same number of runs and name the wrong ones, and the shipped verifier
    rejects that.  So when the item's optimal sets are to hand, the set is compared as well.
    """
    if kind == "recovery":
        return got is None or abs(ref - got) > TOL
    if kind == "plans" and sets is not None:
        return got is None or got["k"] != ref["k"] or sorted(str(x) for x in got["runs"]) not in sets
    return G.differs(kind, ref, got)


def sets_of(kind, q):
    """The optimal sets a pooled plan item carries, if it is one."""
    return [[str(x) for x in s] for s in q["_sets"]] if kind == "plans" else None


def pub_sets(kind, q, plans_meta):
    """The same, for a published item, whose bookkeeping lives in the meta instead."""
    return sets_of(kind, plans_meta[q["id"]]) if kind == "plans" else None


def dead(dd, rid):
    return dd.u(rid) >= dd.box[rid][1] - dd.box[rid][0]


def why_of(dd, rid):
    """The archive's own account of why a run has no recorded value.

    This is the coarsest thing every published column about a missing run is a function of, so it is the
    right equivalence relation to require the graded labels to vary inside.
    """
    if rid in dd.T.value:
        return "recorded"
    return dd.T.dropped[rid][0] if rid in dd.T.dropped else "never_written"


def plan_items(b, live):
    """Plan items for the sealed arm.

    Family D chooses each item's candidate list, because in the world that list is just the set of runs
    the team is willing to pay to request.  Here it is chosen to make the *semantic* layer decide the
    answer: every list is padded with the widest boxes on the relevant days - the runs whose rows the sink
    never delivered, the most attractive candidates under the literal reading of the service and worth
    exactly nothing under the true one.  An agent that gets the semantics wrong does not merely mis-rank
    the candidates; it spends the whole budget on runs that cannot be recovered at all.
    """
    dd = b.ref
    out = []
    for sa, sc, days in live:
        rel = sorted({r for d in days for r in dd.T.free[d]})
        if not 4 <= len(rel) <= 40:
            continue
        far = sorted(r for d in sorted(dd.T.free) if d not in set(days) for r in dd.T.free[d])
        bait = sorted((r for r in rel if dead(dd, r)),
                      key=lambda r: (-(dd.box[r][1] - dd.box[r][0]), r))
        good = sorted((r for r in rel if not dead(dd, r)),
                      key=lambda r: (dd.box[r][1] - dd.box[r][0], r))
        if not good or not bait:
            continue
        cands = sorted(set(good[:9] + bait[:5] + far[::max(1, len(far) // 4)][:3]))
        if not 9 <= len(cands) <= 17:
            continue
        widths = {}
        for k in range(0, MAX_K + 1):
            for sub in itertools.combinations(cands, k):
                widths[sub] = dd.width(sa, sc, set(sub))
        seq = G.greedy_sequence(dd, sa, sc, cands, MAX_K)
        gw = [w for _, w in seq]
        take = {}
        for k in (1, 2, 3):
            for tgt in sorted({w for s, w in widths.items() if len(s) == k}):
                kmin = min(len(s) for s, w in widths.items() if w <= tgt)
                if kmin != k:
                    continue
                tgt = G.publishable_target(widths, kmin, tgt, TOL)
                if tgt is None:
                    continue
                gk = next((i + 1 for i, w in enumerate(gw) if w <= tgt), None)
                lab = "search" if (gk is None or gk > kmin) else ("deep" if kmin >= 2 else None)
                if lab and lab not in take:
                    sets = sorted(sorted(s) for s, w in widths.items() if len(s) == kmin and w <= tgt)
                    take[lab] = {"id": None, "from": sa, "to": sc, "candidates": cands, "target": tgt,
                                 "_k": kmin, "_sets": sets, "_greedy": gk,
                                 "_greedy_seq": [r for r, _ in seq],
                                 "_bait": [r for r in cands if dead(dd, r)]}
                if len(take) == 2:
                    break
            if len(take) == 2:
                break
        out += [take[k] for k in sorted(take)]
    return out


def select(b, items, plans, seeds=()):
    """Greedy set cover over the competing readings, weighted so scarcity beats volume."""
    dd = b.ref
    qall = {k: [dict(q, id="x%03d" % i) for i, q in enumerate(items[k])] for k in items}
    qall["plans"] = [dict(q, id="x%03d" % i) for i, q in enumerate(plans)]
    ans = {n: b.answer(d, qall) for n, d in b.each()}
    ref = ans["reference"]
    decoys = sorted(b.alt)
    killable = {n: sum(1 for k in qall for q in qall[k]
                       if differs(k, ref[k][q["id"]], ans[n][k].get(q["id"]), sets_of(k, q))) for n in decoys}
    weight = {n: 1.0 / max(1, killable[n]) for n in decoys}
    need = {n: MIN_FAIL for n in decoys}
    cap = {"contrasts": N_CONTRAST, "recovery": N_RECOV, "values": N_VALUE,
           "widths": N_WIDTH, "plans": N_PLAN}
    chosen = {k: [] for k in cap}
    rng = random.Random(7)
    seen = set()

    def sig(k, q):
        if k == "recovery":
            return (k, q["run_id"])
        s = (k, tuple(sorted(q["from"].items())), tuple(sorted(q["to"].items())))
        if k == "values":
            return s + (q["run_id"],)
        if k == "widths":
            return s + (tuple(q["recover"]),)
        if k == "plans":
            return s + (tuple(q["candidates"]), str(q["target"]))
        return s

    def grab(k, pred):
        pick = None
        for q in qall[k]:
            if not pred(q) or sig(k, q) in seen:
                continue
            kills = [n for n in need if need[n] > 0
                     and differs(k, ref[k][q["id"]], ans[n][k][q["id"]], sets_of(k, q))]
            sc = (sum(weight[n] for n in kills), len(kills), rng.random())
            if pick is None or sc > pick[0]:
                pick = (sc, q, kills)
        if pick:
            _, q, kills = pick
            chosen[k].append(q)
            seen.add(sig(k, q))
            for n in kills:
                need[n] -= 1

    # Seeds: all three recovery outcomes have to be graded, both value labels have to appear, the empty
    # recovery has to be among the widths, and at least one value item has to sit on an unrecoverable run
    # - otherwise a wrong reading of the service could hide behind a right answer.
    grab("recovery", lambda q: dd.u(q["run_id"]) == 0)
    grab("recovery", lambda q: 0 < dd.u(q["run_id"]) and not dead(dd, q["run_id"]))
    grab("recovery", lambda q: dead(dd, q["run_id"]))
    grab("values", lambda q: ref["values"][q["id"]] == "yes")
    grab("values", lambda q: ref["values"][q["id"]] == "no")
    grab("widths", lambda q: not q["recover"])
    grab("values", lambda q: dead(dd, q["run_id"]))

    # G11 is the gate this arm exists for, so the items that enforce it are seeded rather than left to the
    # weighting: whatever else the paper covers, an agent holding the `d-design-a` reading of the service
    # has to lose points in every section, not only in the one that asks about the service directly.
    nv = "recovery__" + RIVAL
    for sec, want in (("values", MIN_FAIL), ("widths", MIN_FAIL), ("plans", 1)):
        for _ in range(want):
            grab(sec, lambda q, s=sec: differs(s, ref[s][q["id"]], ans[nv][s][q["id"]], sets_of(s, q)))
    for sec, pred in seeds:
        grab(sec, pred)

    # G14 - the graded label has to vary inside every class the archive's own columns can separate.
    # Every raw column about a run that has no recorded value - whether it appears in `retention_log.csv`
    # at all, its `reason`, its `dropped_at` - is a proxy for *why* the value is missing.  If `values`
    # carried one label per reason, the whole section would be answerable by reading one column and
    # nothing about information would be tested; the audit gate found exactly that in the first open arm,
    # where `run.in:retention_log.csv` reproduced all ten labels.  Seeding both labels inside every
    # reason makes any such column non-informative on its own, which is the property the gate checks.
    for w in sorted({why_of(dd, q["run_id"]) for q in qall["values"]}):
        for lab in ("yes", "no"):
            grab("values", lambda q, w=w, l=lab:
                 why_of(dd, q["run_id"]) == w and ref["values"][q["id"]] == l)

    while sum(len(v) for v in chosen.values()) < sum(cap.values()):
        best = None
        for k in cap:
            if len(chosen[k]) >= cap[k]:
                continue
            for q in qall[k]:
                if sig(k, q) in seen:
                    continue
                kills = [n for n in need if need[n] > 0
                         and differs(k, ref[k][q["id"]], ans[n][k][q["id"]], sets_of(k, q))]
                score = (sum(weight[n] for n in kills), len(kills), rng.random())
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
    pre = {"contrasts": "k%02d", "recovery": "r%02d", "values": "v%02d",
           "widths": "w%02d", "plans": "p%02d"}
    for k in cap:
        out[k] = []
        for i, q in enumerate(sorted(chosen[k], key=lambda x: x["id"])):
            q = dict(q)
            remap[(k, pre[k] % (i + 1))] = q["id"]
            q["id"] = pre[k] % (i + 1)
            out[k].append(q)
    carried = {n: {k: {q["id"]: ans[n][k][remap[(k, q["id"])]] for q in out[k]} for k in out}
               for n in ans}
    return out, need, killable, carried


# ---------------------------------------------------------------------------------------------------
# certification


def witness(dd, a, b, S):
    """Concrete recovery outcomes that attain the published guarantee.

    `argmax_surplus` says what each day's remaining surplus has to be for that day's width to be the worst
    case, so the recovered runs on that day have to absorb exactly the difference - and a run absorbs it
    through where its window is *placed*, up to its box width less the window's own width.  Filling them
    in order is the simplest such split, and because every box endpoint, day total and resolution is a
    multiple of 1e-4, the windows it produces are ones the service could really have returned, which the
    caller checks.
    """
    S = set(S)
    arg = dd.argmax_surplus(a, b, S)
    wins = {}
    for day in sorted(dd.T.free):
        here = [r for r in dd.T.free[day] if r in S]
        if not here:
            continue
        T = dd.T.resid[day] - sum((dd.box[r][0] for r in dd.T.free[day]), ZERO)
        left = T - arg[day][0]
        assert left >= 0, (day, float(T), float(arg[day][0]))
        for r in sorted(here):
            lo, hi = dd.box[r]
            u = dd.u(r)
            take = min((hi - lo) - u, left)
            wins[r] = (lo + take, lo + take + u)
            left -= take
        assert left == 0, ("surplus left over after placing the windows", day, float(left))
    return wins


def inject(dd, wins):
    """A `truth_b.Truth` that has been told the recovery outcomes.

    Outcomes are applied by narrowing each run's own support rather than by rewriting the archive, so an
    exact answer and a band go through exactly the same path, and the re-derivation never re-enters the
    machinery being checked: `Truth.diff` reads supports and day totals and knows nothing about any of it.
    """
    t2 = copy.deepcopy(dd.T)
    t2._narrowed = dict(wins)

    def support(self, rid):
        if rid in self._narrowed:
            return self._narrowed[rid]
        return TB.Truth.support(self, rid)

    t2.support = types.MethodType(support, t2)
    return t2


def certify(b, ctx, hidden, log_rows, queries, plans_meta, expect=None):
    dd = b.ref
    cert = {}

    # G2 - the published archive is true of the world that produced it.
    worst = ZERO
    for rid in sorted(dd.unknown):
        lo, hi = dd.box[rid]
        assert lo <= hidden[rid] <= hi, ("run outside its published support", rid)
        worst = max(worst, min(hidden[rid] - lo, hi - hidden[rid]))
    day_err = max(abs(sum((hidden[r] for r in rids), ZERO) - dd.T.resid[day])
                  for day, rids in dd.T.free.items())
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

    # G3 - every published width is attained by an outcome the service could really return, checked by
    # handing that outcome to truth_b and re-deriving the interval from scratch.
    todo = [(q["id"], q["from"], q["to"], list(q["recover"])) for q in queries["widths"]]
    for q in queries["plans"]:
        todo.append((q["id"], q["from"], q["to"], plans_meta[q["id"]]["_sets"][0]))
    att = []
    for qid, a, c, S in todo:
        want = dd.width(a, c, set(S))
        wins = witness(dd, a, c, S)
        for rid, (p, p2) in wins.items():
            lo, hi = dd.box[rid]
            assert lo <= p <= p2 <= hi, ("witness window outside the box", qid, rid)
            assert p2 - p == dd.u(rid), ("witness window is the wrong width", qid, rid)
            assert G.grid_ok(p) and G.grid_ok(p2), ("witness window off the 1e-4 grid", qid, rid)
        lo, hi = inject(dd, wins).diff(a, c)
        att.append({"item": qid, "k": len(S), "width": G.fmt(want), "witness_width": G.fmt(hi - lo),
                    "attained": (hi - lo) == want})
        assert (hi - lo) == want, ("published width not attained", qid, float(want), float(hi - lo))
    cert["G3_attainment"] = {"items": len(att), "all_attained": all(x["attained"] for x in att),
                             "sample": att[:4]}

    # G4 - and no outcome the service could return does worse than the guarantee.
    rng = random.Random(23)
    checked, tight = 0, []
    for qid, a, c, S in todo:
        S = set(S)
        if not S:
            continue
        want = dd.width(a, c, S)
        best = ZERO
        for _ in range(N_SAMPLE):
            wins = {}
            for day in sorted(dd.T.free):
                here = sorted(r for r in dd.T.free[day] if r in S)
                if not here:
                    continue
                T = dd.T.resid[day] - sum((dd.box[r][0] for r in dd.T.free[day]), ZERO)
                rest = sum((dd.u(r) if r in S else dd.box[r][1] - dd.box[r][0]
                            for r in dd.T.free[day]), ZERO)
                cs = sum(((dd.box[r][1] - dd.box[r][0]) - dd.u(r) for r in here), ZERO)
                lo_s, hi_s = max(ZERO, T - rest), min(T, cs)
                sigma = lo_s + Fraction(rng.randint(0, int((hi_s - lo_s) * 10000)), 10000)
                order = here[:]
                rng.shuffle(order)
                left = sigma
                for r in order:
                    lo, hi = dd.box[r]
                    u = dd.u(r)
                    take = min((hi - lo) - u, left)
                    wins[r] = (lo + take, lo + take + u)
                    left -= take
                assert left == 0
            lo, hi = inject(dd, wins).diff(a, c)
            assert hi - lo <= want, ("an outcome the service could return beat the guarantee", qid,
                                     float(hi - lo), float(want))
            best = max(best, hi - lo)
            checked += 1
        tight.append(best == want)
    cert["G4_soundness_sampling"] = {"outcomes_checked": checked,
                                     "items_where_sampling_also_hit_the_max": sum(tight),
                                     "items": len(tight)}

    # G9 - the log pins exactly one reading, and it is the true one.
    seen = read_back(ctx, log_rows)
    surv = [n for n in sorted(RECOV_RULES)
            if all(observed(ctx, rid, n) == u for rid, u in seen.items())]
    cert["G9_identifiability"] = {"log_rows": len(log_rows), "candidate_readings": len(RECOV_RULES),
                                  "readings_consistent_with_the_log": surv,
                                  "outcome_mix": {k: sum(1 for r in log_rows if r["outcome"] == k)
                                                  for k in ("value", "range", "no_bundle")}}
    assert surv == sorted(expect or [TRUTH_RULE]), surv

    # G10 - and no shipped item can be answered by reading it off.
    named = set(q["run_id"] for q in queries["recovery"] + queries["values"])
    for q in queries["widths"]:
        named |= set(q["recover"])
    for q in queries["plans"]:
        named |= set(q["candidates"])
    overlap = sorted(named & set(seen))
    cert["G10_no_lookup"] = {"runs_named_by_items": len(named), "runs_in_the_log": len(seen),
                             "overlap": overlap}
    assert not overlap, overlap
    return cert


def coverage(b, ctx, log_rows, queries, plans_meta, need, killable, ans):
    """G5-G8 as in family D, plus the two gates that make this arm the arm it claims to be."""
    dd = b.ref
    ref = ans["reference"]
    wrong = {n: {k: [q["id"] for q in queries[k]
                     if differs(k, ref[k][q["id"]], ans[n][k].get(q["id"]),
                                pub_sets(k, q, plans_meta))]
                 for k in queries} for n in b.alt}
    cert = {"G5_decoys": {"min_fail": MIN_FAIL,
                          "still_needed": {n: v for n, v in need.items() if v > 0},
                          "items_each_reading_gets_wrong":
                              {n: sum(len(v) for v in wrong[n].values()) for n in sorted(wrong)},
                          "items_in_pool_each_reading_could_lose": killable}}
    assert not any(v > 0 for v in need.values()), need

    labels = sorted({ref["values"][q["id"]] for q in queries["values"]})
    yes_o, no_widest = [], []
    for q in queries["values"]:
        rid = q["run_id"]
        widest = max(dd.box[r][1] - dd.box[r][0] for r in dd.T.free[dd.T.day[rid]])
        if ref["values"][q["id"]] == "yes" and dd.group(q["from"], q["to"], rid) == "O":
            yes_o.append(q["id"])
        if ref["values"][q["id"]] == "no" and (dd.box[rid][1] - dd.box[rid][0]) == widest:
            no_widest.append(q["id"])
    kinds = sorted({("empty" if dead(dd, q["run_id"]) else
                     ("exact" if dd.u(q["run_id"]) == 0 else "band"))
                    for q in queries["recovery"]})
    cert["G6_coverage"] = {"value_labels": labels, "yes_for_a_run_in_neither_cell": yes_o,
                           "no_for_the_widest_box_on_its_day": no_widest,
                           "recovery_outcomes_graded": kinds}
    assert labels == ["no", "yes"], labels
    assert len(kinds) == 3, kinds
    assert no_widest, cert["G6_coverage"]

    search = [{"item": q["id"], "k": plans_meta[q["id"]]["_k"],
               "greedy_needs": plans_meta[q["id"]]["_greedy"],
               "unrecoverable_runs_offered": plans_meta[q["id"]]["_bait"],
               "optimal_sets": plans_meta[q["id"]]["_sets"]} for q in queries["plans"]]
    cert["G7_search"] = {"plans": search,
                         "with_unrecoverable_bait": [s["item"] for s in search
                                                     if s["unrecoverable_runs_offered"]],
                         "with_greedy_strictly_worse":
                             [s["item"] for s in search
                              if s["greedy_needs"] is None or s["greedy_needs"] > s["k"]]}
    assert cert["G7_search"]["with_unrecoverable_bait"], search

    seen = {}
    for q in queries["widths"] + queries["values"]:
        a, c = q["from"], q["to"]
        na, nb = dd.n(a), dd.n(c)
        for day in dd.relevant_days(a, c):
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
            seen.setdefault((lab, forced), []).append("%s@%s" % (day, q["id"]))
    cert["G8_regimes"] = {"%s%s" % (k[0], "_surplus_forced_down" if k[1] else ""): v[:3]
                          for k, v in sorted(seen.items())}
    assert len({k[0] for k in seen}) >= 2, sorted(seen)

    # G11 - losing the inference loses the score.
    naive = wrong["recovery__" + RIVAL]
    cert["G11_inversion_is_load_bearing"] = {
        "reading": RIVAL,
        "items_it_gets_wrong": {k: v for k, v in naive.items() if v},
        "unrecoverable_runs_named_by_items":
            sorted({q["run_id"] for q in queries["values"] + queries["recovery"] if dead(dd, q["run_id"])}),
        "banded_runs_named_by_items":
            sorted({q["run_id"] for q in queries["values"] + queries["recovery"]
                    if 0 < dd.u(q["run_id"]) < dd.box[q["run_id"]][1] - dd.box[q["run_id"]][0]})}
    for sec in ("recovery", "values", "widths"):
        assert len(naive[sec]) >= MIN_FAIL, (sec, naive)
    assert naive["plans"], naive

    # G12 - the resolutions are pinned by the log and are not constant across suites.
    used = sorted({ctx.suite[q["run_id"]] for q in queries["recovery"] + queries["values"]}
                  | {ctx.suite[r] for q in queries["widths"] for r in q["recover"]}
                  | {ctx.suite[r] for q in queries["plans"] for r in q["candidates"]})
    obs = read_back(ctx, log_rows)
    banded = {s: [rid for rid, u in obs.items()
                  if ctx.suite[rid] == s and ctx.why[rid] == "below_retention_floor"
                  and 0 < u < ctx.box[rid][1] - ctx.box[rid][0]]
              for s in used}
    cert["G12_resolutions"] = {"suites_items_depend_on": used,
                               "band_widths_shown_in_the_log":
                                   {s: sorted({float(obs[r]) for r in banded[s]}) for s in used},
                               "banded_rows_per_suite": {s: len(banded[s]) for s in used}}
    for s in used:
        assert len(banded[s]) >= 2, (s, cert["G12_resolutions"])
        assert cert["G12_resolutions"]["band_widths_shown_in_the_log"][s] == [float(BANDS[s])], s
    assert len({BANDS[s] for s in used}) == len(used), used
    return cert


def build_items(b):
    """Family D's pool, plus one item per unrecorded run asking what a recovery of it would leave.

    The recovery section is what makes the inferred semantics gradable on its own rather than only through
    its downstream effect on a width, which matters because a reading can be wrong about a run and still
    produce the right width for a contrast that run does not touch.
    """
    items, live = G.build_items(b)
    items["recovery"] = [{"id": None, "run_id": r} for r in sorted(b.ref.unknown)]
    return items, live


INSTRUCTION = """You are the on-call analyst for a model-evaluation fleet.  A six-week sweep and three
short follow-up campaigns have finished; the archive in `/app` is what survived of them.

Every planned run appears in `plan.csv` with its four configuration knobs (`suite`, `curriculum`,
`retrieval`, `decoder`).  `launch_log.csv` says which of them actually started.  Each launched run
produced a single number, its `pass_rate` on the suite it was assigned - but the archive lost some of
those numbers, in several different ways.  `incidents.md` is the on-call write-up of what happened,
`metric_card.md` says what the number means, `recovery_service.md` describes the cold-storage recovery
service, and `recovery_log.csv` is the record of every recovery request the on-call team issued during
the incident response.  Read all of them, and read the logs: how much each surviving record still
constrains a missing value is for you to work out.  Nothing in `/app` is a decoy: every file is exactly
what it claims to be, and no file is wrong.

## The quantity

Fix a configuration `c` (one value for each of the four knobs).  `L(c)` is the set of run ids that appear
in `launch_log.csv` *and* whose `plan.csv` row has exactly those four knob values; count each run id once,
however many rows mention it.  `N(c) = |L(c)|`, and `m(c)` is the plain average of the true `pass_rate` of
all `N(c)` of them, including the ones whose value the archive lost.

Call an assignment of a value to every launched run whose `pass_rate` the archive did not keep
*consistent* if it contradicts nothing in `/app`.  For a contrast `m(to) - m(from)`, its **sharp
interval** is the smallest closed interval containing its value under every consistent assignment.

## Recoveries

A recovery request names a set `S` of launched runs and is answered out of cold storage.  You must choose
`S` and pay for it before you see anything that comes back.

**This instruction does not tell you what a recovery request comes back with, and it is not the same for
every run.**  `recovery_service.md` says what kind of thing the service is; `recovery_log.csv` is a sample
of what it actually did, on runs that are not the ones you are asked about.  Working out, from those two
and from the rest of the archive, what a request for a given run would leave you knowing is the main part
of this task.  Whatever it is, it is true, it never contradicts `/app`, and it is a statement about the
runs in `S` only.

Call an **outcome** of a request for `S` one possible thing the service could come back with for every run
in `S` - one that contradicts nothing in `/app`.  Given an outcome, the sharp interval of a contrast is
recomputed with the outcome added to everything the archive already says, and it can only get narrower or
stay the same.  Which outcome you get is not yours to choose, so the only thing you can promise your
budget committee before paying is the **guaranteed width**:

    guaranteed_width(contrast, S)
        = the largest width the contrast's sharp interval can have, taken over every outcome that a
          request for S could have.

With `S` empty this is just the width of the sharp interval today.  Recovering a run is **worth taking**
for a contrast when adding it to an empty `S` makes the guaranteed width strictly smaller.

## What to report

For each item in `/app/queries.json`:

  * `contrasts`: the sharp interval for `m(to) - m(from)` today, as `lo` and `hi`.
  * `recovery`: for the single run `run_id`, the **residual width** a request naming just that run would
    leave: the width of the range that run's own `pass_rate` could still take afterwards, in the worst
    case over the outcomes of that request, counting only that run's own record and the answer itself and
    ignoring every other constraint in the archive.  Report `0` if such a request pins the value exactly,
    and the full width of the range that run's own record alone leaves it in if such a request adds
    nothing at all.
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
  "recovery":  {"r01": 0.0},
  "values":    {"v01": "yes"},
  "widths":    {"w01": 0.135294},
  "plans":     {"p01": {"k": 2, "runs": ["r01234", "r02345"]}}
}
```

Every id in `queries.json` must appear.  Numbers are graded to an absolute tolerance of %(tol)s, so give
at least six decimals; the answers are exact arithmetic, not estimates.  Scoring is all-or-nothing over
the %(n)d items.  Do not edit anything under `/app` except `answers.json`.
"""

VERIFY = '''"""Grade `d-design-sealed`.  Key only - no archive reading, no re-derivation."""
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
    for sec in ("contrasts", "recovery", "values", "widths", "plans"):
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
            elif sec in ("widths", "recovery"):
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

SOLVE = '''"""Reference solution for `d-design-sealed`.

The recovery semantics the agent has to infer are handed to this solver as `grain.json` - the residual
width each unrecorded run is left in by a request naming it.  Everything downstream is family D's
machinery with that extra argument filled in.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_d as D
from fractions import Fraction

APP = sys.argv[1] if len(sys.argv) > 1 else "/app"
grain = {k: Fraction(v) for k, v in
         json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grain.json"))).items()}

dd = D.Design(APP, None, None, grain)
q = json.load(open(os.path.join(APP, "queries.json")))
out = {"contrasts": {}, "recovery": {}, "values": {}, "widths": {}, "plans": {}}
for it in q["contrasts"]:
    lo, hi = dd.T.diff(it["from"], it["to"])
    out["contrasts"][it["id"]] = {"lo": float(lo), "hi": float(hi)}
for it in q["recovery"]:
    out["recovery"][it["id"]] = float(dd.u(it["run_id"]))
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


def main():
    w = WD.build(0)
    app = os.path.join(OUT, "environment", "app")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(app)
    G.emit(app, w)

    base = D.Design(app)
    ctx = Ctx(app, base)
    runs = sorted(base.unknown)
    b = BenchS(app, ctx, runs)
    print("pool: readings=%d unrecorded=%d" % (len(b.alt) + 1, len(runs)))

    items, live = build_items(b)
    plans = plan_items(b, live)
    assert plans, "no plan item survived the search"
    queries, need, killable, ans = select(b, items, plans)
    pub, meta = G.publish(queries)
    plans_meta = {q["id"]: dict(meta[q["id"]]) for q in queries["plans"]}
    for q in pub["plans"]:
        k, sets = b.ref.plan(q["from"], q["to"], q["candidates"], Fraction(str(q["target"])))
        assert k == plans_meta[q["id"]]["_k"], (q["id"], k, plans_meta[q["id"]]["_k"])
        assert sorted(sets) == sorted(plans_meta[q["id"]]["_sets"]), q["id"]

    hidden = {k: Fraction(str(v)) for k, v in w["hidden"].items()}
    avoid = {q["run_id"] for q in pub["recovery"] + pub["values"]}
    for q in pub["widths"]:
        avoid |= set(q["recover"])
    for q in pub["plans"]:
        avoid |= set(q["candidates"])
    log_rows = build_log(ctx, b.ref, hidden, random.Random(5), avoid)
    emit(app, w, log_rows)

    cert = certify(b, ctx, hidden, log_rows, pub, plans_meta)
    cert.update(coverage(b, ctx, log_rows, pub, plans_meta, need, killable, ans))
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
        INSTRUCTION % {"tol": TOL, "n": n_items} + H.SUFFIX_T.format(t=5400))
    H.w(os.path.join(OUT, "task.toml"),
        H.task_toml(artifacts=["answers.json"], family="d-design-sealed",
                    tags=["value-of-information", "partial-identification", "experiment-design",
                          "unspecified-input-semantics", "combinatorial-search", "ai4ai-eval"],
                    agent_timeout=5400, expert_hours=6.0, verifier_timeout=300))
    os.makedirs(os.path.join(OUT, "tests"))
    H.w(os.path.join(OUT, "tests", "verify_ds.py"), VERIFY)
    shutil.copy(os.path.join(ROOT, "core", "harbor.py"), os.path.join(OUT, "tests", "harbor.py"))
    json.dump(key, open(os.path.join(OUT, "tests", "key.json"), "w"), indent=1)
    H.w(os.path.join(OUT, "tests", "test.sh"), "#!/bin/bash\nset -eu\npython3 /tests/verify_ds.py\n",
        mode=0o755)
    H.w(os.path.join(OUT, "tests", "Dockerfile"),
        "FROM python:3.11-slim\nCOPY verify_ds.py harbor.py key.json test.sh /tests/\n")
    os.makedirs(os.path.join(OUT, "solution"))
    H.w(os.path.join(OUT, "solution", "ref_solve_ds.py"), SOLVE)
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
    print("G9", {k: v for k, v in cert["G9_identifiability"].items()})
    print("G10", cert["G10_no_lookup"])
    print("G11", cert["G11_inversion_is_load_bearing"]["items_it_gets_wrong"])
    print("G12", cert["G12_resolutions"])
    print("G6", cert["G6_coverage"])
    print("G7 bait:", cert["G7_search"]["with_unrecoverable_bait"],
          "greedy-beats:", cert["G7_search"]["with_greedy_strictly_worse"])
    print("G8 regimes:", sorted(cert["G8_regimes"]))
    for n, v in sorted(cert["G5_decoys"]["items_each_reading_gets_wrong"].items(), key=lambda x: x[1]):
        print("  %-52s wrong_on=%-3d could_lose_in_pool=%d" % (n, v, killable[n]))
    print("underkilled:", cert["G5_decoys"]["still_needed"])
    print("accepted:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
