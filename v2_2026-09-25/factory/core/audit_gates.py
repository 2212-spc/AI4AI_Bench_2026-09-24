"""Audit gates: an independent red team's findings, compiled into checks that run on every build.

A one-off audit improves one task.  A gate improves every task the factory will ever emit, including the
ones nobody thinks to audit.  So each finding from the red-team pass on family D is restated here as a
predicate over *shipped artefacts* - never over the generator's intentions - and run across the whole
corpus.  Three of the five are family-agnostic by construction, because they are driven by things every
task already ships: its `queries.json`, its answer key, and its reference solver.

  A1  threshold margin.   A question that publishes a numeric threshold is only well posed if the answer
      is stable against the arithmetic slop the task itself tolerates.  The check is a re-solve: shift
      every published threshold along a ladder of multiples of the task's own `tol` and re-run its
      reference solver.  The bar is 1x - it is incoherent for a task to accept a continuous answer that is
      within `tol` and then mark the discrete verdict derived from that same arithmetic wrong - and the
      wider rungs are reported as a headroom profile rather than a verdict.  The first version of this
      gate put the bar at 10x and condemned family B, whose `cannot_tell` band is *supposed* to sit near
      the margin; the bar has to come from the task's own contract, not from the auditor's taste.  Because
      each item's answer depends only on its own threshold, bulk perturbation localises the offender free.

  A2  cheap predicate.    An item is only hard if the hidden mechanism is needed to answer it.  The check
      builds every *raw observable* a grader-free agent could read off the environment - each column of
      each shipped table for the run an item names, membership in each file, the knob settings of the
      cells an item compares - and fails the section if any single one of them determines the whole label
      column at a cardinality no greater than the labels' own.  Labels that are constant fail too: a
      section with one answer is answered by guessing.

  A3  load-bearing mechanism.  Every parameter of the hidden mechanism must be doing work.  Where the
      reference solver takes the mechanism as an explicit artefact - family D hands its solver a
      `grain.json` of per-run residual widths - the gate flattens and rescales that artefact and requires
      at least one graded answer to move.  A parameter no answer depends on is decoration the agent can
      skip, and the task's difficulty claim silently shrinks to whatever is left.

  A4  wording ambiguity.  Where a sentence of the instruction admits two readings, both must be scored the
      same, or the item is grading reading comprehension of the author rather than the world.  The gate
      computes family D's `recovery` section under both readings of "the range the run is left in" - the
      run's own record alone, and the archive after its coupling constraints propagate - and fails if they
      disagree anywhere.

  A5  point-hypothesis coverage.  DIAGNOSTIC, NOT PASS/FAIL, and the reason is a result rather than an
      oversight.  The open arm's key is the worst case over an under-determined class.  Guaranteed widths
      are monotone in evidence - more information never widens a sharp interval - so the worst case over
      that class always coincides with the member that concedes the most, and an agent who simply *guesses*
      that member is indistinguishable from one who reasons over the whole class.  Making this pass/fail
      would therefore fail an arm for a property no design with a monotone objective can avoid.  It is
      reported so the number is on the record, and so that a future arm with a non-monotone objective can
      be recognised by this number falling.

Usage: `python3 audit_gates.py [tasks_dir] [out.json]`.  Exit status is non-zero if any gate FAILs; gates
that do not apply to a task report `n/a` with the reason and do not affect the status.
"""
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAM_D = os.path.join(os.path.dirname(HERE), "fam_d")

# Query fields that are *thresholds the answer is compared against*, as opposed to inputs the answer is
# computed from.  Deliberately a whitelist: perturbing an input is a different experiment (that is A3).
THRESHOLD_FIELDS = {"target", "margin"}
# The ladder A1 walks, in multiples of the task's own tolerance.  1x is the bar; the wider rungs turn the
# gate from a verdict into a measurement of how much headroom each item actually has.
LADDER = (1, 2, 10)
DEFAULT_TOL = 5e-4


# --------------------------------------------------------------------------------------------------
# shared helpers


def load(task):
    q = os.path.join(task, "environment", "app", "queries.json")
    queries = json.load(open(q)) if os.path.exists(q) else None
    key = None
    for cand in ("key.json", "truth.json"):
        p = os.path.join(task, "tests", cand)
        if os.path.exists(p):
            key = json.load(open(p))
            break
    return queries, key


def tol_of(key):
    return float(key.get("tol", DEFAULT_TOL)) if isinstance(key, dict) else DEFAULT_TOL


def ref_solver(task):
    c = sorted(glob.glob(os.path.join(task, "solution", "ref_solve*.py")))
    return c[0] if c else None


def same_answer(want, got, tol):
    """Graded equality against a *key* entry, in the same reading the shipped verifiers use.

    The asymmetry is deliberate.  A plan item's key carries `sets` - every optimal set - while a solver
    hands back one `runs`; scoring is "the right size, and a set that is actually optimal".  Comparing the
    two dicts field by field would call every correct answer wrong, which is how a gate quietly turns into
    noise that gets switched off.
    """
    if isinstance(want, dict) and isinstance(got, dict) and "sets" in want:
        if want.get("k") != got.get("k"):
            return False
        runs = got.get("runs", got.get("sets"))
        runs = runs[0] if runs and isinstance(runs[0], list) else runs
        return sorted(runs or []) in [sorted(s) for s in want["sets"]]
    if isinstance(want, dict) and isinstance(got, dict):
        return all(same_answer(want.get(k), got.get(k), tol) for k in want)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(want, bool):
        return abs(float(want) - float(got)) <= tol
    return want == got


def answer_all(dd, queries):
    """The reference answerer, factored out: the five shapes family D grades."""
    from fractions import Fraction
    out = {}
    for sec in queries:
        out[sec] = {}
        for it in queries[sec]:
            i = it["id"]
            if sec == "contrasts":
                lo, hi = dd.T.diff(it["from"], it["to"])
                out[sec][i] = {"lo": float(lo), "hi": float(hi)}
            elif sec == "recovery":
                out[sec][i] = float(dd.u(it["run_id"]))
            elif sec == "values":
                out[sec][i] = "yes" if dd.voi(it["from"], it["to"], it["run_id"]) else "no"
            elif sec == "widths":
                out[sec][i] = float(dd.width(it["from"], it["to"], set(it["recover"])))
            elif sec == "plans":
                k, sets = dd.plan(it["from"], it["to"], it["candidates"], Fraction(str(it["target"])))
                out[sec][i] = {"k": k, "runs": sets[0] if sets else []}
    return out


# --------------------------------------------------------------------------------------------------
# A1 - threshold margin


def a1_threshold_margin(task):
    """Perturb every published threshold along a ladder and record when each answer first moves.

    The bar is the task's *own* declared tolerance, not a number this gate invents.  Every one of these
    tasks promises to accept a continuous answer that is within `tol`; it is incoherent for the same
    arithmetic to be accepted there and marked wrong on a discrete verdict derived from it.  So an answer
    that moves at `delta = tol` is a defect.  The wider rungs are reported as a robustness profile: they
    say how much headroom a correct-but-imprecise agent actually has, which is the quantity an author
    thinks he has controlled and usually has not.
    """
    queries, key = load(task)
    solver = ref_solver(task)
    if queries is None or key is None:
        return {"status": "n/a", "why": "task does not ship queries.json + an answer key"}
    if solver is None:
        return {"status": "n/a", "why": "task ships no importable reference solver to re-run"}
    if not isinstance(queries, dict):
        return {"status": "n/a", "why": "queries are a bare list with no named threshold fields"}
    found = {s: f for s, items in queries.items() for it in items[:1] if isinstance(it, dict)
             for f in THRESHOLD_FIELDS if f in it}
    if not found:
        return {"status": "n/a", "why": "no published numeric threshold in any query"}

    tol = tol_of(key)
    first_move, detail, stopped = {}, [], None
    for mult in LADDER:
        delta = mult * tol
        for sign in (+1, -1):
            tmp = tempfile.mkdtemp(prefix="a1_")
            app = os.path.join(tmp, "app")
            shutil.copytree(os.path.join(task, "environment", "app"), app)
            pert = json.loads(json.dumps(queries))
            for sec, field in found.items():
                for it in pert[sec]:
                    it[field] = round(float(it[field]) + sign * delta, 12)
            json.dump(pert, open(os.path.join(app, "queries.json"), "w"), indent=1)
            r = subprocess.run([sys.executable, solver, app], capture_output=True, text=True, timeout=900)
            ap = os.path.join(app, "answers.json")
            if r.returncode or not os.path.exists(ap):
                # A wide shift can make an item *infeasible* - no subset of the candidate list reaches the
                # lowered target - and the reference solver, which never had to answer that question, dies.
                # That says nothing about the rung the verdict is taken at, so the ladder stops here and
                # keeps what the narrower rungs established.  Only a failure at the first rung leaves the
                # gate with nothing to report.
                stopped = {"multiple_of_tol": mult, "sign": sign,
                           "error": (r.stderr.strip().splitlines() or ["?"])[-1][:120]}
                shutil.rmtree(tmp, ignore_errors=True)
                break
            got = json.load(open(ap))
            for sec in found:
                for qid, want in key[sec].items():
                    if same_answer(want, got.get(sec, {}).get(qid), tol):
                        continue
                    tag = sec + "/" + qid
                    if tag in first_move:
                        continue
                    first_move[tag] = mult
                    detail.append({"item": tag, "moves_at": "%+g" % (sign * delta),
                                   "multiple_of_tol": mult, "key": want,
                                   "under_perturbation": got.get(sec, {}).get(qid)})
            shutil.rmtree(tmp, ignore_errors=True)
        if stopped:
            break
    if stopped and stopped["multiple_of_tol"] <= LADDER[0]:
        return {"status": "n/a", "why": "reference solver failed at the first rung: " + stopped["error"]}
    n = sum(len(queries[s]) for s in found)
    bad = sorted(t for t, m in first_move.items() if m <= 1)
    reached = [m for m in LADDER if not stopped or m < stopped["multiple_of_tol"]]
    profile = {"moves_within_1x_tol": sorted(t for t, m in first_move.items() if m <= 1),
               "moves_within_2x_tol": sorted(t for t, m in first_move.items() if m <= 2),
               "moves_within_%dx_tol" % reached[-1]: sorted(first_move)}
    return {"status": "fail" if bad else "pass",
            "threshold_fields": found, "tolerance": tol, "ladder": [m * tol for m in reached],
            "ladder_stopped": stopped,
            "n_items_with_a_threshold": n, "margin_profile": profile,
            "n_items_that_never_move": n - len(first_move), "detail": detail[:8],
            "why": ("%d of %d thresholds sit within the task's own tolerance of a value at which the "
                    "answer changes, so an agent whose arithmetic is accurate enough to be graded correct "
                    "on the continuous part can still be graded wrong on the discrete part"
                    % (len(bad), n)) if bad else
                   "no answer moves when its own threshold is shifted by the tolerance the task promises "
                   "to accept; %d of %d survive a %dx shift as well"
                   % (n - len(first_move), n, reached[-1])}


# --------------------------------------------------------------------------------------------------
# A2 - cheap predicate


def _tables(app):
    """run id -> {file.column: value}, plus membership flags, for every CSV under the app."""
    rows, files = {}, []
    for p in sorted(glob.glob(app + "/**/*.csv", recursive=True)):
        rel = os.path.relpath(p, app)
        files.append(rel)
        try:
            rd = list(csv.DictReader(open(p, errors="replace")))
        except Exception:
            continue
        idc = next((c for c in (rd[0].keys() if rd else []) if c in ("run_id", "id", "run")), None)
        if not idc:
            continue
        for r in rd:
            d = rows.setdefault(r[idc], {})
            for c, v in r.items():
                d.setdefault("%s.%s" % (rel, c), v)
            d["in:" + rel] = True
    for d in rows.values():
        for rel in files:
            d.setdefault("in:" + rel, False)
    return rows


def _features(item, runrows):
    """Every raw observable about an item: its own published fields, and the shipped table rows for any
    run it names.  Deliberately *raw* - no joins, no arithmetic across tables.  The claim the gate tests
    is that the answer needs the hidden mechanism, and the honest null is what a reader can copy out."""
    f = {}
    for k, v in item.items():
        if k == "id":
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                f["q.%s.%s" % (k, kk)] = vv
        elif isinstance(v, list):
            f["n_" + k] = len(v)
            if len(v) == 1 and isinstance(v[0], str):
                for kk, vv in runrows.get(v[0], {}).items():
                    f["only_%s.%s" % (k, kk)] = vv
        else:
            f["q." + k] = v
    rid = item.get("run_id")
    if rid:
        for k, v in runrows.get(rid, {}).items():
            f["run." + k] = v
    if isinstance(item.get("from"), dict) and isinstance(item.get("to"), dict):
        ch = [k for k in item["from"] if item["from"][k] != item["to"].get(k)]
        f["changed_knob"] = ",".join(sorted(ch))
        f["n_changed_knobs"] = len(ch)
    return f


def _label(v):
    if isinstance(v, float):
        return round(v, 9)
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return v


def label_columns(answers):
    """One column per graded field.

    An answer is often a record - a verdict plus a number, a size plus the sets that reach it - and the
    lookup attack lands on one field at a time.  Collapsing the record to a single label would hide a
    verdict that is a one-bit function of a shipped column behind a co-reported float that is not.
    Fields whose values are lists are skipped: a set-valued answer has no lookup semantics here.
    """
    if all(isinstance(a, dict) for a in answers) and answers:
        fields = sorted(set.intersection(*[set(a) for a in answers]))
        return {f: [_label(a[f]) for a in answers] for f in fields
                if not any(isinstance(a[f], list) for a in answers)}
    return {"": [_label(a) for a in answers]}


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sections(queries, key):
    """(name, items, answers) for every graded section whose items this gate can locate."""
    out = []
    for sec, ans in (key or {}).items():
        if not isinstance(ans, dict) or not ans:
            continue
        if isinstance(queries, dict) and sec in queries:
            items = [it for it in queries[sec] if isinstance(it, dict) and it.get("id") in ans]
        elif isinstance(queries, list):
            items = [it for it in queries if isinstance(it, dict) and it.get("id") in ans]
        else:
            continue
        if items:
            out.append((sec, items, [ans[it["id"]] for it in items]))
    return out


def a2_cheap_predicate(task):
    queries, key = load(task)
    if queries is None or not isinstance(key, dict):
        return {"status": "n/a", "why": "task does not ship queries + an id-keyed answer key"}
    secs = _sections(queries, key)
    if not secs:
        return {"status": "n/a", "why": "no graded section whose items can be matched to a query by id"}
    runrows = _tables(os.path.join(task, "environment", "app"))
    out, bad = {}, []
    for sec, items, answers in secs:
        feats = [_features(it, runrows) for it in items]
        names = sorted({k for f in feats for k in f})
        for field, labs in sorted(label_columns(answers).items()):
            tag = sec + ("." + field if field else "")
            nlab = len(set(labs))
            if nlab > 6:
                out[tag] = {"status": "n/a", "n_items": len(items), "n_labels": nlab,
                            "why": "effectively continuous (%d distinct over %d items); a "
                                   "low-cardinality lookup is not the exposure here" % (nlab, len(items))}
                continue
            if len(items) < 6:
                out[tag] = {"status": "n/a", "n_items": len(items), "n_labels": nlab,
                            "why": "only %d items: on so short a column almost any feature is injective "
                                   "and the test has no power either way" % len(items)}
                continue
            if not names:
                out[tag] = {"status": "n/a", "n_items": len(items), "n_labels": nlab,
                            "n_observable_features": 0,
                            "why": "no raw observable could be extracted for these items, so a PASS here "
                                   "would only record that the gate looked at nothing"}
                continue
            share = max(labs.count(l) for l in set(labs)) / float(len(labs))
            cheap = []
            if nlab < 2:
                cheap.append({"feature": "(none needed)", "kind": "constant label",
                              "detail": "every item has the same answer"})
            for nm in names:
                col = [f.get(nm) for f in feats]
                dv = {str(v) for v in col}
                # a feature only counts as a cheap solver if it *compresses*: no more distinct values than
                # the labels have, and strictly fewer than there are items.  An injective feature - a run
                # id, a timestamp - reproduces any column at all and says nothing about the section.
                if 2 <= len(dv) <= nlab < len(items) and len(dv) < len(items) \
                        and len({(str(v), l) for v, l in zip(col, labs)}) == len(dv):
                    cheap.append({"feature": nm, "kind": "%d-valued lookup" % len(dv),
                                  "detail": {str(v): l for v, l in zip(col, labs)}})
                    continue
                if nlab == 2:
                    nums = [_as_float(v) for v in col]
                    if all(n is not None for n in nums) and len(set(nums)) > 1:
                        for cut in sorted(set(nums))[:-1]:
                            if len({(n <= cut, l) for n, l in zip(nums, labs)}) == 2:
                                cheap.append({"feature": nm, "kind": "one-bit threshold",
                                              "detail": "%s <= %g reproduces the column" % (nm, cut)})
                                break
            out[tag] = {"status": "fail" if cheap else "pass", "n_items": len(items), "n_labels": nlab,
                        "majority_share": round(share, 3), "n_observable_features": len(names),
                        "cheap_predicates": cheap[:4], "n_cheap": len(cheap)}
            if cheap:
                bad.append(tag)
    st = "fail" if bad else ("pass" if any(v["status"] == "pass" for v in out.values()) else "n/a")
    return {"status": st, "columns": out, "failing_columns": bad,
            "n_columns_tested": sum(1 for v in out.values() if v["status"] != "n/a"),
            "why": ("a single raw observable reproduces the whole label column in %s, so the hidden "
                    "mechanism is not on the path to that answer" % ", ".join(bad)) if bad else
                   "no single raw observable determines any graded column"}


# --------------------------------------------------------------------------------------------------
# A3 - load-bearing mechanism


def a3_mechanism_load_bearing(task):
    """Perturb the mechanism the reference solver is handed, and require an answer to move."""
    from fractions import Fraction
    queries, key = load(task)
    solver = ref_solver(task)
    gp = os.path.join(task, "solution", "grain.json")
    if solver is None or not os.path.exists(gp):
        return {"status": "n/a", "why": "the reference solver takes no explicit mechanism artefact this "
                                        "gate knows how to perturb"}
    grain = json.load(open(gp))
    live = {r: v for r, v in grain.items() if Fraction(v) > 0}
    if not live:
        return {"status": "n/a", "why": "the mechanism has no positive-width parameter to flatten"}
    tol = tol_of(key)
    variants = {}
    common = sorted({str(v) for v in live.values()})
    variants["flattened"] = {r: (common[0] if Fraction(v) > 0 else v) for r, v in grain.items()}
    variants["rescaled_1.2x"] = {r: str(Fraction(v) * Fraction(6, 5)) for r, v in grain.items()}

    rows, dead = {}, []
    for name, g in variants.items():
        tmp = tempfile.mkdtemp(prefix="a3_")
        sol = os.path.join(tmp, "solution")
        shutil.copytree(os.path.join(task, "solution"), sol)
        app = os.path.join(tmp, "app")
        shutil.copytree(os.path.join(task, "environment", "app"), app)
        json.dump(g, open(os.path.join(sol, "grain.json"), "w"))
        r = subprocess.run([sys.executable, os.path.join(sol, os.path.basename(solver)), app],
                           capture_output=True, text=True, timeout=900)
        ap = os.path.join(app, "answers.json")
        if r.returncode or not os.path.exists(ap):
            rows[name] = {"error": (r.stderr.strip().splitlines() or ["?"])[-1][:120]}
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        got = json.load(open(ap))
        moved = [(s, q) for s in got if isinstance(key.get(s), dict)
                 for q in key[s] if not same_answer(key[s][q], got[s].get(q), tol)]
        rows[name] = {"n_answers_that_moved": len(moved), "sections":
                      sorted({s for s, _ in moved}), "examples": [s + "/" + q for s, q in moved[:6]]}
        if not moved:
            dead.append(name)
        shutil.rmtree(tmp, ignore_errors=True)
    n_distinct = len({str(v) for v in live.values()})
    return {"status": "fail" if dead else "pass", "variants": rows,
            "n_positive_width_runs": len(live), "n_distinct_widths": n_distinct,
            "inert_variants": dead,
            "why": ("the mechanism survives %s with every graded answer unchanged, so that parameter is "
                    "decoration" % ", ".join(dead)) if dead else
                   "both a flattened and a rescaled mechanism change graded answers, so the parameter is "
                   "load-bearing"}


# --------------------------------------------------------------------------------------------------
# A4 - wording ambiguity


#   Reading -> the phrases an instruction may use to name it.  The gate owns this vocabulary on purpose:
#   an author who has to write one of these sentences has to decide which one he means, and a gate that
#   sniffed for "clear wording" in general would pass anything confident enough.
READINGS = {
    "own_record": ["that run's own record alone", "its own record alone", "that run's record alone"],
    "archive_sharp": ["the archive alone", "the archive leaves that run in",
                      "after every constraint in the archive"],
}


def a4_wording_ambiguity(task):
    """Family D's `recovery` section under both readings of "the range the run is left in".

    Two readings are available to any careful reader: the run's own record, and the archive after its
    day-total constraints have propagated into that run.  They differ - the second is strictly narrower
    wherever a day total binds - so the sentence has to say which.  The gate therefore does not ask
    whether the readings differ (they are allowed to); it asks whether the instruction names the one the
    key follows and avoids naming the other.
    """
    from fractions import Fraction
    queries, key = load(task)
    gp = os.path.join(task, "solution", "grain.json")
    if not isinstance(queries, dict) or "recovery" not in (queries or {}) or not os.path.exists(gp):
        return {"status": "n/a", "why": "no section whose wording this gate knows two readings of"}
    sys.path.insert(0, FAM_D)
    import design_d as D                                                       # noqa: E402
    grain = {k: Fraction(v) for k, v in json.load(open(gp)).items()}
    app = os.path.join(task, "environment", "app")
    dd = D.Design(app, None, None, grain)
    tol = tol_of(key)
    rows, split, orphan = [], [], []
    for it in queries["recovery"]:
        rid = it["run_id"]
        g = grain.get(rid, Fraction(0))
        own = dd.box[rid][1] - dd.box[rid][0]                 # reading A: the run's own record alone
        lo, hi = dd.T.one(rid)                                # reading B: after the archive's couplings
        arch = hi - lo
        a, b = (own if g > own else g), (arch if g > arch else g)
        rows.append({"item": it["id"], "run": rid, "own_record": float(a), "archive_sharp": float(b)})
        want = key["recovery"][it["id"]]
        if abs(float(a) - float(b)) > tol:
            split.append(rows[-1])
        if not same_answer(want, float(a), tol) and not same_answer(want, float(b), tol):
            orphan.append({"item": it["id"], "key": want, "note": "key follows neither reading"})

    follows = [r for r in READINGS
               if all(same_answer(key["recovery"][x["item"]], x[r], tol) for x in rows)]
    src = open(os.path.join(task, "instruction.md")).read()
    names = {r: [p for p in ph if p in src] for r, ph in READINGS.items()}
    named = [r for r, hit in names.items() if hit]

    bad = None
    if orphan:
        bad = "the key follows neither reading on %d items" % len(orphan)
    elif not split:
        pass                                                  # the wording cannot change a graded answer
    elif len(follows) != 1:
        bad = "the key cannot be attributed to a single reading"
    elif named != follows:
        bad = ("the instruction names %s while the key follows %s"
               % (named or ["no reading"], follows))
    return {"status": "fail" if bad else "pass", "n_items": len(rows),
            "n_items_the_two_readings_disagree_on": len(split), "disagreements": split[:6],
            "key_follows": follows, "instruction_names": names, "orphans": orphan[:4],
            "readings": rows[:4],
            "why": bad + "; a reader who takes the other reading loses those items for a reason that has "
                         "nothing to do with the capability under test" if bad else
                   ("both readings agree on every graded item, so the wording cannot decide the score"
                    if not split else
                    "%d items turn on the wording, and the instruction names exactly the reading the key "
                    "follows (%s)" % (len(split), follows[0]))}


# --------------------------------------------------------------------------------------------------
# A5 - point-hypothesis coverage (diagnostic)


def a5_point_hypothesis(task):
    """How much of the key a *guess* at one member of the under-determined class already reproduces.

    The arm's claim is that the agent must reason over the whole class of readings the evidence leaves
    standing.  The cheap alternative is to pick one and commit.  This measures how well that pays: each
    surviving reading is turned into a mechanism, the reference answerer is run under it, and the result
    is scored against the shipped key.
    """
    from fractions import Fraction
    queries, key = load(task)
    cert_p = os.path.join(task, "authoring", "certificate.json")
    if not os.path.exists(cert_p) or not isinstance(queries, dict):
        return {"status": "n/a", "why": "no certificate naming the readings the evidence leaves standing"}
    cert = json.load(open(cert_p))
    surv = (cert.get("gates", {}).get("G9_identifiability", {})
                .get("readings_consistent_with_the_log", []))
    if len(surv) < 2:
        return {"status": "n/a", "n_surviving_readings": len(surv),
                "why": "the evidence pins the mechanism to one reading, so there is no class to guess in"}

    sys.path.insert(0, FAM_D)
    argv = sys.argv                      # the generators read their output directory off argv at import
    sys.argv = [argv[0], task]
    try:
        import design_d as D                                                   # noqa: E402
        import gen_ds as GS                                                    # noqa: E402
        try:
            import gen_do as GD                                                # noqa: E402  (registers the
            _ = GD.COMPLETIONS                                 # completions the open arm adds to the pool)
        except Exception:
            pass
    finally:
        sys.argv = argv
    app = os.path.join(task, "environment", "app")
    grain = {k: Fraction(v) for k, v in
             json.load(open(os.path.join(task, "solution", "grain.json"))).items()}
    ctx = GS.Ctx(app, D.Design(app))
    tol = tol_of(key)
    runs = sorted(grain)
    rows, best = {}, 0
    for name in surv:
        if name not in GS.RECOV_RULES:
            rows[name] = {"error": "reading not in the candidate pool"}
            continue
        dd = D.Design(app, None, None, GS.grain_of(ctx, name, runs))
        got = answer_all(dd, queries)
        per = {s: sum(1 for q in key[s] if same_answer(key[s][q], got[s].get(q), tol))
               for s in queries if isinstance(key.get(s), dict)}
        tot, n = sum(per.values()), sum(len(key[s]) for s in per)
        rows[name] = {"score": tot, "n_items": n, "by_section": per,
                      "is_the_key": tot == n}
        best = max(best, tot)
    n = sum(len(key[s]) for s in queries if isinstance(key.get(s), dict))
    scored = [v["score"] for v in rows.values() if "score" in v]
    exp = round(sum(scored) / float(len(scored)), 2) if scored else None
    return {"status": "diagnostic", "n_surviving_readings": len(surv),
            "surviving_readings": surv, "score_of_each_if_guessed": rows,
            "best_single_guess": best, "n_graded_items": n,
            "ceiling_for_a_guesser": round(best / float(n), 3),
            "expected_score_of_a_uniform_guess": exp,
            "expected_share_of_a_uniform_guess": round(exp / float(n), 3) if exp else None,
            "why": "guaranteed widths are monotone in evidence, so the worst case over the class always "
                   "coincides with the member that concedes the most; a guess at that member is "
                   "indistinguishable from worst-case reasoning, and the ceiling is therefore 1.0 by "
                   "construction, not by oversight.  What carries information is the spread: a uniform "
                   "guess over the %d surviving readings scores %s/%d on average.  A later arm with a "
                   "non-monotone objective is recognisable by the ceiling falling below 1.0."
                   % (len(surv), exp, n)}


# --------------------------------------------------------------------------------------------------


GATES = [("A1_threshold_margin", a1_threshold_margin),
         ("A2_cheap_predicate", a2_cheap_predicate),
         ("A3_mechanism_load_bearing", a3_mechanism_load_bearing),
         ("A4_wording_ambiguity", a4_wording_ambiguity),
         ("A5_point_hypothesis", a5_point_hypothesis)]


def main():
    tasks_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks"
    only = set(sys.argv[3:])
    rows, ok = {}, True
    for t in sorted(glob.glob(tasks_dir + "/*/")):
        t = t.rstrip("/")
        name = os.path.basename(t)
        if only and name not in only:
            continue
        r = {}
        for gname, fn in GATES:
            try:
                r[gname] = fn(t)
            except Exception as e:
                r[gname] = {"status": "error", "why": "%s: %s" % (type(e).__name__, e)}
            st = r[gname]["status"]
            ok &= st not in ("fail", "error")
            print("%-18s %-26s %-10s %s" % (name, gname, st.upper(), r[gname].get("why", "")[:88]))
        rows[name] = r
    print("AUDIT GATES", "PASS" if ok else "FAIL")
    if len(sys.argv) > 2:
        os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
        prev = json.load(open(sys.argv[2])) if (only and os.path.exists(sys.argv[2])) else {"tasks": {}}
        prev["tasks"].update(rows)
        prev["all_pass"] = not any(g["status"] in ("fail", "error")
                                   for v in prev["tasks"].values() for g in v.values())
        json.dump(prev, open(sys.argv[2], "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
