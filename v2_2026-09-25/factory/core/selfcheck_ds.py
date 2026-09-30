"""Self-check for `d-design-sealed`: the shipped verifier rewards the reference and nothing else.

This arm grades five shapes of answer, and the one that is new - `recovery` - is the one an agent reaches
by *inference* rather than by arithmetic, so the probe set has to be able to tell a wrong inference from a
wrong calculation.  Four groups:

  * *near misses* - the reference nudged past the tolerance in each section, a plan answered with a set of
    the right size that is not optimal, a plan naming a run that was never a candidate, a recovery width
    reported as a string or omitted.  These are the ones a careless verifier would accept;
  * *competing readings of the service* - the whole answer re-derived under every rule in
    `gen_ds.RECOV_RULES`.  These are exactly what an agent that reads `recovery_log.csv` and draws the
    wrong conclusion would hand in, including the `d-design-a` reading in which every request returns an
    exact value, and every one of them must score 0;
  * *competing analyses* - the whole answer re-derived under every rule in `design_d.RULES`, with the
    recovery semantics correct.  An agent can get the inference right and still reason wrongly about how a
    recovered range moves the day's surplus, and that has to score 0 too;
  * *must-pass probes* - the reference rounded, perturbed inside the published tolerance, with plan runs
    reordered, labels in a different case, and the exact recoveries written as integers.  If any of these
    fails, the task is punishing presentation rather than reasoning.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

TASK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/d-design-sealed"
FAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fam_d")
sys.path.insert(0, FAM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_d as D                                                          # noqa: E402
import gen_ds as GS                                                           # noqa: E402

APP = os.path.join(TASK, "environment", "app")


def run(app):
    logs = tempfile.mkdtemp()
    env = dict(os.environ, APP=app, TESTS=TASK + "/tests", REWARD_DIR=logs)
    p = subprocess.run([sys.executable, TASK + "/tests/verify_ds.py"], env=env,
                       capture_output=True, text=True, timeout=900)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"reward": None, "stderr": p.stderr[-300:]}


def put(obj):
    d = tempfile.mkdtemp()
    if isinstance(obj, str):
        open(d + "/answers.json", "w").write(obj)
    else:
        json.dump(obj, open(d + "/answers.json", "w"))
    return d


def answer_under(queries, rule=None, grain=None):
    dd = D.Design(APP, rule, None, grain)
    o = {"contrasts": {}, "recovery": {}, "values": {}, "widths": {}, "plans": {}}
    for q in queries["contrasts"]:
        lo, hi = dd.T.diff(q["from"], q["to"])
        o["contrasts"][q["id"]] = {"lo": float(lo), "hi": float(hi)}
    for q in queries["recovery"]:
        o["recovery"][q["id"]] = float(dd.u(q["run_id"]))
    for q in queries["values"]:
        o["values"][q["id"]] = "yes" if dd.voi(q["from"], q["to"], q["run_id"]) else "no"
    for q in queries["widths"]:
        o["widths"][q["id"]] = float(dd.width(q["from"], q["to"], set(q["recover"])))
    for q in queries["plans"]:
        k, sets = dd.plan(q["from"], q["to"], q["candidates"], Fraction(str(q["target"])))
        o["plans"][q["id"]] = {"k": k, "runs": sets[0] if sets else []}
    return o


def main():
    work = tempfile.mkdtemp()
    shutil.copytree(APP, work + "/app")
    subprocess.run([sys.executable, TASK + "/solution/ref_solve_ds.py", work + "/app"],
                   capture_output=True, text=True, check=True, timeout=1800)
    ref = json.load(open(work + "/app/answers.json"))
    key = json.load(open(TASK + "/tests/key.json"))
    queries = json.load(open(os.path.join(APP, "queries.json")))
    tol = key["tol"]
    kid, rid_, vid, wid, pid = (sorted(key["contrasts"]), sorted(key["recovery"]),
                                sorted(key["values"]), sorted(key["widths"]), sorted(key["plans"]))

    out = {"reference": run(work + "/app")}
    imp, must = {}, {}

    a = copy.deepcopy(ref)
    for q in a["contrasts"]:
        a["contrasts"][q]["lo"] -= 0.01
        a["contrasts"][q]["hi"] += 0.01
    for q in a["widths"]:
        a["widths"][q] += 0.01
    imp["valid_but_not_sharp_everything_widened"] = a

    a = copy.deepcopy(ref)
    a["contrasts"][kid[0]]["hi"] += tol + 2e-5
    imp["one_contrast_endpoint_just_outside_tolerance"] = a

    a = copy.deepcopy(ref)
    a["widths"][wid[0]] += tol + 2e-5
    imp["one_width_just_outside_tolerance"] = a

    a = copy.deepcopy(ref)
    a["recovery"][rid_[0]] += tol + 2e-5
    imp["one_recovery_just_outside_tolerance"] = a

    # Skip a point-identified contrast: swapping lo and hi there is not a mutation, so the probe would
    # demand the verifier reject the key itself.  Degenerate contrasts are covered by the widened probe.
    swap = next((q for q in kid if ref["contrasts"][q]["lo"] < ref["contrasts"][q]["hi"]), None)
    if swap is not None:
        a = copy.deepcopy(ref)
        a["contrasts"][swap] = {"lo": ref["contrasts"][swap]["hi"], "hi": ref["contrasts"][swap]["lo"]}
        imp["one_contrast_endpoints_swapped"] = a

    # --- recovery-specific: the shapes a wrong inference takes -----------------------------------------
    a = copy.deepcopy(ref)
    a["recovery"] = {q: 0.0 for q in a["recovery"]}
    imp["every_recovery_returns_an_exact_value"] = a

    a = copy.deepcopy(ref)
    band = next((q for q in rid_ if 0 < key["recovery"][q] < 0.5), None)
    if band is not None:
        a["recovery"][band] = 0.0
        imp["a_banded_recovery_reported_as_exact"] = a

    a = copy.deepcopy(ref)
    empt = next((q for q in rid_ if key["recovery"][q] >= 0.5), None)
    if empt is not None:
        a["recovery"][empt] = 0.0
        imp["an_empty_recovery_reported_as_exact"] = a

    a = copy.deepcopy(ref)
    if band is not None:
        a["recovery"][band] = float(max(key["recovery"].values()))
        imp["a_banded_recovery_reported_as_empty"] = a

    a = copy.deepcopy(ref)
    a["recovery"][rid_[0]] = str(a["recovery"][rid_[0]])
    imp["one_recovery_as_a_string"] = a

    a = copy.deepcopy(ref)
    a["recovery"].pop(rid_[0])
    imp["one_recovery_item_omitted"] = a

    for lab in ("yes", "no"):
        a = copy.deepcopy(ref)
        a["values"] = {q: lab for q in a["values"]}
        imp["every_value_is_%s" % lab] = a

    for src, dst in (("yes", "no"), ("no", "yes")):
        a = copy.deepcopy(ref)
        flip = next((q for q in vid if ref["values"][q] == src), None)
        if flip:
            a["values"][flip] = dst
            imp["one_%s_flipped_to_%s" % (src, dst)] = a

    a = copy.deepcopy(ref)
    a["values"][vid[0]] = "maybe"
    imp["value_outside_the_vocabulary"] = a

    # --- plan-specific near misses ----------------------------------------------------------------------
    cand = {q["id"]: q["candidates"] for q in queries["plans"]}
    p = pid[0]

    a = copy.deepcopy(ref)
    extra = next(r for r in cand[p] if r not in ref["plans"][p]["runs"])
    a["plans"][p] = {"k": ref["plans"][p]["k"] + 1, "runs": ref["plans"][p]["runs"] + [extra]}
    imp["one_plan_answered_with_a_larger_set"] = a

    a = copy.deepcopy(ref)
    ok = set(map(tuple, key["plans"][p]["sets"]))
    bad = next(s for s in [sorted(ref["plans"][p]["runs"][:-1] + [r]) for r in cand[p]]
               if tuple(sorted(s)) not in ok and len(set(s)) == key["plans"][p]["k"])
    a["plans"][p] = {"k": key["plans"][p]["k"], "runs": bad}
    imp["one_plan_right_size_wrong_set"] = a

    a = copy.deepcopy(ref)
    a["plans"][p] = {"k": key["plans"][p]["k"],
                     "runs": ["r99999"] * (key["plans"][p]["k"] - 1) + [ref["plans"][p]["runs"][0]]}
    imp["one_plan_names_runs_that_do_not_exist"] = a

    a = copy.deepcopy(ref)
    a["plans"][p] = {"k": key["plans"][p]["k"],
                     "runs": [ref["plans"][p]["runs"][0]] * key["plans"][p]["k"]}
    imp["one_plan_repeats_a_run_to_reach_k"] = a

    a = copy.deepcopy(ref)
    a["plans"][p] = {"k": key["plans"][p]["k"], "runs": ref["plans"][p]["runs"][:-1]}
    imp["one_plan_k_and_runs_disagree"] = a

    a = copy.deepcopy(ref)
    a["plans"][p]["k"] = str(a["plans"][p]["k"])
    imp["plan_k_as_a_string"] = a

    a = copy.deepcopy(ref)
    a["plans"][p] = ref["plans"][p]["runs"]
    imp["plan_answered_as_a_bare_list"] = a

    # --- shape ------------------------------------------------------------------------------------------
    a = copy.deepcopy(ref)
    a["widths"][wid[0]] = str(a["widths"][wid[0]])
    imp["one_width_as_a_string"] = a

    a = copy.deepcopy(ref)
    a["widths"].pop(wid[0])
    imp["one_width_item_omitted"] = a

    for sec in ("contrasts", "recovery", "values", "widths", "plans"):
        a = copy.deepcopy(ref)
        a.pop(sec)
        imp["%s_section_missing" % sec] = a

    a = copy.deepcopy(ref)
    a["value"] = a.pop("values")
    imp["section_renamed"] = a

    a = copy.deepcopy(ref)
    a["values"][vid[0]] = True
    imp["value_as_a_boolean"] = a

    # --- competing readings of the service, and competing analyses ---------------------------------------
    base = D.Design(APP)
    ctx = GS.Ctx(APP, base)
    runs = sorted(base.unknown)
    truth = GS.grain_of(ctx, GS.TRUTH_RULE, runs)
    for name in sorted(GS.RECOV_RULES):
        if name != GS.TRUTH_RULE:
            imp["reading__" + name] = answer_under(queries, None, GS.grain_of(ctx, name, runs))
    for name, rule in sorted(D.RULES.items()):
        imp["analysis__" + name] = answer_under(queries, rule, truth)

    # --- must pass ---------------------------------------------------------------------------------------
    a = copy.deepcopy(ref)
    for q in a["contrasts"]:
        a["contrasts"][q] = {k: round(v, 6) for k, v in a["contrasts"][q].items()}
    a["widths"] = {q: round(v, 6) for q, v in a["widths"].items()}
    a["recovery"] = {q: round(v, 6) for q, v in a["recovery"].items()}
    must["rounded_to_six_decimals"] = a

    a = copy.deepcopy(ref)
    a["contrasts"][kid[0]]["lo"] -= 0.8 * tol
    a["widths"][wid[0]] = max(0.0, a["widths"][wid[0]] - 0.8 * tol)
    a["recovery"][rid_[0]] = max(0.0, a["recovery"][rid_[0]] - 0.8 * tol)
    must["perturbed_inside_the_published_tolerance"] = a

    a = copy.deepcopy(ref)
    a["values"] = {q: (" %s " % v.upper()) for q, v in a["values"].items()}
    must["labels_upper_case_with_whitespace"] = a

    a = copy.deepcopy(ref)
    for q in a["plans"]:
        a["plans"][q]["runs"] = list(reversed(a["plans"][q]["runs"]))
    must["plan_runs_in_a_different_order"] = a

    a = copy.deepcopy(ref)
    a["recovery"] = {q: (int(v) if float(v).is_integer() else v) for q, v in a["recovery"].items()}
    must["integral_recoveries_written_as_integers"] = a

    a = copy.deepcopy(ref)
    zero = [q for q, v in ref["widths"].items() if v == 0.0]
    if zero:
        a["widths"][zero[0]] = 0e0
    a["widths"][wid[0]] = float("%.10e" % a["widths"][wid[0]])
    a["recovery"][rid_[0]] = float("%.10e" % a["recovery"][rid_[0]])
    must["numbers_in_exponent_notation"] = a

    for name, obj in imp.items():
        out[name] = run(put(obj))
    out["not_json"] = run(put("{not json"))
    out["empty_object"] = run(put({}))
    out["no_file"] = run(tempfile.mkdtemp())
    for name, obj in must.items():
        out["MUSTPASS_" + name] = run(put(obj))

    ok = out["reference"]["reward"] == 1
    for k, v in sorted(out.items()):
        want = 1 if (k == "reference" or k.startswith("MUSTPASS_")) else 0
        good = v["reward"] == want
        ok = ok and good
        print("%-3s %-62s reward=%s want=%d %s" % ("OK " if good else "BAD", k, v["reward"], want,
                                                   json.dumps(v.get("detail", {}).get("failed", []))[:70]))
    print("SELFCHECK", "PASS" if ok else "FAIL", "probes=%d" % len(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
