"""Self-check for family B: the shipped verifier must reward the reference and nothing else.

Two kinds of probe.  The impostors each break exactly one thing and must score 0 - including the two that
are *valid but not sharp* (every interval widened, the retention bound discarded), because in a partial
identification task looseness is the error being measured.  The `must_pass` probes perturb the reference
inside the published tolerance and must still score 1, which is what stops the tolerance from being
decorative: if rounding to four decimals failed, the task would be grading float noise.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/b-bounds-a"


def run(app):
    logs = tempfile.mkdtemp()
    env = dict(os.environ, APP=app, TESTS=TASK + "/tests", REWARD_DIR=logs)
    p = subprocess.run([sys.executable, TASK + "/tests/verify_b.py"], env=env,
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


def main():
    work = tempfile.mkdtemp()
    shutil.copytree(TASK + "/environment/app", work + "/app")
    subprocess.run([sys.executable, TASK + "/solution/ref_solve_b.py", work + "/app"],
                   capture_output=True, text=True, check=True, timeout=1800)
    ref = json.load(open(work + "/app/answers.json"))
    key = json.load(open(TASK + "/tests/key.json"))
    tol = key["tol"]
    cid = sorted(key["cells"])
    kid = sorted(key["contrasts"])
    did = sorted(key["decisions"])

    out = {"reference": run(work + "/app")}
    imp, must = {}, {}

    a = copy.deepcopy(ref)
    for s in ("cells", "contrasts"):
        for q in a[s]:
            a[s][q]["lo"] -= 0.01
            a[s][q]["hi"] += 0.01
    imp["valid_but_not_sharp_every_interval_widened"] = a

    a = copy.deepcopy(ref)
    for q in a["cells"]:
        a["cells"][q]["hi"] += 0.02
    imp["upper_endpoints_inflated"] = a

    a = copy.deepcopy(ref)
    for s in ("cells", "contrasts"):
        for q in a[s]:
            m = 0.5 * (a[s][q]["lo"] + a[s][q]["hi"])
            a[s][q] = {"lo": m, "hi": m}
    imp["midpoint_reported_as_a_point"] = a

    a = copy.deepcopy(ref)
    a["cells"][cid[0]]["hi"] += tol + 2e-5
    imp["one_endpoint_just_outside_tolerance"] = a

    # Skip a point-identified contrast: swapping lo and hi there is not a mutation, so the probe would
    # demand the verifier reject the key itself.  Degenerate contrasts are covered by the widened probe.
    swap = next((q for q in kid if ref["contrasts"][q]["lo"] < ref["contrasts"][q]["hi"]), None)
    if swap is not None:
        a = copy.deepcopy(ref)
        a["contrasts"][swap] = {"lo": a["contrasts"][swap]["hi"], "hi": a["contrasts"][swap]["lo"]}
        imp["one_contrast_endpoints_swapped"] = a

    a = copy.deepcopy(ref)
    for q in a["decisions"]:
        a["decisions"][q] = "cannot_tell"
    imp["every_decision_is_cannot_tell"] = a

    a = copy.deepcopy(ref)
    lab = {"yes": "no", "no": "yes", "cannot_tell": "yes"}
    a["decisions"][did[0]] = lab[ref["decisions"][did[0]]]
    imp["one_decision_flipped"] = a

    a = copy.deepcopy(ref)
    a["decisions"][did[0]] = "probably"
    imp["decision_outside_the_vocabulary"] = a

    a = copy.deepcopy(ref)
    a["cells"].pop(cid[0])
    imp["one_cell_item_omitted"] = a

    a = copy.deepcopy(ref)
    a["contrasts"].pop(kid[0])
    imp["one_contrast_item_omitted"] = a

    a = copy.deepcopy(ref)
    a["cells"][cid[0]] = {"lo": str(a["cells"][cid[0]]["lo"]), "hi": str(a["cells"][cid[0]]["hi"])}
    imp["endpoints_as_strings"] = a

    a = copy.deepcopy(ref)
    a["cells"][cid[0]] = [a["cells"][cid[0]]["lo"], a["cells"][cid[0]]["hi"]]
    imp["endpoints_as_a_list"] = a

    a = copy.deepcopy(ref)
    a["bounds"] = a.pop("cells")
    imp["section_renamed"] = a

    a = copy.deepcopy(ref)
    for q in a["cells"]:
        a["cells"][q] = {"lo": 0.0, "hi": 1.0}
    imp["trivial_support_for_every_cell"] = a

    # The one impostor that is not a perturbation of the reference but a competing *analysis*: every rule
    # applied correctly except that a row lost with its shard file is treated as unconstrained rather than
    # as a row that had already cleared its floor.  It is the answer two frontier agents produced against
    # the previous version of this world, and it has to score zero through the shipped verifier, not just
    # differ in the generator's decoy scan.
    sys.path.insert(0, TASK + "/solution")
    import bounds_b as BB                                                        # noqa: E402
    naive = BB.answer(BB.Arch(TASK + "/environment/app"), json.load(
        open(TASK + "/environment/app/queries.json")), BB.CANDIDATES["surviving_row_evidence_ignored"])
    imp["survivorship_ignored_everything_else_right"] = {k: naive[k] for k in
                                                         ("cells", "contrasts", "decisions")}

    # ---- probes that must still score 1 ----------------------------------------------------------
    a = copy.deepcopy(ref)
    for s in ("cells", "contrasts"):
        for q in a[s]:
            a[s][q] = {k: round(v, 4) for k, v in a[s][q].items()}
    must["rounded_to_four_decimals"] = a

    a = copy.deepcopy(ref)
    a["cells"][cid[0]]["lo"] -= 0.8 * tol
    a["contrasts"][kid[0]]["hi"] += 0.8 * tol
    must["perturbed_inside_the_published_tolerance"] = a

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
        print("%-3s %-52s reward=%s want=%d %s" % ("OK " if good else "BAD", k, v["reward"], want,
                                                   json.dumps(v.get("detail", {}).get("failed", []))[:90]))
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
