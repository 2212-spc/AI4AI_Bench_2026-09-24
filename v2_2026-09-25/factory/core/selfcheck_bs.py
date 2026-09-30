"""Self-check for family B's sealed arm: the shipped verifier must reward the reference and nothing else.

Everything `selfcheck_b.py` probes still applies; what is added here are the failure modes the new sections
make possible.  Three of them are the point of the arm and are *not* perturbations of the reference but
competing analyses, produced by running the reference code under a different rule:

  * `supports_are_box_arithmetic` - every rule right except that a single run's interval is read off its
    own box, ignoring that its day has a published total.  For the run the total pins outright this turns a
    point into a wide interval;
  * `mechanism_codes_swapped` - the two undocumented codes read the other way round, which is the whole
    inference the arm exists to test;
  * `grid_off_by_a_decade` - the rounding step guessed as 0.001 rather than measured.

The `must_pass` probes check that the tolerance is a rounding allowance and not decoration, and that
`screen_code` is compared after stripping whitespace rather than byte for byte.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

TASK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/b-bounds-sealed"


def run(app):
    logs = tempfile.mkdtemp()
    env = dict(os.environ, APP=app, TESTS=TASK + "/tests", REWARD_DIR=logs)
    p = subprocess.run([sys.executable, TASK + "/tests/verify_bs.py"], env=env,
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
    subprocess.run([sys.executable, TASK + "/solution/ref_solve_bs.py", work + "/app"],
                   capture_output=True, text=True, check=True, timeout=1800)
    ref = json.load(open(work + "/app/answers.json"))
    key = json.load(open(TASK + "/tests/key.json"))
    tol = key["tol"]
    cid, kid, sid = sorted(key["cells"]), sorted(key["contrasts"]), sorted(key["supports"])
    did = sorted(key["decisions"])

    out = {"reference": run(work + "/app")}
    imp, must = {}, {}

    a = copy.deepcopy(ref)
    for s in ("cells", "contrasts", "supports"):
        for q in a[s]:
            a[s][q]["lo"] = max(0.0, a[s][q]["lo"] - 0.01)
            a[s][q]["hi"] = min(1.0, a[s][q]["hi"] + 0.01)
    imp["valid_but_not_sharp_every_interval_widened"] = a

    a = copy.deepcopy(ref)
    for s in ("cells", "contrasts", "supports"):
        for q in a[s]:
            m = 0.5 * (a[s][q]["lo"] + a[s][q]["hi"])
            a[s][q] = {"lo": m, "hi": m}
    imp["midpoint_reported_as_a_point"] = a

    a = copy.deepcopy(ref)
    a["cells"][cid[0]]["hi"] += tol + 2e-5
    imp["one_endpoint_just_outside_tolerance"] = a

    a = copy.deepcopy(ref)
    # perturb an endpoint that has room to move: a support whose upper end is already 1.0 cannot be
    # nudged upward without leaving the metric's range, which the verifier rejects for a different reason.
    interior = next(q for q in sid if key["supports"][q]["hi"] < 1.0 - 1e-6)
    a["supports"][interior]["hi"] += tol + 2e-5
    imp["one_support_endpoint_just_outside_tolerance"] = a

    a = copy.deepcopy(ref)
    for q in a["supports"]:
        a["supports"][q] = {"lo": 0.0, "hi": 1.0}
    imp["trivial_support_for_every_run"] = a

    a = copy.deepcopy(ref)
    a["supports"].pop(sid[0])
    imp["one_support_item_omitted"] = a

    a = copy.deepcopy(ref)
    a.pop("supports")
    imp["support_section_missing"] = a

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
    a["supports"][sid[0]] = {"lo": str(a["supports"][sid[0]]["lo"]),
                             "hi": str(a["supports"][sid[0]]["hi"])}
    imp["endpoints_as_strings"] = a

    a = copy.deepcopy(ref)
    a["bounds"] = a.pop("cells")
    imp["section_renamed"] = a

    a = copy.deepcopy(ref)
    a.pop("mechanism")
    imp["mechanism_section_missing"] = a

    a = copy.deepcopy(ref)
    a["mechanism"] = {"grid": ref["mechanism"]["grid"]}
    imp["mechanism_code_omitted"] = a

    a = copy.deepcopy(ref)
    a["mechanism"]["screen_code"] = "RSN-99"
    imp["mechanism_code_not_in_the_log"] = a

    a = copy.deepcopy(ref)
    a["mechanism"]["grid"] = 0.001
    imp["grid_off_by_a_decade"] = a

    a = copy.deepcopy(ref)
    a["mechanism"]["grid"] = "0.0001"
    imp["grid_as_a_string"] = a

    # ---- competing analyses, produced by the reference code under a different rule -------------------
    sys.path.insert(0, TASK + "/solution")
    import bounds_b as BB                                                       # noqa: E402
    queries = json.load(open(TASK + "/environment/app/queries.json"))
    arch = BB.Arch(TASK + "/environment/app")

    boxes = copy.deepcopy(ref)
    ctx = BB.Ctx(arch, BB.REF)
    for q in queries["supports"]:
        lo, hi = ctx.sup[q["run_id"]]
        boxes["supports"][q["id"]] = {"lo": round(lo, 6), "hi": round(hi, 6)}
    imp["supports_are_box_arithmetic"] = boxes

    codes = sorted({r["reason"] for r in arch.retention})
    other = [c for c in codes if c != ref["mechanism"]["screen_code"]]
    a = copy.deepcopy(ref)
    a["mechanism"]["screen_code"] = other[0]
    imp["mechanism_codes_swapped"] = a

    naive = BB.answer(arch, queries, BB.CANDIDATES["surviving_row_evidence_ignored"])
    a = {k: naive[k] for k in ("cells", "contrasts", "decisions", "supports")}
    a["mechanism"] = copy.deepcopy(ref["mechanism"])
    imp["survivorship_ignored_everything_else_right"] = a

    # ---- probes that must still score 1 -------------------------------------------------------------
    a = copy.deepcopy(ref)
    for s in ("cells", "contrasts", "supports"):
        for q in a[s]:
            a[s][q] = {k: round(v, 4) for k, v in a[s][q].items()}
    must["rounded_to_four_decimals"] = a

    a = copy.deepcopy(ref)
    a["cells"][cid[0]]["lo"] -= 0.8 * tol
    a["supports"][sid[0]]["hi"] = min(1.0, a["supports"][sid[0]]["hi"] + 0.8 * tol)
    must["perturbed_inside_the_published_tolerance"] = a

    a = copy.deepcopy(ref)
    a["mechanism"]["screen_code"] = " %s " % ref["mechanism"]["screen_code"]
    must["code_with_surrounding_whitespace"] = a

    a = copy.deepcopy(ref)
    a["mechanism"]["grid"] = 1e-4
    must["grid_in_exponent_notation"] = a

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
                                                   json.dumps(v.get("detail", {}).get("failed", []))[:88]))
    print("SELFCHECK", "PASS" if ok else "FAIL", "probes=%d" % len(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
