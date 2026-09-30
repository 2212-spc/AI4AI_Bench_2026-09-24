"""Self-check for family S: the exported task must reward the reference and nothing else.

Runs the shipped verifier against the reference submission and against a battery of impostors that each
break exactly one thing - a fabricated number where the data are silent, a right abstention with the wrong
reason code, the raw missing fraction reported as the guard fraction, a value just outside tolerance,
malformed or partial JSON.  Every impostor must score 0 and the reference must score 1.
"""
import copy, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/s-censored-a"


def run(app):
    logs = tempfile.mkdtemp()
    env = dict(os.environ, APP=app, TESTS=TASK + "/tests", REWARD_DIR=logs)
    p = subprocess.run([sys.executable, TASK + "/tests/verify_s.py"], env=env,
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
    subprocess.run([sys.executable, TASK + "/solution/ref_solve.py", work + "/app"],
                   capture_output=True, text=True, check=True, timeout=1800)
    ref = json.load(open(work + "/app/answers.json"))
    key = json.load(open(TASK + "/tests/key.json"))
    man = open(TASK + "/environment/app/manifest.csv").read().count("\n") - 1
    res = open(TASK + "/environment/app/results.csv").read().count("\n") - 1

    out = {"reference": run(work + "/app")}
    imp = {}
    und = [q for q, v in key["queries"].items() if v["verdict"] == "underdetermined"]
    ident = [q for q, v in key["queries"].items() if v["verdict"] == "identified"]

    a = copy.deepcopy(ref); a["queries"][und[0]] = {"verdict": "identified", "delta": 1.06}
    imp["fabricates_a_number_where_no_run_survived"] = a

    a = copy.deepcopy(ref)
    for q in und:
        a["queries"][q] = {"verdict": "underdetermined", "reason": "no_surviving_run"}
    imp["one_reason_code_for_every_abstention"] = a

    a = copy.deepcopy(ref)
    codes = [a["queries"][q]["reason"] for q in und]
    for q, c in zip(und, codes[1:] + codes[:1]):
        a["queries"][q]["reason"] = c
    imp["reason_codes_rotated"] = a

    a = copy.deepcopy(ref); a["queries"][und[0]]["reason"] = "the data do not determine this"
    imp["free_text_reason"] = a

    a = copy.deepcopy(ref)
    a["censoring"] = {r: 0.0 for r in key["censoring"]}
    imp["reports_no_censoring"] = a

    a = copy.deepcopy(ref)
    a["censoring"] = {r: min(1.0, key["censoring"][r]["value"] + key["censoring"][r]["tol"] + 0.02)
                      for r in key["censoring"]}
    imp["censoring_just_outside_tolerance"] = a

    a = copy.deepcopy(ref)
    q = max(ident, key=lambda x: key["queries"][x]["tol"])
    a["queries"][q]["delta"] = key["queries"][q]["delta"] + key["queries"][q]["tol"] + 0.002
    imp["one_delta_just_outside_tolerance"] = a

    a = copy.deepcopy(ref)
    for q in ident:
        a["queries"][q]["delta"] = -a["queries"][q]["delta"]
    imp["signs_flipped"] = a

    a = copy.deepcopy(ref); a["queries"].pop(ident[0])
    imp["one_query_omitted"] = a

    a = copy.deepcopy(ref); a["censoring"].pop(sorted(key["censoring"])[0])
    imp["one_report_cell_omitted"] = a

    a = copy.deepcopy(ref); a["queries"][ident[0]]["delta"] = str(a["queries"][ident[0]]["delta"])
    imp["delta_as_string"] = a

    a = copy.deepcopy(ref); a["censoring"][sorted(key["censoring"])[0]] = 1.4
    imp["fraction_out_of_range"] = a

    a = copy.deepcopy(ref)
    a["queries"] = {q: {"verdict": "identified", "delta": 0.0} for q in key["queries"]}
    imp["everything_identified_at_zero"] = a

    for name, obj in imp.items():
        out[name] = run(put(obj))
    out["not_json"] = run(put("{not json"))
    out["empty_object"] = run(put({}))
    out["no_file"] = run(tempfile.mkdtemp())

    print("rows: manifest=%d results=%d missing=%d" % (man, res, man - res))
    ok = out["reference"]["reward"] == 1
    for k, v in out.items():
        flag = "OK " if ((k == "reference") == (v["reward"] == 1)) else "BAD"
        if k != "reference" and v["reward"] != 0:
            ok = False
        print("%-3s %-44s reward=%s %s" % (flag, k, v["reward"],
                                           json.dumps(v.get("detail", {}).get("failed", []))[:110]))
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
