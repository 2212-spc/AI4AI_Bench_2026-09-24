"""End-to-end self-check for family C: run the shipped solution inside a real /app, then probe impostors."""
import json, os, shutil, subprocess, sys, tempfile
TASK = sys.argv[1]
FAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fam_c")
sys.path.insert(0, FAM)


def bwrap(app, tests, argv, timeout=900):
    """Run argv with app bound at /app and tests at /tests, exactly the two paths harbor provides."""
    logs = tempfile.mkdtemp()
    bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
          "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
          "--bind", app, "/app", "--bind", logs, "/logs", "--tmpfs", "/home",
          "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/app",
          "--clearenv", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
          "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8"]
    bw += (["--ro-bind", tests, "/tests"] if tests else [])
    p = subprocess.run(bw + ["--"] + argv, capture_output=True, text=True, timeout=timeout)
    return p, logs


def grade(app, tests):
    p, logs = bwrap(app, tests, ["python3", "/tests/verify_c.py"])
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"reward": None, "stderr": p.stderr[-400:], "stdout": p.stdout[-400:]}


# 1. the shipped solution, executed exactly as harbor would: /app populated from environment/app
WORK = tempfile.mkdtemp()
APP = os.path.join(WORK, "app")
TESTS = os.path.join(WORK, "tests")
shutil.copytree(os.path.join(TASK, "environment/app"), APP)
shutil.copytree(os.path.join(TASK, "tests"), TESTS)
shutil.copy(os.path.join(TASK, "solution/ref_solve.py"), os.path.join(APP, "ref_solve.py"))
shutil.copy(os.path.join(TASK, "solution/solve.sh"), os.path.join(APP, "solve.sh"))
r, _ = bwrap(APP, None, ["bash", "/app/solve.sh"])
if r.returncode:
    print("solve.sh FAILED", (r.stderr or "")[-800:]); sys.exit(1)
g = grade(APP, TESTS)
print("shipped_solution", json.dumps({"reward": g["reward"], "detail": g.get("detail")})[:400], flush=True)
results = {"shipped_solution": g["reward"]}

# 2. impostors, graded against the same truth
import world, estimator as E                                            # noqa: E402
rows = [{k: (int(v) if v.strip().lstrip('-').isdigit() else float(v)) for k, v in r.items()
         if k != "run_id"} for r in __import__("csv").DictReader(open(os.path.join(APP, "log.csv")))]
qs = json.load(open(os.path.join(APP, "queries.json")))
reason = {"zero_stage": "aliased_with_other_knob", "offload": "aliased_with_other_knob",
          "opt_eps": "combination_never_observed", "warmup": "level_never_observed"}
probes = {"all_identified": {q["id"]: {"verdict": "identified", "delta": 0.0} for q in qs},
          "all_underdetermined": {q["id"]: {"verdict": "underdetermined",
                                            "reason": "aliased_with_other_knob"} for q in qs},
          "no_reasons": None, "no_audit_md": None, "empty": {}}
for n, o in list(E.CANDIDATES.items()) + list(E.ALTERNATIVES.items()):
    a = E.answer(rows, qs, o)
    for q in qs:
        if a[q["id"]]["verdict"] == "underdetermined":
            a[q["id"]]["reason"] = reason[q["knob"]]
    probes[n] = a
ref = E.answer(rows, qs, E.REF)
for q in qs:
    if ref[q["id"]]["verdict"] == "underdetermined":
        ref[q["id"]]["reason"] = reason[q["knob"]]
probes["no_reasons"] = {k: {kk: vv for kk, vv in v.items() if kk != "reason"} for k, v in ref.items()}
probes["no_audit_md"] = ref

for name, ans in probes.items():
    app = tempfile.mkdtemp()
    json.dump(ans, open(os.path.join(app, "answers.json"), "w"))
    if name != "no_audit_md":
        open(os.path.join(app, "audit.md"), "w").write("x")
    g = grade(app, TESTS)
    d = g.get("detail", {}) or {}
    results[name] = g["reward"]
    print("%-24s" % name, g["reward"], "n_correct=%s" % d.get("n_correct"),
          (d.get("verdict_or_estimate_errors") or d.get("reason_errors") or [{}])[0], flush=True)

want = {"shipped_solution": 1, "probe_as_numeric": 1}
bad = [k for k, v in results.items() if v != want.get(k, 0)]
print(json.dumps({"selfcheck_pass": not bad, "unexpected": bad}))
