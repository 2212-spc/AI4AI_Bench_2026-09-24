"""Independent re-verification of the *shipped* copy, not the build tree.

Everything else in this repository was run against `/tmp/bench2`, the working directory the generators
write into.  That is the wrong thing to trust at delivery time: a task is only as good as the bytes that
were copied out, and the copy step is itself a place where a file can be missed.  This script therefore
takes `tasks/` and `evidence/` as they were delivered and asks two questions per task:

  1. *does the key still follow from the world?*  The shipped reference solver is run on a fresh copy of
     `environment/app` and its answer is graded by the shipped verifier.  Reward must be 1.  This catches a
     key that was regenerated without its environment, an environment that was copied without its key, and
     a reference solver that depends on a file that never left the build tree.
  2. *do the reported scores still follow from the archived answers?*  Every model answer kept under
     `evidence/<run>/` is graded again by the shipped verifier and compared with the number quoted in
     `evidence/grades.json`.  This is what makes the headline scores checkable by someone who has only this
     directory: they do not have to trust the grading log, they can re-run it.

Both questions are answered with the task's own `tests/verify_*.py`, so a verifier that has drifted out of
step with its key shows up as a failure here rather than as an unexplained score.

    python3 verify_shipped.py [tasks_dir] [evidence_dir] [out.json] [task ...]
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# How to produce a reference answer inside a fresh copy of the world, and which verifier grades it.
# `solver` is run with the app directory as argv[1]; `copy` files are placed in the app directory instead
# (family O's deliverable *is* a module, so there is nothing to compute).
RECIPES = {
    "b-bounds-a":      {"verify": "verify_b.py",  "solver": "ref_solve_b.py"},
    "b-bounds-sealed": {"verify": "verify_bs.py", "solver": "ref_solve_bs.py"},
    "c-audit-a":       {"verify": "verify_c.py",  "solver": "ref_solve.py"},
    "s-censored-a":    {"verify": "verify_s.py",  "solver": "ref_solve.py"},
    "d-design-a":      {"verify": "verify_d.py",  "solver": "ref_solve_d.py"},
    "d-design-sealed": {"verify": "verify_ds.py", "solver": "ref_solve_ds.py"},
    "d-design-open":   {"verify": "verify_ds.py", "solver": "ref_solve_ds.py"},
    "o-pack-a":        {"verify": "verify_o.py",  "copy": ["solution.py", "NOTES.md"]},
    "v-falsify-a":     {"verify": "verify_v.py",  "solver": "make_witnesses.py"},
}

# Which task each archived run answered.  Runs whose task was rebuilt afterwards are listed with the
# revision they answered so that a stale score is never silently re-attributed to the current items.
RUNS = {
    "cc_b3": "b-bounds-a", "gpt_b3": "b-bounds-a",
    "cc_bs": "b-bounds-sealed", "gpt_bs": "b-bounds-sealed",
    "cc_c-audit-a": "c-audit-a", "gpt_c-audit-a": "c-audit-a",
    "cc_s2": "s-censored-a", "gpt_s-censored-a": "s-censored-a",
    "cc_d1v2": "d-design-a", "gpt_d1v2": "d-design-a",
    "cc_dsv2": "d-design-sealed", "gpt_dsv2": "d-design-sealed",
    "cc_dov2": "d-design-open", "gpt_dov2": "d-design-open",
    "cc_o-pack-a": "o-pack-a", "gpt_o-pack-a": "o-pack-a",
    "cc_v-falsify-a": "v-falsify-a", "gpt_v-falsify-a": "v-falsify-a",
}
SUPERSEDED = {"cc_b-bounds-a", "cc_b2", "gpt_b2", "cc_s-censored-a",
              "cc_d1", "gpt_d1", "cc_ds", "gpt_ds", "cc_do", "gpt_do"}


def grade(task, app):
    """Run the task's shipped verifier over `app` and return its reward record."""
    logs = tempfile.mkdtemp()
    env = dict(os.environ, APP=app, TESTS=os.path.join(task, "tests"), REWARD_DIR=logs)
    v = os.path.join(task, "tests", RECIPES[os.path.basename(task)]["verify"])
    p = subprocess.run([sys.executable, v], env=env, capture_output=True, text=True, timeout=1800)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"reward": None, "stderr": p.stderr[-400:]}


def fresh(task):
    """A writable copy of the task's environment, with the solution directory beside it."""
    work = tempfile.mkdtemp()
    shutil.copytree(os.path.join(task, "environment", "app"), os.path.join(work, "app"))
    if os.path.isdir(os.path.join(task, "solution")):
        shutil.copytree(os.path.join(task, "solution"), os.path.join(work, "solution"),
                        ignore=shutil.ignore_patterns("__pycache__"))
    return work


def reference(task):
    """Re-derive the answer with the shipped reference solver and grade it."""
    name = os.path.basename(task)
    rec = RECIPES[name]
    work = fresh(task)
    app = os.path.join(work, "app")
    if rec.get("copy"):
        for f in rec["copy"]:
            src = os.path.join(work, "solution", f)
            if os.path.exists(src):
                shutil.copy(src, app)
    else:
        p = subprocess.run([sys.executable, os.path.join(work, "solution", rec["solver"]), app],
                           capture_output=True, text=True, timeout=1800)
        if p.returncode:
            return {"reward": None, "solver_failed": (p.stderr.strip().splitlines() or ["?"])[-1][:200]}
    return grade(task, app)


def replay(task, run_dir):
    """Grade an archived model answer by dropping it into a fresh copy of the world."""
    work = fresh(task)
    app = os.path.join(work, "app")
    for src in glob.glob(os.path.join(run_dir, "*")):
        dst = os.path.join(app, os.path.basename(src))
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, app)
    return grade(task, app)


def main():
    tasks_dir = sys.argv[1] if len(sys.argv) > 1 else "tasks"
    ev = sys.argv[2] if len(sys.argv) > 2 else "evidence"
    outp = sys.argv[3] if len(sys.argv) > 3 else None
    only = set(sys.argv[4:])
    quoted = {r["run"]: r for r in json.load(open(os.path.join(ev, "grades.json")))}

    out = {"reference": {}, "replay": {}}
    if outp and only and os.path.exists(outp):
        out = json.load(open(outp))
    ok = True
    for name in sorted(RECIPES):
        if only and name not in only:
            continue
        task = os.path.join(tasks_dir, name)
        r = reference(task)
        good = r.get("reward") == 1
        ok = ok and good
        out["reference"][name] = {"reward": r.get("reward"), "pass": good,
                                 "why": r.get("solver_failed") or json.dumps(
                                     r.get("detail", {}).get("failed", []))[:200]}
        print("%-4s reference  %-18s reward=%s %s" % ("OK" if good else "BAD", name, r.get("reward"),
                                                      out["reference"][name]["why"][:90]))
        for run, t in sorted(RUNS.items()):
            if t != name or not os.path.isdir(os.path.join(ev, run)):
                continue
            g = replay(task, os.path.join(ev, run))
            want = quoted.get(run, {}).get("reward")
            good = g.get("reward") == want
            ok = ok and good
            out["replay"][run] = {"task": name, "reward": g.get("reward"), "quoted": want,
                                  "pass": good, "n_failed": g.get("detail", {}).get("n_failed"),
                                  "failed": [f if isinstance(f, str) else f.get("item")
                                             for f in g.get("detail", {}).get("failed", [])][:8]}
            print("%-4s replay     %-18s %-16s reward=%s quoted=%s %s"
                  % ("OK" if good else "BAD", name, run, g.get("reward"), want,
                     json.dumps(out["replay"][run]["failed"])[:70]))
    for run in sorted(SUPERSEDED):
        if os.path.isdir(os.path.join(ev, run)):
            out["replay"].setdefault(run, {"pass": None, "note": "answered a superseded revision; "
                                           "kept for the record, not re-gradeable against the shipped key"})
    out["all_pass"] = ok and all(v["pass"] for v in out["reference"].values()) \
        and all(v.get("pass") is not False for v in out["replay"].values())
    if outp:
        json.dump(out, open(outp, "w"), indent=1)
    print("SHIPPED VERIFICATION", "PASS" if out["all_pass"] else "FAIL")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
