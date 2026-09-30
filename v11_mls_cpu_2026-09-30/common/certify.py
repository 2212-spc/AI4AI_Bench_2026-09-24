"""Correctness certificate for one task.

    python3 common/certify.py tasks/<task> [--seeds N]

Grades every candidate solution with the task's own host-side grader:
  reference/*.py   -> must PASS          (the task is solvable, threshold is reachable)
  workdir/baselines/*.py -> must FAIL    (strong known methods are not enough)
  decoys/*.py      -> must FAIL          (generic 'robust' hedges are not enough)
Writes tasks/<task>/certificate.json.  A task ships only if certificate["ok"] is true.
"""
import importlib.util, json, os, shutil, sys, tempfile, time, glob


def load_grader(task_dir):
    spec = importlib.util.spec_from_file_location("grade", os.path.join(task_dir, "grade.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def solution_dir_for(task_dir, sol_file, target="solution.py"):
    tmp = tempfile.mkdtemp(prefix="v11cert_")
    d = os.path.join(tmp, "workdir")
    shutil.copytree(os.path.join(task_dir, "workdir"), d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(sol_file, os.path.join(d, target))
    return tmp, d


def main():
    task_dir = os.path.abspath(sys.argv[1])
    g = load_grader(task_dir)
    cands = [("reference", p) for p in sorted(glob.glob(os.path.join(task_dir, "reference", "*.py")))]
    cands += [("baseline", p) for p in sorted(glob.glob(os.path.join(task_dir, "workdir", "baselines", "*.py")))]
    cands += [("decoy", p) for p in sorted(glob.glob(os.path.join(task_dir, "decoys", "*.py")))]
    rows, ok = [], True
    for kind, path in cands:
        tmp, d = solution_dir_for(task_dir, path, getattr(g, "SOL_FILE", "solution.py"))
        t0 = time.time()
        try:
            r = g.grade(d)
        except Exception as e:  # a crashing candidate simply fails
            r = {"pass": False, "error": repr(e)[:500], "settings": {}}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        want = kind == "reference"
        good = bool(r["pass"]) == want
        ok &= good
        fails = [k for k, s in r.get("settings", {}).items() if not s.get("ok")]
        rows.append(dict(kind=kind, name=os.path.basename(path), passed=bool(r["pass"]), as_expected=good,
                         failed_settings=fails, secs=round(time.time() - t0, 1), detail=r))
        print(f"[{'OK ' if good else 'BAD'}] {kind:9s} {os.path.basename(path):28s} pass={r['pass']!s:5s} "
              f"failed={fails} {time.time()-t0:.1f}s" + (f" err={r.get('error')}" if 'error' in r else ""))
        for k, s in r.get("settings", {}).items():
            print(f"        {k:10s} hidden={s.get('hidden')!s:5s} metric={s.get('metric'):.4g} thr={s.get('threshold'):.4g} ok={s.get('ok')}")
    # hidden settings must matter: at least one non-reference candidate passes all dev settings
    # but fails a hidden one (otherwise the hidden split adds no discrimination; reported, not required)
    dev_only = [r["name"] for r in rows if r["kind"] != "reference" and r["failed_settings"]
                and all(r["detail"]["settings"][k]["hidden"] for k in r["failed_settings"])]
    cert = dict(task=os.path.basename(task_dir), ok=ok, created=time.strftime("%Y-%m-%d %H:%M"),
                candidates=rows, pass_dev_fail_hidden=dev_only)
    with open(os.path.join(task_dir, "certificate.json"), "w") as f:
        json.dump(cert, f, indent=1, default=float)
    print("CERTIFICATE", "OK" if ok else "FAILED", "| pass-dev-but-fail-hidden:", dev_only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
