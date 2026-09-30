"""Local self-check of exported Harbor tasks (no docker available here, so each container is emulated by a
subprocess with the same files and env): oracle must score 1, nop 0, and every near-miss 0.

  python3 selfcheck.py <task_dir> [...]"""
import json, os, re, shutil, socket, subprocess, sys, tempfile, time

PY = sys.executable


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def verify(task, app, env_extra=None):
    rd = tempfile.mkdtemp()
    if os.path.exists(f"{task}/tests/verify_e1.py"):
        env = dict(os.environ, DECISION=f"{app}/decision.json", SCORING=f"{task}/tests/scoring.json", REWARD_DIR=rd)
        cmd = [PY, f"{task}/tests/verify_e1.py"]
    else:
        env = dict(os.environ, APP=app, SPEC=f"{task}/tests/verify_spec.json", REWARD_DIR=rd, PYTHONPATH=f"{task}/tests")
        cmd = [PY, f"{task}/tests/verify_a1.py"]
    out = subprocess.run(cmd, env=dict(env, **(env_extra or {})), capture_output=True, text=True, timeout=900)
    r = open(f"{rd}/reward.txt").read().strip() if os.path.exists(f"{rd}/reward.txt") else "missing"
    return r, (out.stdout.strip() or out.stderr.strip())[-300:]


def check_e1(task):
    res = []
    port = free_port(); led = tempfile.mktemp()
    budget = int(re.search(r"LAB_BUDGET=(\d+)", open(f"{task}/environment/docker-compose.yaml").read()).group(1))
    srv = subprocess.Popen([PY, "lab_service.py"], cwd=f"{task}/environment/lab",
                           env=dict(os.environ, LAB_WORLD="world.json", LAB_BUDGET=str(budget), LAB_LEDGER=led,
                                    LAB_PORT=str(port), LAB_BIND="127.0.0.1"))
    try:
        app = tempfile.mkdtemp(); shutil.copytree(f"{task}/environment/app", app, dirs_exist_ok=True)
        env = dict(os.environ, APP=app, LAB_URL=f"http://127.0.0.1:{port}/")
        for _ in range(60):
            if subprocess.run([PY, f"{app}/lab", "status"], env=env, capture_output=True).returncode == 0: break
            time.sleep(0.5)
        t = time.time()
        o = subprocess.run([PY, f"{task}/solution/expert_e1.py"], env=env, capture_output=True, text=True, timeout=600)
        res.append(("oracle (blind expert, %.0fs)" % (time.time() - t), "1", *verify(task, app)))
        extra = subprocess.run([PY, f"{app}/lab", "run", "--changes", "c1"], env=env, capture_output=True, text=True)
        res.append(("budget enforced (run #%d refused)" % (budget + 1), "1", "1" if "exhausted" in extra.stdout else "0", extra.stdout.strip()[:120]))
        # nop
        nop = tempfile.mkdtemp(); res.append(("nop (no decision.json)", "0", *verify(task, nop)))
        # near misses built from the hidden truth
        sc = json.load(open(f"{task}/tests/scoring.json")); mu = sc["mu"]; opt = sc["opt"]
        names = ["c1", "c2", "c3", "c4", "c5", "c6"]; nm = lambda m: [n for i, n in enumerate(names) if m >> i & 1]
        note = open(f"{task}/environment/app/notes/teammate_ablation.md").read()
        plan = re.search(r"ship the winners together \(([^)]*)\)", note).group(1).replace(" ", "").split(",")
        pm = sum(1 << names.index(c) for c in plan)
        reg = [x / mu[opt] - 1 for x in mu]
        worst_ok = max((m for m in range(64) if reg[m] <= sc["delta_rel"]), key=lambda m: reg[m])
        best_bad = min((m for m in range(64) if reg[m] > sc["delta_rel"]), key=lambda m: reg[m])
        cases = [("positive control: optimum, exact prediction", nm(opt), mu[opt], "1"),
                 ("positive control: worst accepted cell, prediction +5%", nm(worst_ok), mu[worst_ok] * 1.05, "1"),
                 ("near-miss: best rejected cell %s, exact prediction" % "+".join(nm(best_bad)), nm(best_bad), mu[best_bad], "0"),
                 ("near-miss: teammate plan %s" % "+".join(plan), plan, mu[pm], "0"),
                 ("near-miss: ship all six", names, mu[63], "0"),
                 ("near-miss: optimum, prediction off by 7%", nm(opt), mu[opt] * 1.07, "0"),
                 ("malformed: duplicate names", nm(opt) + nm(opt)[:1], mu[opt], "0"),
                 ("malformed: prediction is a string", nm(opt), str(mu[opt]), "0")]
        for label, ship, pred, want in cases:
            d = tempfile.mkdtemp(); json.dump({"ship": ship, "predicted_val_mse": pred}, open(f"{d}/decision.json", "w"))
            res.append((label, want, *verify(task, d)))
    finally:
        srv.terminate()
    return res


def check_a1(task):
    res = []
    app = tempfile.mkdtemp(); shutil.copytree(f"{task}/environment/app", app, dirs_exist_ok=True)
    res.append(("nop (unmodified repo)", "0", *verify(task, app)))
    fixed = tempfile.mkdtemp(); shutil.copytree(f"{task}/environment/app", fixed, dirs_exist_ok=True)
    p = subprocess.run(["patch", "-p1", "-i", f"{task}/solution/fix.patch"], cwd=fixed, capture_output=True, text=True)
    res.append(("oracle (solve.sh patch)", "1", *verify(task, fixed)))
    return res


if __name__ == "__main__":
    allok = True
    for task in sys.argv[1:]:
        print("==", os.path.basename(task.rstrip("/")))
        for label, want, got, info in (check_e1 if os.path.exists(f"{task}/tests/verify_e1.py") else check_a1)(task):
            ok = got == want; allok &= ok
            print("  %-4s %-55s want=%s got=%s  %s" % ("ok" if ok else "FAIL", label, want, got, info[:140].replace("\n", " ")))
    print("ALL OK" if allok else "SOME CHECKS FAILED")
