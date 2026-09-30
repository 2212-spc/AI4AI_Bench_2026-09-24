"""Execution-based oracle pass rate: run the blind expert against a fresh lab sidecar with N different seed bases."""
import json, os, re, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfcheck import free_port, verify, PY
task, N = sys.argv[1], int(sys.argv[2])
budget = int(re.search(r"LAB_BUDGET=(\d+)", open(f"{task}/environment/docker-compose.yaml").read()).group(1))
wins = 0; ends = {}
for k in range(N):
    port = free_port(); led = tempfile.mktemp()
    srv = subprocess.Popen([PY, "lab_service.py"], cwd=f"{task}/environment/lab", env=dict(os.environ, LAB_WORLD="world.json",
                           LAB_BUDGET=str(budget), LAB_LEDGER=led, LAB_PORT=str(port), LAB_BIND="127.0.0.1"))
    try:
        app = tempfile.mkdtemp(); shutil.copytree(f"{task}/environment/app", app, dirs_exist_ok=True)
        env = dict(os.environ, APP=app, LAB_URL=f"http://127.0.0.1:{port}/", SEED_BASE=str(200000 + 1013 * k))
        for _ in range(60):
            if subprocess.run([PY, f"{app}/lab", "status"], env=env, capture_output=True).returncode == 0: break
            time.sleep(0.3)
        subprocess.run([PY, f"{task}/solution/expert_e1.py"], env=env, capture_output=True, text=True, timeout=600)
        r, info = verify(task, app); wins += r == "1"
        try: e = "+".join(json.loads(info)["ship"])
        except Exception: e = "?"
        ends[e] = ends.get(e, 0) + 1
    finally:
        srv.terminate()
print(os.path.basename(task), "oracle pass %d/%d" % (wins, N), ends)
