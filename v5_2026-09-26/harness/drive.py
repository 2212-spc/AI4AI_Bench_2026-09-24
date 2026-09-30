"""Advance every running agent run in parallel for one host time slice.
  SEC_DIR=<operator secret dir> python3 drive.py <budget_seconds> [run_dir ...]
Starts (inside this process, so they die with it) the loopback API gateway - the only holder of the
upstream key - and the lab service over RUN_ROOT.  Then ticks each running agent until the deadline."""
import glob, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v3/runs")
budget = float(sys.argv[1])
dirs = [d.rstrip("/") for d in (sys.argv[2:] or sorted(glob.glob(RUN_ROOT + "/*/")))]
deadline = time.time() + budget
import gateway
gateway.start(os.path.join(RUN_ROOT, "gateway.log"))
from scalelab import lab_server
lab_server.start()
procs = []
for d in dirs:
    sp = d + "/state.json"
    if not os.path.exists(sp):
        continue
    st = json.load(open(sp))
    if st["status"] != "running":
        continue
    kind = "cc_agent.py" if "session" in st else "gpt_agent.py"
    procs.append((d, subprocess.Popen(["python3", os.path.join(HERE, kind), "tick", d, str(deadline)],
                                      stdout=open(d + "/tick.log", "a"), stderr=subprocess.STDOUT)))
for d, p in procs:
    try:
        p.wait(timeout=max(1, deadline - time.time() + 8))
    except subprocess.TimeoutExpired:
        p.kill()
n_run = 0
for d in sorted(glob.glob(RUN_ROOT + "/*/")):
    d = d.rstrip("/")
    if not os.path.exists(d + "/state.json"):
        continue
    st = json.load(open(d + "/state.json"))
    n_led = sum(1 for _ in open(d + "/lab_ledger.jsonl")) if os.path.exists(d + "/lab_ledger.jsonl") else 0
    extra = ("calls=%d out=%d" % (st["n_calls"], st["usage"]["output"])) if "usage" in st else ("ticks=%d turns=%d $%.2f" % (st["n_ticks"], st["turns"], st["cost"]))
    n_run += st["status"] == "running"
    print("%-40s %-9s %-24s lab_runs=%-3d %.0fmin" % (os.path.basename(d), st["status"], extra, n_led, (time.time() - st["started"]) / 60))
print("running:", n_run)
