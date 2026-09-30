"""Advance every running agent run in parallel for one host time slice (v7 / L1.5).
  SEC_DIR=<operator secret dir> python3 drive.py <budget_seconds> [run_dir ...]
Starts (inside this process, so they die with it) the loopback API gateway - the only holder of the
upstream key - and the L1.5 lab service over RUN_ROOT.  Then ticks each running agent until the deadline.
Runs older than MAX_RUN_MIN wall-clock minutes of *agent* time are stopped (status 'timeout') and graded as is."""
import glob, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v7/runs")
MAX_RUN_MIN = float(os.environ.get("MAX_RUN_MIN", "100"))
MAX_PAR = int(os.environ.get("MAX_PAR", "10"))
budget = float(sys.argv[1])
dirs = [d.rstrip("/") for d in (sys.argv[2:] or sorted(glob.glob(RUN_ROOT + "/*/")))]
deadline = time.time() + budget
import gateway
gateway.start(os.path.join(RUN_ROOT, "gateway.log"))
from l15 import server
server.start()


def active_minutes(st):
    # agent time = sum of slices actually spent ticking (tracked by us), not wall time since launch
    return st.get("agent_sec", 0.0) / 60


# Pick the LEAST-advanced runs first, not the alphabetically-first ones.
#
# The original loop took the first MAX_PAR in sorted order, which is stable across ticks - so with 21 live
# runs and MAX_PAR=12 the same 12 ran every slice and the tail (k6w1, k8a1, ...) sat at 0 lab calls for an
# hour while the head finished.  Fair queueing by accumulated agent time means every run advances, and a
# sweep that is 80% done stops looking like a sweep that is stuck.
ready = []
for d in dirs:
    sp = d + "/state.json"
    if not os.path.exists(sp):
        continue
    st = json.load(open(sp))
    if st["status"] != "running":
        continue
    if active_minutes(st) > MAX_RUN_MIN:
        st["status"] = "timeout"; json.dump(st, open(sp, "w")); continue
    # Age-credit so long-running runs cannot be starved forever.
    #
    # Pure agent_sec fair queueing inverted the original bug: with 26 live runs and MAX_PAR=12, the five
    # runs with the MOST accumulated time (k2w1 sonnet5 at 5712s, k4w1 gpt6 at 4345s, ...) never re-entered
    # the front of the queue and sat untouched for 100 minutes.  Subtracting the idle time means a run that
    # has been waiting a while climbs back; a run that just executed drops.  Both directions of starvation
    # are now bounded.
    idle = time.time() - os.path.getmtime(os.path.join(d, "state.json"))
    ready.append((st.get("agent_sec", 0.0) - idle, d, st))
ready.sort(key=lambda x: x[0])
procs = []
for _sec, d, st in ready:
    if len(procs) >= MAX_PAR:
        break
    kind = "cc_agent.py" if "session" in st else "gpt_agent.py"
    procs.append((d, time.time(), subprocess.Popen(["python3", os.path.join(HERE, kind), "tick", d, str(deadline)],
                                                   stdout=open(d + "/tick.log", "a"), stderr=subprocess.STDOUT)))
for d, t0, p in procs:
    try:
        p.wait(timeout=max(1, deadline - time.time() + 8))
    except subprocess.TimeoutExpired:
        p.kill()
    st = json.load(open(d + "/state.json"))
    st["agent_sec"] = st.get("agent_sec", 0.0) + (time.time() - t0)
    tmp = d + "/state.json.tmp2"; json.dump(st, open(tmp, "w")); os.replace(tmp, d + "/state.json")
counts = {}
lines = []
for d in sorted(glob.glob(RUN_ROOT + "/*/")):
    d = d.rstrip("/")
    if not os.path.exists(d + "/state.json"):
        continue
    st = json.load(open(d + "/state.json"))
    counts[st["status"]] = counts.get(st["status"], 0) + 1
    if st["status"] != "running" and "--all" not in sys.argv:
        continue
    n_led = sum(1 for _ in open(d + "/lab_ledger.jsonl")) if os.path.exists(d + "/lab_ledger.jsonl") else 0
    extra = ("c=%d" % st["n_calls"]) if "usage" in st else ("t=%d/%d" % (st["n_ticks"], st["turns"]))
    lines.append("%-34s %-8s lab=%-3d %3.0fm" % (os.path.basename(d)[:34], extra, n_led, active_minutes(st)))
print("\n".join(lines))
print("status:", json.dumps(counts))
