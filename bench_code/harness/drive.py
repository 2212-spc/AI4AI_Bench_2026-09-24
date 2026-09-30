"""Advance all running agent runs in parallel for one host time slice.
usage: python drive.py <budget_seconds> [run_dir ...]   (default: all dirs under /tmp/bench/runs with state.json)"""
import glob, json, os, subprocess, sys, time
budget = float(sys.argv[1])
dirs = sys.argv[2:] or sorted(glob.glob("/tmp/bench/runs/*/"))
deadline = time.time() + budget
procs = []
if any(os.path.exists(x.rstrip("/") + "/lab.json") for x in dirs):
    sys.path.insert(0, "/tmp/bench/lab")
    import lab_server; lab_server.start()
for d in dirs:
    d = d.rstrip("/")
    sp = d + "/state.json"
    if not os.path.exists(sp) or os.path.basename(d).startswith(("cctest", "gpttest")): continue
    st = json.load(open(sp))
    if st["status"] != "running": continue
    kind = "cc_agent.py" if "session" in st else "gpt_agent.py"
    procs.append((d, subprocess.Popen(["python3", "/tmp/bench/harness/" + kind, "tick", d, str(deadline)],
                                      stdout=open(d + "/tick.log", "a"), stderr=subprocess.STDOUT)))
for d, p in procs:
    try: p.wait(timeout=max(1, deadline - time.time() + 8))
    except subprocess.TimeoutExpired: p.kill()
for d in sorted(glob.glob("/tmp/bench/runs/*/")):
    d = d.rstrip("/")
    if os.path.basename(d).startswith(("cctest", "gpttest")) or not os.path.exists(d + "/state.json"): continue
    st = json.load(open(d + "/state.json"))
    extra = ("calls=%d tok_in=%d out=%d" % (st["n_calls"], st["usage"]["input"], st["usage"]["output"])) if "usage" in st else ("ticks=%d turns=%d cost=%.2f" % (st["n_ticks"], st["turns"], st["cost"]))
    print("%-34s %-9s %s  %.0fmin" % (os.path.basename(d), st["status"], extra, (time.time() - st["started"]) / 60))
