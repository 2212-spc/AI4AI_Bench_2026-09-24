"""Re-grade finished runs with the verifier each task ships.

With no arguments this regrades every run in `/tmp/bench2/runs`.  Because a full sweep can outrun the
harness' per-call time budget, run names may be passed as arguments; those runs are graded and their rows
are merged into the existing `evidence/grades.json` instead of replacing it.
"""
import glob, json, os, subprocess, sys, tempfile
V = {"o-pack-a": "verify_o.py", "v-falsify-a": "verify_v.py",
     "s-censored-a": "verify_s.py", "b-bounds-a": "verify_b.py", "c-audit-a": "verify_c.py",
     "b-bounds-sealed": "verify_bs.py", "d-design-a": "verify_d.py",
     "d-design-sealed": "verify_ds.py", "d-design-open": "verify_ds.py"}
STALE = {"cc_b-bounds-a": "solved b-bounds-a v1 (24/24); v1 key superseded, not re-gradeable",
         "gpt_b-bounds-a": "abandoned mid-run when b-bounds-a v2 replaced v1",
         "cc_b2": "solved b-bounds-a v2 (24/24); v2 key was wrong by one grid step, superseded by v3",
         "gpt_b2": "solved b-bounds-a v2 (24/24); v2 key was wrong by one grid step, superseded by v3",
         "cc_s-censored-a": "superseded by the cc_s2 re-run on the same task revision",
         # The three D arms were rebuilt after the audit gates found a knife-edge plan target, a
         # one-column proxy for the values labels and an ambiguous recovery sentence.  The rebuild moved
         # the items, so these runs answered questions that no longer exist; their v1 scores are quoted
         # here rather than re-graded against a key they never saw.
         "cc_d1": "scored 27/27 on d-design-a v1; items rebuilt in v2, superseded by cc_d1v2",
         "gpt_d1": "scored 27/27 on d-design-a v1; items rebuilt in v2, superseded by gpt_d1v2",
         "cc_ds": "scored 31/31 on d-design-sealed v1; items rebuilt in v2, superseded by cc_dsv2",
         "gpt_ds": "scored 26/31 on d-design-sealed v1; items rebuilt in v2, superseded by gpt_dsv2",
         "cc_do": "scored 31/31 on d-design-open v1; items rebuilt in v2, superseded by cc_dov2",
         "gpt_do": "scored 30/31 on d-design-open v1; items rebuilt in v2, superseded by gpt_dov2"}
OUTP = "/sessions/cool-magical-cerf/mnt/outputs/evidence/grades.json"
only = set(sys.argv[1:])
out = []
for d in sorted(glob.glob("/tmp/bench2/runs/*/")):
    d = d.rstrip("/"); n = os.path.basename(d)
    if not os.path.exists(d + "/state.json") or n.startswith(("cctest", "gpttest")): continue
    if only and n not in only: continue
    st = json.load(open(d + "/state.json"))
    SHORT = {"b2": "b-bounds-a", "b3": "b-bounds-a", "s2": "s-censored-a", "bs": "b-bounds-sealed",
             "d1": "d-design-a", "ds": "d-design-sealed", "do": "d-design-open",
             "d1v2": "d-design-a", "dsv2": "d-design-sealed", "dov2": "d-design-open"}
    task = SHORT.get(n.split("_", 1)[1], n.split("_", 1)[1])
    if n in STALE:                       # answered an earlier revision of a task whose key has changed
        out.append({"run": n, "task": task, "model": "gpt-6-astra" if "usage" in st else "fable-5.1",
                    "status": st["status"], "reward": None, "note": STALE[n]})
        continue
    t = "/tmp/bench2/tasks/" + task
    if not os.path.exists(t + "/tests/" + V.get(task, "")): continue
    logs = tempfile.mkdtemp()
    env = dict(os.environ, APP=d + "/app", TESTS=t + "/tests", REWARD_DIR=logs)
    try:
        p = subprocess.run(["python3", t + "/tests/" + V[task]], env=env, capture_output=True,
                           text=True, timeout=1800)
        j = json.loads(p.stdout.strip().splitlines()[-1])
        det = j.get("detail", {})
        sect = det.get("by_section") or {}
        n_it = det.get("n_items") or (sum(v["n"] for v in sect.values()) if sect else None)
        n_fl = det.get("n_failed")
        if n_fl is None and sect: n_fl = sum(v["failed"] for v in sect.values())
        fail = [(f if isinstance(f, str) else f.get("item")) for f in det.get("failed", [])]
        out.append({"run": n, "task": task, "model": "gpt-6-astra" if "usage" in st else "fable-5.1",
                    "status": st["status"], "reward": j.get("reward"),
                    "n_failed": n_fl, "n_items": n_it, "failed": fail[:8]})
    except Exception as e:
        out.append({"run": n, "task": task, "error": str(e)[:200]})
if only and os.path.exists(OUTP):                 # merge a partial sweep into the standing record
    keep = [r for r in json.load(open(OUTP)) if r["run"] not in {r2["run"] for r2 in out}]
    out = sorted(keep + out, key=lambda r: r["run"])
json.dump(out, open(OUTP, "w"), indent=1)
for r in out:
    print("%-22s %-14s %-12s reward=%s  failed=%s/%s %s" % (r["run"], r.get("task"), r.get("model"),
          r.get("reward"), r.get("n_failed"), r.get("n_items"), r.get("failed", r.get("note", ""))))
