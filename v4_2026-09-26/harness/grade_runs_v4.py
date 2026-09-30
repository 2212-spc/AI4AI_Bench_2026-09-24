"""Grade every finished v4 run with the *shipped* grader.

  LAB_RUN_ROOT=/tmp/v4/runs python3 grade_runs_v4.py [out.jsonl]

v3's grade_runs.py called queries.grade(items, ans) directly, which is wrong for the three new forms:
cex and prereg are graded by execution and need a grading context (hidden world, disclosed rows, the
agent's own ledger).  So this driver runs each task's own `tests/grade.py` in a subprocess - the same
bytes a third party would run - and only parses its JSON.  Nothing here can make a run pass that the
shipped grader would fail.
"""
import glob, json, os, subprocess, sys

RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v4/runs")
out = []
for d in sorted(glob.glob(RUN_ROOT + "/*/lab.json")):
    rd = os.path.dirname(d)
    lj = json.load(open(d))
    st = json.load(open(rd + "/state.json"))
    ap = rd + "/app/answers.json"
    led = rd + "/lab_ledger.jsonl"
    g = {"score": 0.0, "all_pass": False, "items": {}}
    if os.path.exists(ap):
        p = subprocess.run(["python3", os.path.join(lj["task_dir"], "tests", "grade.py"), ap]
                           + ([led] if os.path.exists(led) else []),
                           capture_output=True, text=True, timeout=1800)
        try:
            g = json.loads(p.stdout[p.stdout.index("{"):])
        except Exception:
            g["err"] = (p.stdout + p.stderr)[-400:]
    n_led = sum(1 for _ in open(led)) if os.path.exists(led) else 0
    cost = None
    if os.path.exists(led):
        rows = [json.loads(l) for l in open(led)]
        cost = sum(r.get("cost", 0.0) for r in rows)
    rec = {"run": os.path.basename(rd), "task": lj["task"], "model": lj["model"], "hint": lj["hint"],
           "status": st["status"], "score": g["score"], "all_pass": g["all_pass"],
           "items": {q: v["pass"] for q, v in g.get("items", {}).items()},
           "why": {q: v.get("why") for q, v in g.get("items", {}).items()},
           "n_lab_runs": n_led, "lab_cost": cost,
           "n_calls": st.get("n_calls", st.get("turns")), "err": g.get("err")}
    out.append(rec)
    print("%-30s %-17s H%d %-9s score=%.2f all=%-5s %s lab=%-3d %s"
          % (rec["run"], rec["model"], rec["hint"], rec["status"], rec["score"], rec["all_pass"],
             "".join("1" if v else "0" for v in rec["items"].values()), n_led, rec["err"] or ""))
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", sys.argv[1])
