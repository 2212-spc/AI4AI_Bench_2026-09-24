"""Grade every finished run: RUN_ROOT/*/app/answers.json against the task's tests/key.json.
  python3 grade_runs.py [out.jsonl]"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scalelab import queries as Q
RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v3/runs")
out = []
for d in sorted(glob.glob(RUN_ROOT + "/*/lab.json")):
    rd = os.path.dirname(d); lj = json.load(open(d))
    st = json.load(open(rd + "/state.json"))
    items = json.load(open(os.path.join(lj["task_dir"], "tests", "key.json")))
    try:
        ans = json.load(open(rd + "/app/answers.json"))
    except Exception as e:
        ans = {}
    g = Q.grade(items, ans)
    led = [json.loads(l) for l in open(rd + "/lab_ledger.jsonl")] if os.path.exists(rd + "/lab_ledger.jsonl") else []
    flops = sum(6 * r["req"].get("N", 0) * r["req"].get("D", 0) for r in led)
    rec = {"run": os.path.basename(rd), "task": lj["task"], "model": lj["model"], "hint": lj["hint"], "status": st["status"],
           "score": g["score"], "all_pass": g["all_pass"], "items": {q: v["pass"] for q, v in g["items"].items()},
           "answers": ans, "n_lab_runs": len(led)}
    out.append(rec)
    print("%-40s %-18s H%d %-9s score=%.2f all=%s  %s  lab=%d" % (rec["run"], rec["model"], rec["hint"], rec["status"], rec["score"],
          rec["all_pass"], "".join("1" if v else "0" for v in rec["items"].values()), rec["n_lab_runs"]))
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
