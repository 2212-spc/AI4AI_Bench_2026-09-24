"""Collect one cost/effort row per run from the agents' own `state.json` files.

The two drivers keep different books - the Fable driver reports billed cost and turns, the GPT driver
reports API calls, token counts and how many times the harness had to downgrade reasoning effort after a
read timeout - so this flattens both into a single shape and writes `evidence/runs.json`.  Kept separate
from `grade_all.py` because it touches no verifier and so always finishes inside one harness call.
"""
import glob
import json
import os
import time

OUTP = "/sessions/cool-magical-cerf/mnt/outputs/evidence/runs.json"
rows = []
for d in sorted(glob.glob("/tmp/bench2/runs/*/")):
    d = d.rstrip("/")
    n = os.path.basename(d)
    if not os.path.exists(d + "/state.json") or n.startswith(("cctest", "gpttest")):
        continue
    st = json.load(open(d + "/state.json"))
    gpt = "usage" in st
    u = st.get("usage") or {}
    end = max([os.path.getmtime(d + "/state.json")] +
              [os.path.getmtime(p) for p in glob.glob(d + "/*.jsonl")])
    rows.append({
        "run": n,
        "model": "gpt-6-astra" if gpt else "fable-5.1",
        "status": st.get("status"),
        "minutes": round((end - st.get("started", end)) / 60.0, 1),
        "calls": st.get("n_calls") if gpt else None,
        "turns": None if gpt else st.get("turns"),
        "ticks": None if gpt else st.get("n_ticks"),
        "cost_usd": None if gpt else round(st.get("cost", 0.0), 3),
        "tok_in": u.get("input") if gpt else None,
        "tok_out": u.get("output") if gpt else None,
        "api_errors": st.get("api_errors") if gpt else None,
        "effort_downgrades": st.get("effort_downgrades") if gpt else None,
    })
json.dump(rows, open(OUTP, "w"), indent=1)
print("%-22s %-12s %-8s %6s %6s %8s %10s %10s" % ("run", "model", "status", "min", "calls", "cost", "tok_in", "tok_out"))
for r in rows:
    print("%-22s %-12s %-8s %6s %6s %8s %10s %10s" % (r["run"], r["model"], r["status"], r["minutes"],
          r["calls"] or r["turns"], r["cost_usd"] or "-", r["tok_in"] or "-", r["tok_out"] or "-"))
print(time.strftime("collected %Y-%m-%d %H:%M"))
