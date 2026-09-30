"""Print one row per run: name, model, status, pass, score, acc, T, failing items, agent_min, api_errors.
  python3 harness/results_table.py [run_root]   (default /tmp/v10/runs; run_evidence/ works too)"""
import json, os, glob, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v10/runs"
rows=[]
for d in sorted(glob.glob(os.path.join(ROOT, "*/"))):
    n=os.path.basename(d.rstrip("/"))
    st=json.load(open(d+"state.json")) if os.path.exists(d+"state.json") else {}
    g=json.load(open(d+"grade.json")) if os.path.exists(d+"grade.json") else None
    if g:
        fails=[k for k,v in g["items"].items() if not v["ok"]]
        acc=g.get("diag",{}).get("acc") if isinstance(g.get("diag"),dict) else None
        T=g.get("ref",{}).get("T") if isinstance(g.get("ref"),dict) else None
        rows.append((n,g.get("model"),st.get("status"),g["pass"],round(g["score"],3),acc,T,",".join(fails),g.get("agent_min"),g.get("api_errors")))
    else:
        rows.append((n,st.get("model"),st.get("status"),None,None,None,None,"",None,None))
for r in rows: print("\t".join(str(x) for x in r))
