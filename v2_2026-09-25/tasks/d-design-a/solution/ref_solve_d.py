"""Reference solution for `d-design-a`."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_d as D

APP = sys.argv[1] if len(sys.argv) > 1 else "/app"
from fractions import Fraction

dd = D.Design(APP)
q = json.load(open(os.path.join(APP, "queries.json")))
out = {"contrasts": {}, "values": {}, "widths": {}, "plans": {}}
for it in q["contrasts"]:
    lo, hi = dd.T.diff(it["from"], it["to"])
    out["contrasts"][it["id"]] = {"lo": float(lo), "hi": float(hi)}
for it in q["values"]:
    out["values"][it["id"]] = "yes" if dd.voi(it["from"], it["to"], it["run_id"]) else "no"
for it in q["widths"]:
    out["widths"][it["id"]] = float(dd.width(it["from"], it["to"], set(it["recover"])))
for it in q["plans"]:
    k, sets = dd.plan(it["from"], it["to"], it["candidates"], Fraction(str(it["target"])))
    out["plans"][it["id"]] = {"k": k, "runs": sets[0]}
json.dump(out, open(os.path.join(APP, "answers.json"), "w"), indent=1)
print(json.dumps({k: len(v) for k, v in out.items()}))
