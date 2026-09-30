"""Re-gate the tier-2 items certified BEFORE the shown-number check (results/prefix_T2_certified.json) with the current
gates: how many were leaks (the task showed a number within T of the answer) or were beaten by the added route?
usage: python3 audit_leak.py"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cards import CARDS            # noqa: E402
from compiler import evaluate      # noqa: E402

CARD = {c.key: c for c in CARDS}
old = json.load(open(os.path.join(HERE, "results", "prefix_T2_certified.json")))
cnt, why = collections.Counter(), collections.Counter()
for o in old:
    res, _ = evaluate(CARD[o["card"]], o["tpl"], o["world"], o["design"], o["seed"])
    cnt[(o["tpl"], "still" if res["certified"] else "lost")] += 1
    if not res["certified"]:
        bad = [k for k, v in res.get("shown_over_T", {}).items() if v < 1.0]
        why[(o["tpl"], res["first_fail"], ",".join(bad) or res.get("nearest_red") or "")] += 1
for t in ("M1", "M2", "M3", "M6"):
    print("%s: previously certified %d, still certified %d, lost %d" % (
        t, cnt[(t, "still")] + cnt[(t, "lost")], cnt[(t, "still")], cnt[(t, "lost")]))
for k, v in why.most_common():
    print("   lost:", k, v)
