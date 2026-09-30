"""Quick diagnostics over results/*.json: first-fail histogram, nearest rival at G5/G9, loop actions."""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
for fn in sorted(glob.glob(os.path.join(HERE, "results", pat + ".json"))):
    J = json.load(open(fn))
    if J["na"]:
        continue
    ff = collections.Counter(r["first_fail"] or "CERT" for r in J["results"])
    near = collections.Counter(r["nearest"] for r in J["results"] if r["first_fail"] in ("G5_B", "G9_redteam"))
    acts = collections.Counter()
    for c in J["trace"]:
        for s in c["steps"]:
            if "action" in s:
                acts[s["action"].split(" -> ")[-1][:40]] += 1
    Bs = sorted(round(r["B_lo"], 1) for r in J["results"] if r.get("B_lo") is not None)
    print(os.path.basename(fn)[:-5], dict(ff))
    print("    nearest@fail:", dict(near.most_common(5)))
    print("    B_lo quartiles:", Bs[len(Bs) // 4] if Bs else None, Bs[len(Bs) // 2] if Bs else None,
          Bs[3 * len(Bs) // 4] if Bs else None, "max", Bs[-1] if Bs else None)
    if acts:
        print("    actions:", dict(acts.most_common(6)))
