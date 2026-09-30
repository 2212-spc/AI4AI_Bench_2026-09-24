"""python3 harness/summarize.py  -> pass rate per task per agent (targets: GPT-6 < 0.8, Opus 5 < 0.5 overall)."""
import glob, json, os
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tab = defaultdict(lambda: defaultdict(list))
for p in glob.glob(os.path.join(ROOT, "harness", "results", "*", "*", "run*.json")):
    r = json.load(open(p)); tab[r["task"]][r["agent"]].append(bool(r["pass"]))
agents = sorted({a for t in tab.values() for a in t})
print(f"{'task':18s}" + "".join(f"{a:>14s}" for a in agents))
tot = defaultdict(list)
for t in sorted(tab):
    row = f"{t:18s}"
    for a in agents:
        v = tab[t][a]; tot[a] += [sum(v) / len(v)] if v else []
        row += f"{(f'{sum(v)}/{len(v)}' if v else '-'):>14s}"
    print(row)
print(f"{'mean pass rate':18s}" + "".join(f"{(sum(tot[a])/len(tot[a]) if tot[a] else float('nan')):>14.2f}" for a in agents))
