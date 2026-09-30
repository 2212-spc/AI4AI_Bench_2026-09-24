"""Trajectory diagnostics for E1 runs: which cells were explored, how budget was split, whether the key cells were ever touched.
  python3 traj_e1.py runs/E1w14*"""
import json, os, sys, collections
NAMES = ["c1","c2","c3","c4","c5","c6"]
def key(ch): return "+".join(sorted(ch, key=NAMES.index)) or "base"
def core(ch): return "+".join(c for c in sorted(ch, key=NAMES.index) if c != "c6") or "base"
for r in sys.argv[1:]:
    led = [json.loads(l) for l in open(r + "/lab_ledger.jsonl") if l.strip()]
    try: dec = json.load(open(r + "/app/decision.json")); ship = core(dec["ship"])
    except Exception: ship = "NONE"
    cnt = collections.Counter(core(e["changes"]) for e in led)
    top, topn = cnt.most_common(1)[0]
    has = lambda s: cnt.get(s, 0)
    first = next((e["run"] for e in led if core(e["changes"]) == ship), None)
    singles = sum(1 for e in led if len([c for c in e["changes"] if c != "c6"]) <= 1)
    print("%-20s ship=%-12s runs=%2d cells(core)=%2d  top=%s x%d  | c1c2c3c5:%d c1c2c3:%d c1c2c5:%d c3c4c5:%d  | first_hit_ship=%s singles=%d" % (
        os.path.basename(r), ship, len(led), len(cnt), top, topn,
        has("c1+c2+c3+c5"), has("c1+c2+c3"), has("c1+c2+c5"), has("c3+c4+c5"), first, singles))
