"""Sweep world seeds for one blueprint, log every attempt, export by a stated rule.

    python3 tools/sweep.py <blueprint> <ws_from> <ws_to> [--want K] [--per-defect] [--out DIR]

Same builder and same log format as `scalelab.suite`; the only difference is the export rule.  `--want K`
takes the first K seeds that pass every gate.  `--per-defect` takes, for each defect id the injector can
choose, the first passing seed that drew it - a rule fixed before the sweep, so the exported set covers
the three defects without selecting on how hard any instance turned out to be.  Every attempt is written
to the log either way, so the natural yield is recoverable from the record rather than from this script's
choices.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from scalelab import build as B, export as X          # noqa: E402
from scalelab.suite import attempt                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bp"); ap.add_argument("ws_from", type=int); ap.add_argument("ws_to", type=int)
    ap.add_argument("--want", type=int, default=0)
    ap.add_argument("--per-defect", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "tasks"))
    ap.add_argument("--log", default=os.path.join(HERE, "build_log.jsonl"))
    a = ap.parse_args()
    bp = B.load(a.bp)
    seen = set(); n_exp = 0; n_ok = 0
    for ws in range(a.ws_from, a.ws_to + 1):
        inst, rec = attempt(a.bp, ws)
        d = (inst["w"]["ctx"] or {}).get("defect")
        rec["defect"] = d
        take = False
        if inst["ok"]:
            n_ok += 1
            if a.per_defect:
                take = d not in seen
                if take:
                    seen.add(d)
            elif a.want and n_exp < a.want:
                take = True
        if take:
            tid = "%s-ws%d" % (bp.ID, ws)
            X.export(inst, a.out, tid)
            rec["exported"] = tid; n_exp += 1
        rec["export_rule"] = "per_defect" if a.per_defect else ("first_%d" % a.want if a.want else "none")
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("%s ws=%-3d ok=%-5s %4.1fs defect=%-32s %s%s" % (
            a.bp, ws, rec["ok"], rec["secs"], d or "-", "EXPORT " + rec["exported"] if take else "",
            "" if inst["ok"] else "fails=" + ",".join(rec["fails"])))
        sys.stdout.flush()
    print("attempted %d, passed %d, exported %d" % (a.ws_to - a.ws_from + 1, n_ok, n_exp))


if __name__ == "__main__":
    main()
