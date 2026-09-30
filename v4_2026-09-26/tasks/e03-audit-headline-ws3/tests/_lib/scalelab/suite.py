"""Suite driver: for each blueprint, draw world seeds in order, build + gate each, export the first K that pass.

  python3 -m scalelab.suite <blueprint> <ws_from> <ws_to> [--want K] [--out DIR] [--log FILE]

Every attempted seed is logged (pass or fail, and why) so the natural yield and the reasons for rejection
are part of the record - the adversarial selection is explicit, not hidden."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scalelab import build as B, export as X

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def attempt(bp_name, ws):
    t = time.time()
    inst = B.build_instance(bp_name, ws, shipped_text_fn=X.shipped_text)
    fails = {g: {k: v for k, v in gv.items() if k not in ("certs", "item_kills", "cards")}
             for g, gv in inst["gates"].items() if not gv["pass"]}
    rec = {"bp": bp_name, "ws": ws, "ok": inst["ok"], "secs": round(time.time() - t, 1), "fails": fails,
           "tol": {k: round(v, 4) for k, v in inst["cal"]["tol"].items()},
           "ver_allpass": inst["cal"]["ver_allpass"], "ver_pass": inst["cal"]["ver_pass"],
           "rivals": {k: {"score": round(v["score"], 3), "margin": v["margin_items"], "must_kill": v["must_kill"]}
                      for k, v in inst["mat"].items()}}
    return inst, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bp"); ap.add_argument("ws_from", type=int); ap.add_argument("ws_to", type=int)
    ap.add_argument("--want", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "tasks"))
    ap.add_argument("--log", default=os.path.join(ROOT, "build_log.jsonl"))
    a = ap.parse_args()
    bpmod = B.load(a.bp)
    have = len([d for d in os.listdir(a.out) if d.startswith(bpmod.ID + "-ws")]) if os.path.isdir(a.out) else 0
    for ws in range(a.ws_from, a.ws_to + 1):
        if a.want and have >= a.want:
            break
        inst, rec = attempt(a.bp, ws)
        if inst["ok"] and a.want:
            tid = "%s-ws%d" % (bpmod.ID, ws)
            X.export(inst, a.out, tid); rec["exported"] = tid; have += 1
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("%s ws=%d ok=%s %.1fs fails=%s%s" % (a.bp, ws, rec["ok"], rec["secs"], list(rec["fails"]),
              ("  " + json.dumps(rec["fails"].get("G2_kill_matrix", ""))) if "G2_kill_matrix" in rec["fails"] else ""))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
