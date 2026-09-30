"""Family B, sealed arm: verifier.  Reads only `/app/answers.json` against `/tests/key.json`.

Same all-or-nothing rule as the documented arm, over two more kinds of item.  `supports` are graded like
any other interval - a valid but loose single-run bound fails, because the day's published total is part of
what is known.  `mechanism` is graded exactly: `screen_code` must be one of the codes that actually occur
in the removal log and must be the right one, and `grid` must be the exact step, not a plausible power of
ten.  The per-section tally in `detail` is reported whether the run passes or fails, so a failure says
which capability broke rather than only that something did.
"""
import json
import os
import sys

TESTS = os.environ.get("TESTS", "/tests")
APP = os.environ.get("APP", "/app")
REWARD_DIR = os.environ.get("REWARD_DIR", "/logs/verifier")
LABELS = ("yes", "no", "cannot_tell")
RANGE = {"cells": (0.0, 1.0), "contrasts": (-1.0, 1.0), "supports": (0.0, 1.0)}


def finish(ok, detail):
    os.makedirs(REWARD_DIR, exist_ok=True)
    open(os.path.join(REWARD_DIR, "reward.txt"), "w").write("1" if ok else "0")
    print(json.dumps({"reward": 1 if ok else 0, "detail": detail}, default=str)[:20000])
    sys.exit(0)


def num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError("not a number: %r" % (x,))
    return float(x)


def main():
    key = json.load(open(os.path.join(TESTS, "key.json")))
    tol = float(key["tol"])
    path = os.path.join(APP, "answers.json")
    if not os.path.exists(path):
        finish(False, {"error": "no /app/answers.json"})
    try:
        sub = json.load(open(path))
    except Exception as e:
        finish(False, {"error": "answers.json is not valid JSON: %s" % e})
    if not isinstance(sub, dict):
        finish(False, {"error": "answers.json must be a JSON object"})

    bad, err, per = [], {}, {}
    for section in ("cells", "contrasts", "supports"):
        lim = RANGE[section]
        got_sec = sub.get(section)
        per[section] = {"n": len(key[section]), "failed": 0}
        if not isinstance(got_sec, dict):
            finish(False, {"error": "`%s` must be a JSON object" % section})
        for qid in sorted(key[section]):
            want = key[section][qid]
            got = got_sec.get(qid)
            if not isinstance(got, dict):
                bad.append({"item": qid, "why": "missing or malformed"})
                per[section]["failed"] += 1
                continue
            try:
                lo, hi = num(got.get("lo")), num(got.get("hi"))
            except ValueError as e:
                bad.append({"item": qid, "why": str(e)})
                per[section]["failed"] += 1
                continue
            if lo > hi:
                bad.append({"item": qid, "why": "lo must not exceed hi", "got": [lo, hi]})
                per[section]["failed"] += 1
                continue
            if not (lim[0] - 1e-9 <= lo and hi <= lim[1] + 1e-9):
                bad.append({"item": qid, "why": "endpoint outside the possible range %s" % (lim,),
                            "got": [lo, hi]})
                per[section]["failed"] += 1
                continue
            e = max(abs(lo - want["lo"]), abs(hi - want["hi"]))
            err[qid] = round(e, 6)
            if e > tol + 1e-12:
                side = "lo" if abs(lo - want["lo"]) > abs(hi - want["hi"]) else "hi"
                bad.append({"item": qid, "why": "outside tolerance", "worst_side": side,
                            "abs_error": round(e, 6), "tol": tol,
                            "direction": ("too wide" if lo < want["lo"] - tol or hi > want["hi"] + tol
                                          else "too narrow")})
                per[section]["failed"] += 1

    got_sec = sub.get("decisions")
    per["decisions"] = {"n": len(key["decisions"]), "failed": 0}
    if not isinstance(got_sec, dict):
        finish(False, {"error": "`decisions` must be a JSON object"})
    for qid in sorted(key["decisions"]):
        g = got_sec.get(qid)
        if g not in LABELS:
            bad.append({"item": qid, "why": "decision must be one of %s" % (LABELS,), "got": g})
            per["decisions"]["failed"] += 1
        elif g != key["decisions"][qid]:
            bad.append({"item": qid, "why": "wrong decision", "got": g})
            per["decisions"]["failed"] += 1

    mech = sub.get("mechanism")
    per["mechanism"] = {"n": 2, "failed": 0}
    if not isinstance(mech, dict):
        finish(False, {"error": "`mechanism` must be a JSON object"})
    want = key["mechanism"]
    code = mech.get("screen_code")
    if not isinstance(code, str) or code.strip() not in want["codes_present"]:
        bad.append({"item": "mechanism.screen_code",
                    "why": "must be one of the codes that occur in retention_log.csv", "got": code})
        per["mechanism"]["failed"] += 1
    elif code.strip() != want["screen_code"]:
        bad.append({"item": "mechanism.screen_code", "why": "wrong code", "got": code})
        per["mechanism"]["failed"] += 1
    try:
        g = num(mech.get("grid"))
        if abs(g - float(want["grid"])) > 1e-12:
            bad.append({"item": "mechanism.grid", "why": "wrong grid", "got": g})
            per["mechanism"]["failed"] += 1
    except ValueError as e:
        bad.append({"item": "mechanism.grid", "why": str(e)})
        per["mechanism"]["failed"] += 1

    n = sum(v["n"] for v in per.values())
    finish(not bad, {"n_items": n, "n_failed": len(bad), "failed": bad[:12], "by_section": per,
                     "abs_errors": err})


if __name__ == "__main__":
    main()
