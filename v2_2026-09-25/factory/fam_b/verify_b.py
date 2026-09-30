"""Family B verifier.  Reads only `/app/answers.json` against `/tests/key.json`.

Three shapes of item, one rule: the published answer is exact arithmetic, so the tolerance is a rounding
allowance (5e-4), not a confidence band.  A wider-than-necessary interval fails the same way a wrong one
does - reporting a valid but loose bound is the specific error this task exists to catch.  Scoring is
all-or-nothing over every item in the key.
"""
import json
import os
import sys

TESTS = os.environ.get("TESTS", "/tests")
APP = os.environ.get("APP", "/app")
REWARD_DIR = os.environ.get("REWARD_DIR", "/logs/verifier")
LABELS = ("yes", "no", "cannot_tell")


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

    bad, err = [], {}
    for section, lim in (("cells", (0.0, 1.0)), ("contrasts", (-1.0, 1.0))):
        got_sec = sub.get(section)
        if not isinstance(got_sec, dict):
            finish(False, {"error": "`%s` must be a JSON object" % section})
        for qid in sorted(key[section]):
            want = key[section][qid]
            got = got_sec.get(qid)
            if not isinstance(got, dict):
                bad.append({"item": qid, "why": "missing or malformed"})
                continue
            try:
                lo, hi = num(got.get("lo")), num(got.get("hi"))
            except ValueError as e:
                bad.append({"item": qid, "why": str(e)})
                continue
            if lo > hi:
                bad.append({"item": qid, "why": "lo must not exceed hi", "got": [lo, hi]})
                continue
            if not (lim[0] - 1e-9 <= lo and hi <= lim[1] + 1e-9):
                bad.append({"item": qid, "why": "endpoint outside the possible range %s" % (lim,),
                            "got": [lo, hi]})
                continue
            e = max(abs(lo - want["lo"]), abs(hi - want["hi"]))
            err[qid] = round(e, 6)
            if e > tol + 1e-12:
                side = "lo" if abs(lo - want["lo"]) > abs(hi - want["hi"]) else "hi"
                bad.append({"item": qid, "why": "outside tolerance", "worst_side": side,
                            "abs_error": round(e, 6), "tol": tol,
                            "direction": ("too wide" if lo < want["lo"] - tol or hi > want["hi"] + tol
                                          else "too narrow")})

    got_sec = sub.get("decisions")
    if not isinstance(got_sec, dict):
        finish(False, {"error": "`decisions` must be a JSON object"})
    for qid in sorted(key["decisions"]):
        g = got_sec.get(qid)
        if g not in LABELS:
            bad.append({"item": qid, "why": "decision must be one of %s" % (LABELS,), "got": g})
        elif g != key["decisions"][qid]:
            bad.append({"item": qid, "why": "wrong decision", "got": g})

    n = len(key["cells"]) + len(key["contrasts"]) + len(key["decisions"])
    finish(not bad, {"n_items": n, "n_failed": len(bad), "failed": bad[:12], "abs_errors": err})


if __name__ == "__main__":
    main()
