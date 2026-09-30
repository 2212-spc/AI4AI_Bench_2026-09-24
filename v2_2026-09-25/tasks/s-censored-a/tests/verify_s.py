"""Family S verifier.  Reads only the agent artifact /app/answers.json against /tests/key.json.

No agent-authored code is executed, so there is nothing to sandbox; the key simply lives in the verifier
container, which the agent cannot reach.  Scoring is item-level and all-or-nothing:

  * 13 verdicts, and for each abstention the reason *code* (a controlled vocabulary - distinguishing
    "every run was censored" from "this combination was never launched" from "this level was never
    launched" is the point of the item, so a right verdict with the wrong code does not count);
  * 9 contrasts within their published per-query tolerance;
  * 6 guard-removal fractions within their published tolerance.

The tolerances were measured, not chosen: each is 5 sigma of the reference estimator's own sampling error
over 10 independent worlds, floored at 0.030 / 0.040.  `authoring/certificate.json` records the scan.
"""
import json, os, sys

TESTS = os.environ.get("TESTS", "/tests")
APP = os.environ.get("APP", "/app")
REWARD_DIR = os.environ.get("REWARD_DIR", "/logs/verifier")
REASONS = {"no_surviving_run", "combination_never_launched", "level_never_launched"}


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
    path = os.path.join(APP, "answers.json")
    if not os.path.exists(path):
        finish(False, {"error": "no /app/answers.json"})
    try:
        sub = json.load(open(path))
    except Exception as e:
        finish(False, {"error": "answers.json is not valid JSON: %s" % e})
    if not isinstance(sub, dict):
        finish(False, {"error": "answers.json must be a JSON object"})

    qs = sub.get("queries") or {}
    cs = sub.get("censoring") or {}
    bad, detail = [], {}
    if not isinstance(qs, dict) or not isinstance(cs, dict):
        finish(False, {"error": "`queries` and `censoring` must be JSON objects"})

    for qid in sorted(key["queries"]):
        want = key["queries"][qid]
        got = qs.get(qid)
        if not isinstance(got, dict):
            bad.append({"item": qid, "why": "missing or malformed"})
            continue
        v = got.get("verdict")
        if v not in ("identified", "underdetermined"):
            bad.append({"item": qid, "why": "verdict must be 'identified' or 'underdetermined'", "got": v})
            continue
        if v != want["verdict"]:
            bad.append({"item": qid, "why": "wrong verdict", "got": v})
            continue
        if want["verdict"] == "underdetermined":
            r = got.get("reason")
            if r not in REASONS:
                bad.append({"item": qid, "why": "reason must be one of %s" % sorted(REASONS), "got": r})
            elif r != want["reason"]:
                bad.append({"item": qid, "why": "wrong reason code", "got": r})
            continue
        try:
            d = num(got.get("delta"))
        except ValueError as e:
            bad.append({"item": qid, "why": str(e)})
            continue
        err = abs(d - want["delta"])
        detail[qid] = round(err, 5)
        if err > want["tol"] + 1e-12:
            bad.append({"item": qid, "why": "outside tolerance", "abs_error": round(err, 5),
                        "tol": want["tol"]})

    for rid in sorted(key["censoring"]):
        want = key["censoring"][rid]
        try:
            g = num(cs.get(rid))
        except ValueError as e:
            bad.append({"item": rid, "why": str(e)})
            continue
        if not (0.0 <= g <= 1.0):
            bad.append({"item": rid, "why": "a fraction must lie in [0, 1]", "got": g})
            continue
        err = abs(g - want["value"])
        detail[rid] = round(err, 5)
        if err > want["tol"] + 1e-12:
            bad.append({"item": rid, "why": "outside tolerance", "abs_error": round(err, 5),
                        "tol": want["tol"]})

    n_items = len(key["queries"]) + len(key["censoring"])
    finish(not bad, {"n_items": n_items, "n_failed": len(bad), "failed": bad[:12],
                     "abs_errors": detail})


if __name__ == "__main__":
    main()
