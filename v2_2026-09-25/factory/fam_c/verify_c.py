"""Family C verifier.  Grades a fleet-evidence audit: 12 verdicts plus the estimates that are claimed.

Truth comes from the structural model that generated the log, evaluated analytically at authoring time;
the verifier only compares numbers, so it executes no agent code at all.
"""
import json, os, sys

TESTS = os.environ.get("TESTS", "/tests")
APP = os.environ.get("APP", "/app")
REWARD_DIR = os.environ.get("REWARD_DIR", "/logs/verifier")
VERDICTS = {"identified", "underdetermined"}


def finish(ok, detail):
    os.makedirs(REWARD_DIR, exist_ok=True)
    open(os.path.join(REWARD_DIR, "reward.txt"), "w").write("1" if ok else "0")
    print(json.dumps({"reward": 1 if ok else 0, "detail": detail}, default=str)[:20000])
    sys.exit(0)


def main():
    truth = json.load(open(os.path.join(TESTS, "truth.json")))
    path = os.path.join(APP, "answers.json")
    if not os.path.exists(path):
        finish(False, {"error": "missing /app/answers.json"})
    try:
        sub = json.load(open(path))
    except Exception as e:
        finish(False, {"error": "answers.json is not valid JSON: %s" % e})
    if isinstance(sub, dict) and "queries" in sub:
        sub = sub["queries"]
    if not isinstance(sub, dict):
        finish(False, {"error": "answers.json must be an object keyed by query id"})

    errs, reason_errs = [], []
    for qid, t in sorted(truth.items()):
        a = sub.get(qid)
        if not isinstance(a, dict):
            errs.append({"q": qid, "why": "no answer object"}); continue
        v = a.get("verdict")
        if v not in VERDICTS:
            errs.append({"q": qid, "why": "verdict not in %s" % sorted(VERDICTS), "got": v}); continue
        if v != t["verdict"]:
            errs.append({"q": qid, "why": "wrong verdict", "got": v}); continue
        if t["verdict"] == "identified":
            d = a.get("delta", a.get("estimate"))
            try:
                d = float(d)
            except (TypeError, ValueError):
                errs.append({"q": qid, "why": "identified but no numeric delta"}); continue
            if abs(d - t["delta"]) > t["tol"]:
                errs.append({"q": qid, "why": "delta outside tolerance", "got": round(d, 6),
                             "tol": t["tol"]})
        else:
            r = a.get("reason")
            if r not in t["reasons_ok"]:
                reason_errs.append({"q": qid, "why": "reason not in %s" % t["reasons_ok"], "got": r})

    extra = sorted(set(sub) - set(truth))
    audit = os.path.exists(os.path.join(APP, "audit.md"))
    det = {"n_queries": len(truth), "verdict_or_estimate_errors": errs[:12],
           "reason_errors": reason_errs[:6], "unknown_query_ids": extra[:6], "audit_md_present": audit,
           "n_correct": len(truth) - len({e["q"] for e in errs})}
    finish(not errs and not reason_errs and not extra and audit, det)


if __name__ == "__main__":
    main()
