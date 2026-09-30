"""Family V verifier.

For every claim the submission either supplies a witness or asserts the claim holds.  A witness counts
only if (a) it parses, (b) it is inside the claim's published scope and (c) an independent predicate
confirms the violation while running the PRISTINE modules shipped with the tests - the agent's copy under
/app/dataprep is never imported.  A confirmed witness is always graded correct, even for a claim the key
records as holding, so a genuine discovery can never be punished.
"""
import json, os, sys

TESTS = os.environ.get("TESTS", "/tests")
APP = os.environ.get("APP", "/app")
REWARD_DIR = os.environ.get("REWARD_DIR", "/logs/verifier")
MAX_WITNESS_BYTES = 2 * 1024 * 1024
sys.path.insert(0, TESTS)
import claims as C                                                                # noqa: E402


def finish(ok, detail):
    os.makedirs(REWARD_DIR, exist_ok=True)
    open(os.path.join(REWARD_DIR, "reward.txt"), "w").write("1" if ok else "0")
    print(json.dumps({"reward": 1 if ok else 0, "detail": detail}, default=str)[:20000])
    sys.exit(0)


def load_witness(rel):
    if not isinstance(rel, str) or not rel:
        raise ValueError("witness must be a relative path string")
    p = os.path.realpath(os.path.join(APP, rel))
    if not p.startswith(os.path.realpath(APP) + os.sep):
        raise ValueError("witness path escapes /app")
    if not os.path.exists(p):
        raise ValueError("witness file %s does not exist" % rel)
    if os.path.getsize(p) > MAX_WITNESS_BYTES:
        raise ValueError("witness file larger than 2 MiB")
    return json.load(open(p))


def main():
    key = json.load(open(os.path.join(TESTS, "key.json")))
    path = os.path.join(APP, "findings.json")
    if not os.path.exists(path):
        finish(False, {"error": "missing /app/findings.json"})
    try:
        sub = json.load(open(path))
    except Exception as e:
        finish(False, {"error": "findings.json is not valid JSON: %s" % e})
    if not isinstance(sub, dict):
        finish(False, {"error": "findings.json must be an object keyed by claim id"})

    per, wrong = {}, []
    for cid in sorted(key):
        a = sub.get(cid)
        if not isinstance(a, dict):
            per[cid] = {"ok": False, "why": "no entry"}; wrong.append(cid); continue
        st = a.get("status")
        if st == "violated":
            try:
                wit = load_witness(a.get("witness"))
                confirmed, det = C.CHECK[cid](wit)
            except ValueError as e:
                per[cid] = {"ok": False, "why": "witness rejected: %s" % e}; wrong.append(cid); continue
            except Exception as e:
                per[cid] = {"ok": False, "why": "witness raised %s: %s" % (type(e).__name__, e)}
                wrong.append(cid); continue
            per[cid] = {"ok": bool(confirmed), "why": "confirmed" if confirmed
                        else "the module satisfied the claim on this witness", "check": det}
            if not confirmed:
                wrong.append(cid)
        elif st == "holds":
            arg = a.get("argument") or ""
            if len(arg.strip()) < 80:
                per[cid] = {"ok": False, "why": "an argument of at least 80 characters is required"}
                wrong.append(cid); continue
            ok = (key[cid] == "holds")
            per[cid] = {"ok": ok, "why": "matches key" if ok else "the claim is in fact falsifiable"}
            if not ok:
                wrong.append(cid)
        else:
            per[cid] = {"ok": False, "why": "status must be 'violated' or 'holds', got %r" % st}
            wrong.append(cid)

    finish(not wrong, {"n_claims": len(key), "n_correct": len(key) - len(wrong),
                       "wrong": wrong, "per_claim": per})


if __name__ == "__main__":
    main()
