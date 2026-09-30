"""Item types, keys and grading.

Every numeric question is answered with an interval {"lo": x, "hi": y}.  The key is the identification
set [a, b] of the queried quantity: a == b when the lab plus the manual's guarantees pin the value down
(up to seed noise), a < b when it depends on a documented known-unknown with zero footprint in the
runnable region.  Pass iff |lo - a| <= T and |hi - b| <= T.  Point-collapse on a set item and a
confidence-interval-style answer on a point item therefore both fail, by construction (set items are
required to have b - a >= 4T; see gates.py).

decision: {"choice": option}; pass iff regret(choice) <= r_tol (regret computed on the noise-free world).
verdict:  {"verdict": "supported" | "refuted" | "undetermined"} relative to the identification set.
"""
import math

VERDICTS = ("supported", "refuted", "undetermined")


def point(qid, text, unit, value, tol, **meta):
    return dict(id=qid, kind="point", text=text, unit=unit, key={"lo": value, "hi": value}, tol=tol, **meta)


def interval(qid, text, unit, lo, hi, tol, **meta):
    return dict(id=qid, kind="set" if hi - lo > 1e-12 else "point", text=text, unit=unit,
                key={"lo": lo, "hi": hi}, tol=tol, **meta)


def decision(qid, text, options, losses, r_tol, **meta):
    best = min(losses.values())
    reg = {k: v - best for k, v in losses.items()}
    choice = min(losses, key=losses.get)
    return dict(id=qid, kind="decision", text=text, options=list(options), key={"choice": choice, "regret": reg},
                r_tol=r_tol, **meta)


def verdict(qid, text, holds_over_set, **meta):
    """holds_over_set: list of booleans - the claim's truth value at every member of the identification set
    (dense grid over the known-unknown range, endpoints included)."""
    v = "supported" if all(holds_over_set) else ("refuted" if not any(holds_over_set) else "undetermined")
    return dict(id=qid, kind="verdict", text=text, key={"verdict": v}, **meta)


def grade_item(it, ans):
    if ans is None:
        return False, "missing"
    k = it["kind"]
    try:
        if k in ("point", "set"):
            lo = float(ans["lo"]); hi = float(ans["hi"])
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
                return False, "malformed interval"
            e1 = abs(lo - it["key"]["lo"]); e2 = abs(hi - it["key"]["hi"])
            ok = e1 <= it["tol"] and e2 <= it["tol"]
            return ok, "endpoint errors %.4g / %.4g vs tol %.4g" % (e1, e2, it["tol"])
        if k == "decision":
            c = ans["choice"]
            if c not in it["key"]["regret"]:
                return False, "unknown option %r" % (c,)
            r = it["key"]["regret"][c]
            return r <= it["r_tol"], "regret %.4g vs tol %.4g" % (r, it["r_tol"])
        if k == "verdict":
            v = str(ans["verdict"]).strip().lower()
            return v == it["key"]["verdict"], "answered %s" % v
    except (KeyError, TypeError, ValueError) as e:
        return False, "malformed: %r" % (e,)
    return False, "unknown kind"


def grade(items, answers):
    res = {}
    for it in items:
        ok, why = grade_item(it, (answers or {}).get(it["id"]))
        res[it["id"]] = {"pass": bool(ok), "why": why, "kind": it["kind"]}
    n = sum(r["pass"] for r in res.values())
    return {"items": res, "score": n / max(1, len(items)), "n_pass": n, "n_items": len(items),
            "all_pass": n == len(items)}


def public_view(it):
    """What the agent sees for one question (no kind, no key)."""
    q = {"id": it["id"], "question": it["text"]}
    if it["kind"] in ("point", "set"):
        q["answer_format"] = {"lo": "number", "hi": "number"}; q["unit"] = it["unit"]
    elif it["kind"] == "decision":
        q["answer_format"] = {"choice": "one of " + ", ".join(map(str, it["options"]))}
    else:
        q["answer_format"] = {"verdict": "supported | refuted | undetermined"}
    return q
