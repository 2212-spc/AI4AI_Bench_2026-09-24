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


# ------------------------------------------------------------------ the three v4 forms
def cex(qid, text, claim, schema, entailed, **meta):
    """Counterexample witness.  The agent submits a world; the grader checks consistency and flip.

    schema: {"params": {name: {"path": [...], "lo": x, "hi": y, "type": "float"|"int"|"choice"}}}
    entailed: certified at build time - True iff no consistent world falsifies the claim (gate G14).
    """
    return dict(id=qid, kind="cex", text=text, claim=claim, schema=schema, entailed=bool(entailed),
                key={"verdict": "entailed" if entailed else "refutable"}, **meta)


def prereg(qid, text, max_runs, budget, worlds, **meta):
    """Pre-registered plan.  `worlds` is the build-time set of consistent worlds the plan is tested in;
    each entry is {"name", "params", "label", "salts"} and `label` is that world's correct conclusion."""
    return dict(id=qid, kind="prereg", text=text, max_runs=int(max_runs), budget=float(budget),
                worlds=worlds, key={"labels": sorted({w["label"] for w in worlds})}, **meta)


def audit(qid, text, defect, sites, lo, hi, max_width, **meta):
    """Defect audit.  `sites` is a list of [file, first_line, last_line] spans that count as the defect's
    location; [lo, hi] is the accepted range for the corrected number."""
    return dict(id=qid, kind="audit", text=text, defect=defect, sites=sites,
                key={"defect": defect, "lo": lo, "hi": hi}, lo=lo, hi=hi, max_width=max_width, **meta)



def grade_item(it, ans, ctx=None):
    """ctx (only needed by the v4 forms): see scalelab.ctx.make_ctx.
    {"backend", "base", "rows", "claim_fn", "session_for", "cost_fn"}"""
    if ans is None:
        return False, "missing"
    k = it["kind"]
    if k in ("cex", "prereg", "audit"):
        from . import verify as V
        if k == "audit":
            r = V.grade_audit(it, ans)
        elif ctx is None:
            return False, "grader context unavailable for a %s item" % k
        elif k == "cex":
            cf = ctx["claim_fn"]
            if isinstance(cf, dict):
                fn = cf.get(it["id"])
            elif cf is None:
                fn = None
            else:
                fn = lambda p, _q=it["id"]: cf(_q, p)
            if fn is None:
                return False, "no claim function for %s" % it["id"]
            r = V.grade_cex(it, ans, ctx["backend"], ctx["base"], ctx["rows"], fn)
        else:
            item = dict(it); item["_cost_fn"] = ctx["cost_fn"]
            r = V.grade_prereg(item, ans, it["worlds"], ctx["session_for"])
        return bool(r["pass"]), r.get("reason", "")
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


def grade(items, answers, ctx=None):
    res = {}
    for it in items:
        ok, why = grade_item(it, (answers or {}).get(it["id"]), ctx)
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
    elif it["kind"] == "cex":
        q["answer_format"] = {"verdict": "refutable | entailed",
                              "witness": {k: ("one of %s" % v["values"]) if v.get("type") == "choice"
                                          else "number in [%g, %g]" % (v["lo"], v["hi"])
                                          for k, v in it["schema"]["params"].items()}}
        q["claim"] = it["claim"]
    elif it["kind"] == "prereg":
        q["answer_format"] = {"runs": "list of lab requests, each optionally with a \"label\"",
                              "rule": {"expr": "arithmetic over <label>.<field>",
                                       "cuts": "[[op, threshold, label], ..., [\"else\", label]]"}}
        q["limits"] = {"max_runs": it["max_runs"], "budget": it["budget"],
                       "allowed_labels": it["key"]["labels"]}
    elif it["kind"] == "audit":
        q["answer_format"] = {"defect": "one defect id from the taxonomy in manual.md Section 6",
                              "site": "file:line", "corrected": {"lo": "number", "hi": "number"}}
    else:
        q["answer_format"] = {"verdict": "supported | refuted | undetermined"}
    return q
