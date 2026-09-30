"""Grading context for the v4 item forms.

`queries.grade` needs more than a key for `cex` and `prereg`: it needs the lab backend, the true world,
the rows a witness has to reproduce, the claim as a function of a world, and a way to open a session in
*another* world.  Building that in one place means the build-time gates, the exported grader and the
smoke tests all grade with identical code - the property v3 relied on for the pretraining forms, where
the oracle, the rivals and the gates all went through `Session.run`.

Blueprint protocol (only for blueprints that emit `cex` items):

    claim_fn(qid, p) -> bool        does the claim of question `qid` hold in world `p`?

The world handed to `claim_fn` is always a complete parameter set (`backend.full`), with only the
question's declared free parameters moved away from the truth.
"""
import json
from .lab import Session, caps_of
from . import labs


def ledger_rows(path):
    """Every result the agent actually obtained, from a lab server ledger (`{"req","result","t"}` lines)."""
    out = []
    try:
        for line in open(path):
            if line.strip():
                out.append(json.loads(line)["result"])
    except FileNotFoundError:
        return []
    return out


def cost_fn(params, spec, salt=0):
    """Cost one request without executing or charging it.

    Raises LabError on a request the lab would refuse, which `verify.check_plan_shape` reports as a
    plan-shape failure - so an illegal request in a plan is caught before any world is touched."""
    def f(req):
        s = Session(params, spec, salt)
        cfg, seed, extra = s.validate(dict(req))
        return s.cost(cfg, extra)
    return f


def session_maker(spec, budget=None, max_runs=None):
    """(world, salt) -> callable returning a fresh Session in that world.

    The declared budget is enforced by `verify.check_plan_shape` *before* anything is executed, so the
    session used for execution only has to keep the lab's per-request cap; the totals below are a
    backstop, set from the largest allowance any plan item declared.
    """
    c = caps_of(spec)

    def make(world, salt):
        sp = json.loads(json.dumps(spec))
        sp["caps"] = {"run_cost": c["run_cost"],
                      "total_cost": float(budget if budget else c["total_cost"]) * 1.001 + 1.0,
                      "max_runs": int(max_runs if max_runs else c["max_runs"])}
        params = world["params"] if isinstance(world, dict) and "params" in world else world
        return lambda: Session(params, sp, salt)
    return make


def make_ctx(params, spec, salt, rows, items=(), claim_fn=None, ledger=None):
    """Build the ctx dict `queries.grade(items, answers, ctx)` expects.

    rows:     the disclosed notebook rows (result dicts).  With `ledger`, every run the agent made is
              appended - a witness must reproduce its own evidence too.
    claim_fn: `claim_fn(qid, p) -> bool`, or a dict {qid: fn(p)}.
    """
    rows = list(rows or [])
    if ledger:
        rows += ledger_rows(ledger)
    pre = [it for it in items if it.get("kind") == "prereg"]
    budget = max([float(it["budget"]) for it in pre], default=None)
    nruns = max([int(it["max_runs"]) for it in pre], default=None)
    return {"backend": labs.backend(spec.get("lab", "pretrain")),
            "base": params, "rows": rows, "claim_fn": claim_fn,
            "session_for": session_maker(spec, budget, nruns),
            "cost_fn": cost_fn(params, spec, salt)}
