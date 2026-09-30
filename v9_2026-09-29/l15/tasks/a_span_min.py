"""A-SPAN-MIN: the a_span instances with the world model taken OUT of the task text.

Same instances, mechanics, truth, strategies, principles and gates as `a_span`; only the text changes (see
b_control_min for why - v8 frontier round 1: claude-fable-5-1 passed a_span 2/2 by executing the protocol
that the task's "What production traffic is" section and docs/stack.md spelled out: production is a mixture of
the five replayed families in unmeasured proportions, a set's effect is not the combination of its members'
effects, a change that looks worthless alone can be the best buy once another has removed a stage...).

Kept, because they are operational facts an engineer on the team would have: what each op does, its cost and
noise, the canary's one-change limit (a hard operational rule, and without it the task would be a different
task), and that the five replays were recorded from production, one per request family the router
distinguishes.  Removed: the stage model, the statement that production is a mixture, the statement that
shares are unmeasured, and every stated consequence.

DISCOVERABILITY CERTIFICATE (conditions D1-D2 in `instance_gate`, computed from the reference protocol's own
error model).  With the recommended measurements the agent can falsify the two naive readings of the setup:
  D1 "production is (like) one of the families": the stock canary latency differs from every family's stock
     replay latency by >= 3 sigma at 3 canary reps and 1 replay rep;
  D2 "production is their plain average": the seven canary points (stock + each single change) reject the
     equal-weight prediction built from 1-rep replays, chi-square beyond the 3-sigma (0.99865) quantile.
Neither diagnostic is suggested by the text.
"""
import math
import numpy as np
from . import a_span as A
from .a_span import (NSTAGE, CHANGES, TRACES, BUDGET, TRACE_COST, CANARY_COST, CANARY_MAX_OPTS, MAX_REPS,
                     TAU_GAIN, WIDTH_CAP, WEAK_EPS, CANARY_REPS_REF, SWEEP_REPS, REFINE_REPS, N_CANARY_DESIGN,
                     sample_params, factors, latency, speedup, truth, STRATEGIES, NOISY_FAIL, SEARCH,
                     PRINCIPLES, MUTATE)

_CHI2_7_3SIG = 23.5       # chi-square(7) quantile at 0.99865


class World(A.World):
    NAME = "a_span_min"


def _disc(p):
    sig = p["sig"]
    prod0 = latency(p, [], "prod")
    # D1
    z1 = min(abs(math.log(prod0 / latency(p, [], T))) / (sig * math.sqrt(1.0 / 3 + 1.0)) for T in TRACES)
    # D2
    chi = 0.0
    for S in [[]] + [[c] for c in CHANGES]:
        Lp = latency(p, S, "prod")
        Lk = [latency(p, S, T) for T in TRACES]
        pred = sum(Lk) / len(TRACES)
        var = (sig * Lp) ** 2 / 3 + sum((sig * x / len(TRACES)) ** 2 for x in Lk)
        chi += (pred - Lp) ** 2 / var
    return [z1 >= 3.0, chi >= _CHI2_7_3SIG], {"disc_z_not_one_family": round(z1, 2),
                                               "disc_chi2_not_uniform": round(chi, 1)}


def instance_gate(p):
    ok, info = A.instance_gate(p)
    c2, info2 = _disc(p)
    info = dict(info)
    info.update(info2)
    info["c_min"] = [bool(x) for x in c2]
    return bool(ok and all(c2)), info


def pool_gate(seeds):
    return A.pool_gate([s for s in seeds if instance_gate(sample_params(s))[0]])


def instruction(p, t):
    cat = "\n".join("| `%s` | %d |" % (c, p["cost"][c]) for c in CHANGES)
    return """# Serving-stack rollout plan for production traffic

You own the inference serving stack.  Six changes are queued; each costs review-days out of this quarter's
budget, and you may ship any subset whose total review cost is at most **%d review-days**.

| change | review-days |
|---|---|
%s

Your objective is **mean latency on live production traffic**.

You have **%g lab credits**.

* `lab replay opts=<ids> trace=<%s> reps=<1..%d>` - replay a recorded trace through the stack with those
  changes applied.  **reps credits.**  Any set of changes, any trace.  The five traces were recorded from
  production, one per request family the router distinguishes.
* `lab canary opts=<ids> reps=<1..%d>` - serve live production traffic with those changes applied.
  **reps x %g credits.**  The canary rollout gate admits **at most %d change per window** - a hard
  operational rule, not a budget matter: `opts` may name one change or none, and a larger set cannot be put
  in front of production traffic before the plan is approved.

Every measurement carries **+-%.2f%%** independent multiplicative noise (mean zero), and repeats average it
down.  There is no other randomness in this task.

## Deliverable: `/app/plan.json`

```json
{"opts": ["%s", "%s"],
 "speedup": {"lo": 1.00, "hi": 1.00},
 "weakest": "%s"}
```

* `opts` - the set you will ship.  Graded on the fraction of the achievable production gain it captures;
  you need at least **%.0f%%**.  Total review cost must be within the cap.
* `speedup` - an interval for the production speedup (stock latency / your set's latency) **of the set you
  yourself named**.  Relative width `(hi-lo)/truth` may not exceed **%.0f%%**.
* `weakest` - the member of your own set whose removal would cost the least production speedup, with the
  rest of your set still in place.
""" % (p["cap"], cat, BUDGET, "|".join(TRACES), MAX_REPS, MAX_REPS, CANARY_COST, CANARY_MAX_OPTS,
       100 * p["sig"], CHANGES[0], CHANGES[2], CHANGES[0], 100 * TAU_GAIN, 100 * WIDTH_CAP)


def docs(p):
    return {"docs/plan_format.md": A.docs(p)["docs/plan_format.md"]}


def hints(p):
    return A.hints(p)
