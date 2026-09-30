# Serving-stack rollout plan for production traffic

You own the inference serving stack.  Six changes are queued; each costs review-days out of this quarter's
budget, and you may ship any subset whose total review cost is at most **11 review-days**.

| change | review-days |
|---|---|
| `C-114` | 3 |
| `C-127` | 4 |
| `C-203` | 4 |
| `C-241` | 3 |
| `C-318` | 5 |
| `C-352` | 5 |

Your objective is **mean latency on live production traffic**.

You have **320 lab credits**.

* `lab replay opts=<ids> trace=<R1|R2|R3|R4|R5> reps=<1..6>` - replay one request family through the stack with those
  changes applied.  **reps credits.**  Any set of changes, any family.
* `lab canary opts=<ids> reps=<1..6>` - serve live production traffic with those changes applied.
  **reps x 4 credits.**  The canary rollout gate admits **at most 1 change per window**: `opts` may name
  one change, or none.  A larger set cannot be canaried at any price.

Every measurement carries **+-0.22%** independent multiplicative noise (mean zero), and repeats average it
down.  There is no other randomness in this task.

## What production traffic is

Requests are routed to one of five **request families**.  The five trace replays are exactly those families,
one family each, in isolation.  Production traffic is a mixture of the same five families; the routing layer
does not tag requests, so **the share of each family in production traffic is not instrumented** and appears
in no dashboard.

Latency for a given set of changes is the sum over the stack's internal stages of the time spent in each
stage.  A change divides the stages it touches by a fixed factor and adds a small fixed fraction to one
other stage; those factors are properties of the change, not of the workload.  How much of a request's time
lands in each stage is a property of the request family.  Neither the stage breakdown nor the per-change
factors are published, and the vendor's release notes give no per-change numbers.

## Deliverable: `/app/plan.json`

```json
{"opts": ["C-114", "C-203"],
 "speedup": {"lo": 1.00, "hi": 1.00},
 "weakest": "C-114"}
```

* `opts` - the set you will ship.  Graded on the fraction of the achievable production gain it captures;
  you need at least **90%**.  Total review cost must be within the cap.
* `speedup` - an interval for the production speedup (stock latency / your set's latency) **of the set you
  yourself named**.  Relative width `(hi-lo)/truth` may not exceed **6%**.
* `weakest` - the member of your own set whose removal would cost the least production speedup, with the
  rest of your set still in place.
