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

* `lab replay opts=<ids> trace=<R1|R2|R3|R4|R5> reps=<1..6>` - replay a recorded trace through the stack with those
  changes applied.  **reps credits.**  Any set of changes, any trace.  The five traces were recorded from
  production, one per request family the router distinguishes.
* `lab canary opts=<ids> reps=<1..6>` - serve live production traffic with those changes applied.
  **reps x 4 credits.**  The canary rollout gate admits **at most 1 change per window** - a hard
  operational rule, not a budget matter: `opts` may name one change or none, and a larger set cannot be put
  in front of production traffic before the plan is approved.

Every measurement carries **+-0.22%** independent multiplicative noise (mean zero), and repeats average it
down.  There is no other randomness in this task.

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
