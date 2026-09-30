# Launch memo: Model B on the shared serving pool

## Recommendation

Launch **Model B to 100% of traffic in every hour**. `rollout.conf` routes fraction 1.0 to B for hours 00-23.

## What we measured (24 experiment-days, calendar days 15-38)

Schedule tested every day: B at 100% in hours 00-07 and 22-23, B at 30% (the guardrail maximum) in hours 08-21.
Together with the 14 days of all-A history this gives ~2.75M B-served requests and ~6.3M A-served requests.

| quantity | A | B |
|---|---|---|
| mean rating of served answers | 0.6402 | 0.6617 |
| mean generation time (s) | 1.696 | 3.92 |

* The rating gain of B is **+0.0215 per served request** (standard error 0.0002). It is the same in every hour
  of the day (per-hour differences +0.019 to +0.024 are within noise), the same on every calendar day, and the
  same whether B carried 30% or 100% of the hour.
* B answers take 2.3x longer to generate, so the autoscaler runs 2.3x the replicas. Peak hour (14:00) needs
  about 24-25 replicas at 100% B, comfortably under the 32-replica quota, so the quota never binds and no hour
  degrades into overload.

## Queueing model

Pool behaviour is well described by an M/M/c+M (Erlang-A) queue with the autoscaler's `ceil(load/0.75)` rule
(min 4). The abandonment hazard fitted from history is **0.0157 per second** (mean patience ~64 s); the model
reproduces observed abandon rates, mean waits and utilization in both the history and all experiment hours
(mean queue cost: observed 0.00344 vs model 0.00346 per request in the experiments).

Under 100% B the queue cost (abandonment + 0.0076/s of waiting) is actually slightly *lower* than under A
(+0.0013 per request), because the larger replica counts pool better at the same 75% target utilization.
Hour-by-hour the value of B is monotonically increasing in the fraction routed to B, so no partial schedule
beats all-B; the nearest alternatives (90% in the tightest hours 02, 03, 22) are worse by 0.00003-0.00006.

## Answers

Value = long-run mean per-request score under the schedule minus all-A. Point estimate for all-B: **+0.0227**
(rating gain +0.0215 weighted by traffic, plus +0.0013 lower queue cost). Uncertainty is dominated by the
rating difference (SE 0.0002); sensitivity of the queue term to the fitted hazard (0.0145-0.017/s) and to B's
mean generation time (3.90-3.95 s) moves the value by less than 0.0001.

Q1: [+0.0202, +0.0252]

Q2: [+0.0208, +0.0246]

(Q1 and Q2 describe the same schedule, since the recommended rollout is 100% B in every hour.)

## Notes for the platform team

* Expected replica usage at 100% B: 8 at night, ~24-25 at the afternoon peak. Headroom to the quota is
  ~7 replicas (~30% traffic growth at peak) before the autoscaler would be capped.
* If traffic grows beyond that, the peak hours 13-16 are the first to revisit; at the current level there is no
  reason to hold back any hour.
