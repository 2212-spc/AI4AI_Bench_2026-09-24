# Launch memo: model B on the shared serving pool

**Recommendation:** launch B to 100% of traffic in every hour except 11:00-17:59, where B gets
52-96% of traffic (exact schedule in `/app/rollout.conf`).  Expected gain: about +0.027 score
points per request versus all-A.  Launching B to 100% everywhere would gain essentially nothing
(roughly -0.001), because peak-hour load would exceed the 24-replica quota and abandonment would
wipe out B's quality advantage.

## Headline numbers

* `Q1: [-0.0042, +0.0018]` - value of routing 100% of traffic to B in every hour.
* `Q2: [+0.0249, +0.0289]` - value of the schedule in `rollout.conf`.

(Score units per request, relative to all-A.  Point estimates: Q1 about -0.0012, Q2 about +0.0269.)

## What we learned about B (24 experiment-days, 1.9M B-served requests)

| | A | B, off-peak (18:00-08:59) | B, peak (09:00-17:59) |
|---|---|---|---|
| mean rating of served answers | 0.6206 | 0.6436 | 0.6670 |
| mean generation time | 2.47 s | 4.20 s | 5.75 s |

* B's rating uplift is +0.023 off-peak and +0.046 at peak.  Ratings and generation times did not
  depend on the fraction routed (tested 15%, 30%, 50%, 100%), only on the hour block.
* B is roughly 1.7x (off-peak) to 2.3x (peak) more expensive in GPU-seconds per request than A.
* The abandonment hazard is constant at 0.0127 per second of queueing (mean patience about 79 s),
  measured from the abandon_rate / mean_queue_wait ratio over 855 hour-rows.

## Why the schedule caps B at peak

The autoscaler sizes the pool to 75% utilization but cannot exceed 24 replicas.  At 100% B the
offered load in 11:00-17:59 reaches 24-30 replica-equivalents on a weekday, so the pool runs at
utilization 0.9-1.25, abandonment climbs to 5-20% and mean queueing wait to 4-15 s.  Each abandoned
request loses its whole rating (about 0.65) plus 0.0098/s of wait, so the per-request loss from
overload dwarfs B's +0.046 rating uplift.

The schedule therefore routes to B only as much peak traffic as keeps the pool near the 24-replica
ceiling at moderate utilization (about 0.85 on a Thursday, our busiest weekday).  The value curve
is flat near the optimum: moving any peak-hour fraction by +/-0.05 costs under 0.0003 per request,
so the schedule is robust to the remaining estimation error.

Per-hour plan (fraction to B): 00-10 -> 1.00; 11 -> 0.96; 12 -> 0.80; 13 -> 0.67; 14 -> 0.56;
15 -> 0.54; 16 -> 0.52; 17 -> 0.59; 18-23 -> 1.00.

Decomposition of the long-run mean score per request (model, averaged over the weekly traffic
pattern and day-to-day noise):

| | all-A | rollout | all-B |
|---|---|---|---|
| mean rating of the served mix | 0.6206 | 0.6485 | 0.6551 |
| loss from abandoned requests | 0.0016 | 0.0020 | 0.0180 |
| queue-wait penalty | 0.0020 | 0.0024 | 0.0209 |
| **score** | **0.6171** | **0.6440** | **0.6162** |

## How the numbers were produced

1. **Traffic model.**  Hourly arrivals = hour profile x day multiplier.  Day multipliers follow the
   weekday (Mon 1.01, Tue 1.07, Wed 1.04, Thu 1.10, Fri 1.10, Sat 0.78, Sun 0.75) with about 5%
   log-normal day-to-day noise, fitted on 38 days (14 history + 24 experiment).
2. **Queue model.**  Each hour is a steady-state M/M/c+M (Erlang-A) queue with the mix's mean
   generation time, c from the autoscaler rule, hazard 0.0127/s.  It reproduces the observed
   abandonment and waits in production and in all experiments, including three nightly capacity
   drills that pushed utilization to 0.9-1.3 (observed/predicted abandonment ratio 0.98 +/- 0.03).
   Over the 504 experiment hour-rows the model's mean score differs from the observed one by
   +0.0001 per request.
3. **Value.**  Long-run mean score = request-weighted average over hours and over the weekday /
   noise distribution of (rating of served mix) x (1 - P_abandon) - 0.0098 x E[queue wait], minus
   the same quantity for all-A.  The schedule is the per-hour maximizer of this value (0.01 grid).
4. **Intervals.**  Parametric Monte Carlo (160 draws) over the uncertainty in weekday multipliers,
   noise level, per-hour ratings and generation times, and a 3% queue-model calibration error.
   Q2's standard error is 0.0004 (the interval is about +/-5 SE).  Q1's standard error is 0.0026,
   driven almost entirely by the +/-0.8% uncertainty in the average traffic level: under all-B the
   peak hours are deep in overload, where value moves by about 0.003 per 1% of traffic.  The Q1
   interval uses the full allowed width and is roughly a 75% interval.

## Operational notes

* Night hours 00-08 and 18-23 need 10-24 replicas at 100% B (vs 6-16 today); the autoscaler
  handles this within quota.
* Peak hours sit at the 24-replica ceiling.  If traffic grows by more than about 5%, lower the
  11:00-17:59 fractions proportionally (each +1% traffic is roughly -0.01 to -0.02 in fraction),
  otherwise the protection against overload is lost.  If the quota is ever raised, B can go to
  100% at peak once the quota reaches about 34 replicas.
* Experiment files: `/app/data/abtest_000.csv` .. `abtest_007.csv` (`002`, `003`, `006` are the
  capacity drills in the 01:00-05:59 maintenance window).
