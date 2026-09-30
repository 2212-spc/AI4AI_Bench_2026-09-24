# Launch memo: model B on the shared serving pool

## Recommendation

Launch B on a **time-of-day schedule** (`/app/rollout.conf`): 100% B overnight and in the shoulders
(21:00-09:00), tapering to roughly 35% B at the mid-afternoon peak (14:00-16:00).  Do **not** launch B to
100% of traffic in every hour - that is a net loss versus today.

* `Q1: [-0.0266, -0.0208]`  (value of 100% B in every hour; point estimate -0.0238 per request)
* `Q2: [+0.0181, +0.0217]`  (value of the schedule in rollout.conf; point estimate +0.0199 per request)

## Why

B's answers are better (mean rating 0.6584 vs 0.6332 for A, +0.0252, s.e. 0.0004 - measured on 1.3M B
requests over 19 experiment-days) but B takes **2.36x longer to generate** (7.24 s vs 3.07 s mean).  The
autoscaler compensates only up to the **16-replica quota**.  All-B needs 16 replicas from about 10:00 and
is overloaded (offered load > capacity) from roughly 12:00 to 18:00: at 15:00 the pool would run at 117%
load, queues explode, and abandonment plus wait penalty (0.008/s) swamp the rating gain.  Overnight the
pool has room, so B is a clean +0.025-0.034 per request there.

The schedule keeps every hour's offered load inside the quota with the mix that maximises rating gain
minus queueing cost.  Only hours 10:00-20:00 are throttled below 100%.

## Method

1. Served system modelled as an M/M/c+M (Erlang-A) queue per hour: Poisson arrivals, exponential service
   with the mix's mean, FIFO, constant abandonment hazard.  The hazard is unmeasured, so it was estimated
   from telemetry as abandon_rate / mean_queue_wait: **theta = 0.0218 /s** (mean patience ~46 s), stable
   across 812 hour-rows including deliberate overload drills.
2. Model validation: predicted abandonment and mean wait match the 14 history days, 19 days of A/B traffic
   (100% B at night, 30% B by day) and three capacity drills (B at 100% capped at 6-7 replicas, 50/50 mix
   capped at 5) to within ~2% in aggregate, including at 98% utilisation.  Predicted per-arm scores match
   observed to within 0.0001.  A final 1-day run of the (guardrail-clipped) schedule matched the model's
   score prediction to 0.0001.
3. Arrival rates: per-hour means from 34 days, with day-to-day multiplicative variation (sd 3%) carried
   through the queue model (overload cost is convex, so this matters for Q1).
4. The schedule is the per-hour maximiser of expected score on a 0.005 grid.  Regret against the optimum
   under +-2 s.e. shifts of every fitted parameter is < 0.0001, so the schedule is robust.
5. Intervals: parametric uncertainty (ratings, generation times, hazard) propagated by Monte Carlo, widened
   for residual model/arrival-mix error.  Q1 is more uncertain because it depends on the overloaded regime,
   where the score is very sensitive to B's generation time and the hazard.

## Notes for platform

* Peak-hour capacity is the binding constraint.  Raising the quota (or reducing B's latency) would let the
  afternoon fraction rise; each replica at peak is worth roughly +0.001 per request across the day.
* Guardrail: the schedule routes >30% to B in 08:00-09:59 and 19:00-21:59; those hours were validated at
  30% (experiment) and by model, not at the launch fraction.  Expected values there sit well inside the
  capacity margin (offered load <= 14 replicas at 09:00, 16 at 20:00).
