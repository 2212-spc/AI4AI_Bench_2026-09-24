# Launch model B on the shared serving pool

Our chat assistant is served from **one pool of GPU replicas** behind **one FIFO request queue**.  Model A
serves all traffic today.  Model B won the offline evaluation and the product team wants it launched.  You
decide the launch: **the fraction of each hour's traffic routed to B**, for every hour of the day (the same
schedule every day of the week).

## How serving works (from the platform team)

* Requests arrive as a Poisson stream; the rate depends on the hour of day and varies from day to day.
  `/app/data/history.csv` has the last 14 days of production telemetry (all traffic on A).
* A replica generates one response at a time.  Generation times are exponentially distributed; each model
  has its own mean.  Routing is per request, at random, with the hour's fraction.
* The autoscaler sets the replica count for each hour to `ceil(offered_load / 0.75)`, at least 4 and at
  most the GPU quota of **24 replicas**.  `offered_load` = that hour's actual arrival rate x the mean
  generation time of the traffic mix actually running in that hour.  The quota cannot be raised.
* A request that finds every replica busy waits in the queue (first come, first served, both models share
  it).  A waiting user leaves (abandons) at a constant hazard - how long they have already waited does not
  change the chance of leaving in the next second - and nobody has measured their mean patience.  An
  abandoned request gets no answer.  Treat each hour as being in steady state.

## The metric

Per-request score = the user's rating of the answer (0..1) if it is served, **0 if abandoned**, minus
**0.0062 per second spent waiting in the queue** (abandoned requests count the time they waited).

The **value** of a launch schedule = the long-run mean per-request score under it (total score over all
requests, over many days, divided by the number of requests) minus the same quantity for all-A.

## Experiments

You have **24 experiment-days**.

* `lab abtest fraction=<f> hours=<which> days=<n> [max_replicas=<k>]` (or `schedule=[24 fractions]` instead
  of `fraction`) - runs live for n consecutive calendar days; each tested hour routes that fraction of its
  traffic to B; hours not listed run production (all A) and are not reported.  Writes per-(day, hour)
  telemetry to a CSV under `/app/data/` and prints a per-arm summary.  Cost: 1 experiment-day per day.
* **Guardrail:** between 08:00 and 21:59 an experiment may route at most **30%** of traffic to B.
* `max_replicas` caps the autoscaler for the tested hours; SRE only allows it inside the nightly
  maintenance window 01:00-05:59 (every tested hour must be in that window).
* Column definitions: `/app/docs/telemetry.md`.  `lab spec`, `lab status`, `lab history` are free.

## Deliverables

1. **`/app/rollout.conf`** - the launch schedule, one line per hour `HH fraction` (24 lines, hours 00-23,
   fraction of that hour's traffic on B, 0..1).  Graded: the value of your schedule must be within
   **0.0010** of the best value any schedule achieves.
2. **`/app/report.md`** - your launch memo, free-form, for the product and platform teams.  It must contain
   exactly one line of each of these two forms (values are score units per request, e.g. `+0.0123`):
   * `Q1: [lo, hi]` - the value of launching B to **100% of traffic in every hour**.  Width <= 0.0060.
   * `Q2: [lo, hi]` - the value of **your** schedule in `rollout.conf`.  Width <= 0.0040.

Graded against the exact expected values; there is no partial credit inside an item.
