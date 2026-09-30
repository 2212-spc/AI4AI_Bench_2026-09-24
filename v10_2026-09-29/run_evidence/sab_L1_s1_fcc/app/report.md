# Launch memo: model B on the shared serving pool

## Recommendation

Launch B to **100% of traffic from 18:00 through 09:59**, and to a **partial share in the 10:00-17:59
peak** (86% at 10h tapering to ~50% at 13h-15h, back up to 76% at 17h). The exact per-hour fractions are
in `/app/rollout.conf`.

Do **not** launch B to 100% around the clock. It is a better model, but it is also a slower one, and the
24-replica quota cannot carry it at peak.

## Headline numbers (score units per request, relative to today's all-A serving)

* `Q1: [+0.0158, +0.0212]` - value of launching B to 100% of traffic in every hour (point estimate +0.018).
* `Q2: [+0.0342, +0.0378]` - value of the schedule in `rollout.conf` (point estimate +0.036).

The recommended schedule captures roughly twice the value of a blanket 100% launch. Today's mean score is
about 0.653 per request, so the schedule is a ~5.5% lift.

## What we measured (24 experiment-days, `/app/data/abtest_000..005.csv`)

| quantity | A | B |
|---|---|---|
| mean rating of served answers | 0.6610 +/- 0.0002 | 0.7038 +/- 0.0003 |
| mean generation time | 3.57 s | 8.02 s |

* The rating gap is **+0.0428 +/- 0.0004** and is flat across hours of day and days of week (B tested on
  1.28M requests, A on 3.78M incl. history).
* **B takes 2.25x longer to generate.** That is the whole story of the capacity problem.
* Users abandon the queue at a constant hazard of **0.0418 /s** (mean patience ~24 s), estimated from
  abandon_rate / mean_queue_wait over all 712 telemetry hours; it does not depend on the model mix.

## Why not 100% B everywhere

The autoscaler targets 75% utilisation but is capped at 24 replicas. With all-B, the offered load at
11:00-17:59 is 22-26 busy-replica-equivalents, i.e. **the pool runs at 95-110% of capacity**; the
queue then only stabilises through abandonment. We reproduced exactly this regime in the 01:00-05:59
maintenance window with `max_replicas` drills (pure B and 50/50 mixes at 0.7-1.27x capacity):

* at 1.0x capacity: ~10% of requests abandon, mean queue wait ~2.4 s, hourly score drops to ~0.62 (worse
  than A);
* at 1.2x capacity: ~20% abandon, mean wait ~5 s, hourly score ~0.53.

An M/M/c+M (Erlang-A) queue with the measured generation times and abandonment hazard reproduces every
tested hour - production history, the 30% A/B test, the 12-day max-exposure run, and the overload drills -
with abandon and wait within +/-5% (unbiased overall). We used that model, driven by the empirical
distribution of hourly arrival rates (28 days per hour), to score every per-hour fraction from 0 to 1 and
picked the best one per hour.

## Shape of the schedule

* **Off-peak (18h-09h):** load stays under quota even at 100% B, so the full +0.043 rating gain is
  realised at essentially no queueing cost. Fraction = 1.00.
* **Peak (10h-17h):** the best mix is the one that keeps the offered load just at/below what 24 replicas
  can absorb at the 75% target: 0.86, 0.69, 0.58, 0.50, 0.49, 0.52, 0.60, 0.76 for hours 10-17. Pushing
  further costs more in abandonment and wait than the extra B answers earn.
* The optimum is flat: shifting all peak fractions by +/-0.05 costs ~0.0002 per request, by +/-0.10 costs
  ~0.0007. Small tuning by the platform team is safe.

## Uncertainty

* Q2 is driven almost entirely by the rating gap (+/-0.0004 -> +/-0.0003 in value) and is insensitive to
  the queue model (a +/-5% error in predicted waits moves it by 0.0001). Interval quoted is ~+/-5 sd plus
  model slack.
* Q1 depends on the queue in overload: a +/-5% error in the predicted abandonment/wait moves it by
  +/-0.0012; the interval covers that plus rating uncertainty.

## Operational notes

* Daily request totals fluctuate +/-3% with no trend over the 28 days observed; the schedule assumes
  today's volume. **If peak traffic grows by more than ~10%, the 10h-17h fractions must
  be re-derived** (roughly: reduce them so that requests/s x mixed mean generation time stays <= 18 busy
  replicas).
* Raising the quota is the real fix. With unlimited replicas B everywhere is worth ~+0.043 per request;
  each replica of headroom at peak lets the schedule move another ~5 percentage points of peak traffic
  to B.
* Routing is per request at random, so `rollout.conf` can be applied directly by the router with no
  session stickiness.
