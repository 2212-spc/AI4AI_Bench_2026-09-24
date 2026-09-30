# Launch memo: model B on the shared serving pool

**Recommendation:** launch B to 100% of traffic in every hour except 10:00-17:59, where B gets a
partial share (82% at 10:00 falling to 45% at 14:00 and back to 66% at 17:00). The schedule is in
`/app/rollout.conf`. It captures essentially all of the quality gain that the 16-replica quota allows.

Q1: [-0.0115, -0.0055]
Q2: [+0.0176, +0.0214]

(Q1 = value of 100% B in every hour; Q2 = value of the schedule in `rollout.conf`. Score units per
request, relative to all-A.)

## What we learned (24 experiment-days, 5 runs)

| quantity | A | B (hours 0-8, 18-23) | B (hours 9-17) |
|---|---|---|---|
| mean rating of served answers | 0.6421 (se 0.0002) | 0.6602 (se 0.0004) | 0.6762 (se 0.0006) |
| mean generation time | 2.536 s | 4.67 s | 6.04 s |

* B is better rated in every hour: +0.018 off-peak, +0.034 during business hours (9-17). Neither the
  A rating nor the B rating varies inside those two bands (chi-square tests pass), and ratings do not
  depend on how long the user queued.
* B is about twice as slow per answer as A, and slower still on business-hour traffic. That is the
  whole story of this launch: quality gain vs. GPU capacity.
* Users abandon the queue at a constant hazard of 0.01686 per second waiting (mean patience ~59 s).
  This came from the exact identity abandon_rate = hazard x mean_queue_wait, which holds in every one
  of the 817 observed (day, hour) rows.
* The pool behaves as an Erlang-A (M/M/c+M) queue using the mean generation time of the traffic mix.
  Predictions matched production hours and two deliberate overload drills (5 replicas at night, up to
  30% abandonment and 17 s queue waits) within the stated 10% measurement noise.
* Arrivals: Poisson around an hour-of-day profile times a day factor (weekdays ~1.05-1.11, weekends
  ~0.83, with ~5% extra day-to-day spread). No additional hourly noise.

## Why not 100% B everywhere

The autoscaler runs ceil(load / 0.75) replicas, capped at the quota of 16. With A only, the busiest
hour needs about 10 replicas. With B only, business-hour load (2.8 req/s x 6.04 s ~ 17) exceeds what
16 replicas can serve at all: the queue saturates, ~5% of requests abandon and the survivors wait
several seconds. In hours 12-16 the wait penalty alone (0.0112/s) wipes out the +0.034 rating gain
several times over, so 100% B is a net loss for the day (Q1 above is negative), even though B is the
better model.

The fix is to cap B's share in hours 10-17 so the mixed offered load stays inside the quota on the
heaviest weekdays. Each hour's value curve is flat for about +/-0.07 around the chosen fraction, and
re-optimising under every plausible perturbation of the fitted parameters (hazard, generation times,
rating deltas, arrival distribution) moves the schedule's value by less than 0.0001. If the quota is
ever raised to ~20 replicas, hours 10-17 can go to 100% B as well, worth roughly another +0.007 per
request over the day.

## Value of the schedule

Central estimate +0.0195 per request (~+3% on a baseline mean score of 0.634). The Q2 interval covers
the statistical uncertainty in the four rating means, the generation-time means, the abandonment
hazard, and the day-to-day arrival distribution (empirical 33 days vs. fitted parametric models).

The Q1 interval is wider because the all-B value is dominated by how often business-hour load sits
above the quota, which is sensitive to the spread of the day factors and to B's peak generation time.

## Experiment log

1. Days 15-16: max-allowed schedule (100% B off-peak, 30% B 08-21), 2 days.
2. Days 17-19: night capacity drill, replicas capped at 5, B at 50% and 100% (overload calibration).
3. Days 20-24: max-allowed schedule, 5 days.
4. Days 25-26: night drill, cap 5, B at 50%/70%/100% (mixed-service queue check).
5. Days 27-38: max-allowed schedule, 12 days.

Raw telemetry: `/app/data/abtest_000.csv` ... `abtest_004.csv`. Model and fitting code: `/app/model.py`.
