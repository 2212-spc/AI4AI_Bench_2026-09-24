# Launch memo: model B on the shared serving pool

## Recommendation

Launch B to 100% of traffic in every hour except 10:00-17:59, where B's share is throttled
so the pool stays inside the 16-replica quota on busy days. Schedule (`/app/rollout.conf`):

| hours | fraction to B |
|---|---|
| 00-09, 18-23 | 1.00 |
| 10 | 0.86 |
| 11 | 0.70 |
| 12 | 0.60 |
| 13 | 0.54 |
| 14 | 0.52 |
| 15 | 0.52 |
| 16 | 0.60 |
| 17 | 0.70 |

Q1: [+0.0000, +0.0060]
Q2: [+0.0182, +0.0222]

Q1 is the value (mean per-request score minus all-A) of launching B to 100% in every hour.
Q2 is the value of the schedule above. Point estimates: Q1 about +0.003, Q2 about +0.020.

## Why not 100% everywhere

B answers are rated higher than A (about +0.020 at night, +0.030 in business hours), but B
takes about twice as long to generate: 5.9 s versus 3.2 s for A, and 7.8 s in hours 09-17.
Below the quota the autoscaler absorbs the extra load, and a larger pool at the same target
utilization queues less, so B is strictly better there. Between 10:00 and 17:59, however, all-B
traffic would need 18-25 replicas on a typical weekday and the pool is capped at 16. Utilization
then exceeds 100% on busy days, queue waits grow to tens of seconds, and 10-25% of users abandon.
That wipes out B's rating gain: all-B is worth roughly -0.04 per request at 14:00-15:59 while the
throttled schedule keeps those hours at about +0.016.

The chosen fractions keep the expected offered load in the peak hours just under the quota on the
heavier weekdays (Wed-Fri run about 9% above the daily average, weekends about 18% below).

## What the experiments showed (24 experiment-days)

* 21 days at 100% B overnight (22:00-07:59) and 30% B during 08:00-21:59: B parameters per hour,
  38 daily traffic factors in total (with the 14 history days).
* 2 capacity drills in the maintenance window (100% B with 5 replicas; 60% B with 6 replicas)
  to check the queueing model in overload, including with mixed traffic.

Findings:

* Abandonment hazard is 0.0171 per second (mean patience about 58 s), constant across hours; the
  ratio of abandonment rate to mean queue wait was the same in all 336 history hours.
* The pool behaves as an M/M/c queue with exponential abandonment (Erlang-A), with the mixed
  traffic treated as one exponential service at the mix's mean generation time. Observed
  abandonment, wait and utilization matched this model with no bias (observed/predicted 1.01 on
  average across 500 hour-observations) including at 95-130% load in the drills.
* Traffic has a strong day-of-week pattern (Sat/Sun about 0.82x, Mon/Tue about 1.04x, Wed-Fri about
  1.09x the average) with about 4.5% day-to-day noise inside each weekday; hourly counts are
  Poisson around that.
* B rating: 0.712 in 09:00-17:59, 0.702 otherwise; A rating 0.682. B generation time 7.83 s in
  09:00-17:59, 5.91 s otherwise; A 3.18 s.

## Uncertainty

Q2 is robust: across every model variant tried (day-of-week versus lognormal traffic, abandonment
hazard +/-10%, B generation time +/-0.1 s, within-weekday spread 3-6%) the schedule's value
stayed between +0.0196 and +0.0204, and it was always within 0.0001 of that variant's own optimum.
Q1 is far more sensitive to the traffic tail because all-B saturates the pool on busy afternoons;
the same variants gave +0.0016 to +0.0061, hence the wider interval.

## Operational notes

* Peak-hour replicas will run at 16 (quota) on Wed-Fri with this schedule; utilization will be
  around 75-85%. If the quota is ever raised, the 10-17 fractions can be moved toward 1.0.
* Re-check the 10-17 fractions if the traffic level grows: they are tuned to today's volume.
