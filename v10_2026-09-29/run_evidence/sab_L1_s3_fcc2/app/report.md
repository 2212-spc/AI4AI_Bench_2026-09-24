# Launch memo: model B on the shared serving pool

## Decision

Launch B with an hour-of-day schedule (`/app/rollout.conf`): 100% B overnight and in the evening
(19:00-08:59), tapering to roughly 40% B at the midday peak (13:00-16:00). Never launch B to 100% at
all hours: that would make the product worse than today.

## Why not 100% everywhere

B is better liked but slower. Measured over 1.93M served B answers and 4.4M served A answers:

| | model A | model B |
|---|---|---|
| mean rating | 0.6422 | 0.6706 |
| mean generation time | 2.53 s | 5.72 s |

B needs 2.26x the GPU-seconds per request. With the 16-replica quota, peak-hour traffic (about
11,400 req/h at 14:00-15:00) can only be all-B if utilization exceeds 100%: the queue then grows until
abandonment balances it, roughly 12-15% of requests get no answer and the rest wait 7-10 s. The night
capacity drills (hours 01-05 capped at 6 replicas, 100% and 40% B) reproduced exactly this regime and
matched the queue model (Erlang-A, exponential patience with hazard 0.0169/s, mean patience about 59 s;
estimated from the abandon/wait ratio in 864 hours of telemetry) within measurement noise.

The mixed-fleet fraction in each hour is set so that the pool stays at or just under the quota at the
0.75 target utilization on a typical day. In the peak hours the marginal queueing cost of another 1% B
equals B's rating gain, so the schedule tapers with load. Between hours 09:00-18:00 the fraction is
therefore well below 1; everywhere else B fits inside the quota and runs at 100%.

## Expected value (score units per request, relative to all-A today)

Q1: [-0.0205, -0.0145]
Q2: [0.0206, 0.0246]

Q1 is the value of routing 100% to B in every hour (negative: the midday overload wipes out the rating
gain). Q2 is the value of the schedule in `rollout.conf`. Point estimates: Q1 = -0.0174, Q2 = +0.0225.
Q2 comes almost entirely from the rating uplift (+0.0284 per served B request) times the B share of
traffic (about 75% of requests overall), with a small queueing penalty; the estimate averages over the
observed day-to-day traffic variation (36 days of hourly rates, ±3% daily factor).

## Evidence used (24 experiment-days)

* 22 days of the maximum allowed exposure (100% B at 22:00-07:59, 30% B at 08:00-21:59): ratings,
  generation times, and the queue behaviour of the mixed fleet at 0.70-0.75 utilization.
* 2 capacity drills at night (01:00-05:59, autoscaler capped at 6 replicas, 100% and 40% B) to validate
  the queue/abandonment model at 0.72-1.0 utilization and in overload, which is the regime that governs
  the peak-hour trade-off and Q1.
* 14 days of production history (all A) for the baseline, patience hazard, and traffic distribution.

Ratings of B were flat across hour of day, weekday, B fraction, and queue wait, so the uplift is
treated as constant. Generation times were stable (exponential, means above).

## Operational notes

* No quota increase is assumed. If the quota is later raised, the midday fractions can rise toward 1.
* The autoscaler will run at 16 replicas from 08:00 through 19:00 under this schedule; expect
  utilization 0.73-0.75 in those hours and about 0.70 overnight.
* Traffic growth of more than ~5% at peak would call for lowering the 11:00-17:00 fractions; the loss
  from a fraction slightly too high is small, but sustained overload is not.
