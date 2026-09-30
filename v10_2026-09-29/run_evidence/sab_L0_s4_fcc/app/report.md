# Launch memo: model B on the shared serving pool

**Recommendation: route 100% of traffic to B in every hour** (`/app/rollout.conf`, all 24 lines = `1.0`).

Q1: [+0.0425, +0.0455]
Q2: [+0.0425, +0.0455]

(Q1 = value of B at 100% in every hour; Q2 = value of the schedule in `rollout.conf`. They are the same
schedule, so the same interval. Point estimate **+0.0440** score units per request, i.e. mean per-request
score rises from ~0.636 to ~0.680.)

## What we measured (24 experiment-days, 20 calendar days of live traffic)

| quantity | A | B |
|---|---|---|
| mean rating of served answers | 0.6404 ± 0.0001 (6.4 M requests) | 0.6831 ± 0.0002 (2.9 M requests) |
| mean generation time | 1.95 s | 4.09 s |

* B's rating lift over A is **+0.0427 ± 0.0003** (2 s.e.).  It is the same at night (100% B) and by day (30% B),
  the same at 30% / 50% / 100% routing, flat across the 20 experiment days (no novelty decay) and
  uncorrelated with queue wait, utilization or replica count.
* Nothing else differs between arms: both arms share one FIFO queue, so abandonment and waiting are pool
  properties (confirmed: `score_arm = rating_arm x (1 - abandon) - 0.0069 x wait` to 5 decimals in every
  row).

## Why the queue does not hurt B

B's answers take 2.1x longer to generate, so the fear was queueing.  It does not materialise:

* The autoscaler sizes the pool to the mix actually running (`ceil(load/0.75)`), so utilization stays at
  ~0.70-0.74 under 100% B exactly as under A.  Bigger pools queue *less* at the same utilization, so B
  actually has slightly lower abandonment/wait than A (queue penalty ~0.0035 vs ~0.0048 per request);
  this adds ~+0.0013 to the lift, giving +0.0440 total.
* **Quota headroom:** the busiest hour (13:00-15:00, ~15.9 k req/h, worst observed day ~16.6 k) needs
  `ceil(16600 x 4.09 / 3600 / 0.75) = 26` replicas at 100% B - inside the 32-replica quota with ~20% margin.
  The quota would only bind if peak traffic grew ~22% (to >19.5 k req/h) or B's generation time rose to
  >5.4 s.  If either happens the pool saturates and the value degrades quickly (see drill below) - the
  autoscaler/quota should be watched as B traffic grows.
* Queue model: M/M/c with exponential patience (Erlang-A).  Fitted mean patience 1/θ = 55 s
  (θ = abandon_rate / mean_wait = 0.0182 s⁻¹, identical in history, mixed hours and 100% B hours).
  Predicted abandonment/wait match observations within 1% on average across all 34 days x 24 hours.
* **Capacity drill** (01:00-05:59, 100% B, autoscaler capped at 9 replicas, utilization 0.76 -> 0.95): observed
  abandonment 0.8% -> 6.0% and wait 0.5 s -> 3.3 s matched the Erlang-A prediction hour by hour, so the model
  is trusted for the overload regime too.  B's rating stayed at 0.683 even at 3 s waits.

## Why not a partial rollout

Per-request score is linear in the fraction routed to B (rating lift) plus a small queue term that also
favours B.  Every hour's optimum on the model is f = 1.0; the nearest alternative (0.9 in every hour) loses
~0.0045 per request, well outside the 0.001 tolerance, and the 30% daytime guardrail configuration we ran
live is worth only ~+0.013.  There is no hour where A is preferable.

## Interval construction

Value = (rating_B - rating_A) + (queue penalty_A - queue penalty_B), request-weighted over the empirical
hourly arrival distribution of 34 days (history + experiments).  Statistical s.e. of the rating difference is
0.00025; queue terms are ~0.0045 each with model error <10%.  The quoted interval ±0.0015 around +0.0440
covers ~6 s.e. of rating noise plus the full plausible queue-model error.  Cross-check: the directly measured
night-hour lift (100% B experiment rows vs. all-A history, same hours) is +0.0441.

## Data

* `/app/data/abtest_000.csv` - day 15: 50% B nights / 30% B days.
* `/app/data/abtest_001.csv`, `abtest_003.csv`, `abtest_004.csv` - days 16-19, 21-38: 100% B 22:00-07:59, 30% B 08:00-21:59.
* `/app/data/abtest_002.csv` - day 20: capacity drill (100% B, 01:00-05:59, cap 9 replicas).
* `/app/model.py`, `/app/value.py` - Erlang-A model and schedule valuation.
