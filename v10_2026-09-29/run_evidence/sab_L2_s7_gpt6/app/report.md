# Launch Model B with an hourly schedule

Launch the schedule in `rollout.conf`, repeated every day. Route all traffic to B outside
10:00–17:59; during those eight hours use the fractions below. The estimated improvement is
**+0.0196 score units per arriving request** relative to all-A. A blanket 100% B launch has a
much smaller estimated improvement, **+0.0028**, and is sensitive to traffic and service time.

Q1: [-0.0002, +0.0057]
Q2: [+0.0176, +0.0215]

These are uncertainty ranges for long-run, request-weighted value relative to all-A. They
include allowance for traffic sampling, rating estimates, generation-time estimates, and
queue-model approximation; they are not guarantees for individual days.

| Hour | Fraction routed to B |
|---|---:|
| 00–09 | 1.000 |
| 10 | 0.828 |
| 11 | 0.678 |
| 12 | 0.628 |
| 13 | 0.542 |
| 14 | 0.488 |
| 15 | 0.494 |
| 16 | 0.564 |
| 17 | 0.698 |
| 18–23 | 1.000 |

## Evidence and experiment use

The analysis uses all 14 historical days and all 24 authorized experiment-days:

- Days 15–28: 30% B in every hour, providing concurrent arm measurements across two full weeks.
- Days 29–38: 100% B at 01:00–05:59 and 30% B in every other hour, checking pure-B behavior
  overnight and adding observations across the whole traffic profile.

The experiment files are `data/abtest_000.csv` and `data/abtest_001.csv`. Every experiment
respected the daytime 30% guardrail. No replica-cap drill was used. The 30% restriction is an
experiment guardrail; the requested launch optimization permits fractions up to 100%.

Request-weighted estimates, pooled within the two service regimes visible in the telemetry:

| Quantity | A | B, 00–08 and 18–23 | B, 09–17 |
|---|---:|---:|---:|
| Mean generation time, seconds | 3.181 | 5.904 | 7.840 |
| Mean served-answer rating | 0.68234 | 0.70129 | 0.71162 |

B improves answer ratings but consumes substantially more capacity. The quota binds during
peak traffic, so a complete migration raises abandonment and queue costs for the whole pool.
The proposed schedule keeps most of B's benefit while limiting that congestion.

## Estimation and checks

Under constant abandonment hazard θ, steady-state flow balance gives
`P(abandon) = θ × E[queue wait]`, with waiting time averaged over **all** arrivals. The ratio
of aggregate abandonment to aggregate waiting time estimates θ as 0.01706 per second, or
58.63 seconds mean patience. Separate historical and experimental estimates are consistent.

For each hourly traffic scenario and B fraction f, I recompute the replica count as
`min(16, max(4, ceil(arrival_rate × ((1-f) × mean_A + f × mean_B) / 0.75)))`.
The score calculation is
`((1-f) × rating_A + f × rating_B) × (1-P(abandon)) - 0.008 × E[queue wait]`.
Both models experience the shared queue. A per-arm experimental score difference alone
would miss the effect of changing f on the whole pool.

The queue predictor uses an Erlang-A stationary distribution with the traffic mix's mean
service time. For a mixture of unequal exponential generation times this is an approximation;
it is exact for each pure-model service distribution. Its mean waiting-time predictions are
0.4346 seconds versus 0.4365 observed in history and 0.4056 versus 0.4083 in experiments,
using the actual recorded replica counts. A separate check inferring arrival rates from
utilization gives observed/predicted waiting-time ratios of 1.004 for A, 1.006 for mixed
traffic, and 1.017 for pure B. These checks support its use here, while the reported ranges
allow for approximation error when extrapolating to untested daytime fractions.

I average total expected scores divided by total expected arrivals, giving each weekday equal
calendar weight and retaining day-to-day traffic variation. I optimize each hour independently
on a fine fraction grid, with slight smoothing of empirical traffic to reduce sensitivity to
individual autoscaler boundaries. The selected schedule's loss against an unsmoothed empirical
grid optimum is only 0.00011 score units per request under the fitted model.

Whole-day resampling within weekday strata preserves the relationship between hours. Its
5th–95th percentile ranges, for traffic uncertainty alone, are approximately +0.00085 to
+0.00472 for all-B and +0.01928 to +0.01984 for this schedule. The reported ranges above are
wider to allow for the other estimated quantities. Replacing observed counts with
utilization-inferred arrival rates gives gains of +0.00305 and +0.01967 respectively.
Increasing all arrival rates by 1% changes these estimates to +0.00094 and +0.01934: the
proposed schedule is considerably less sensitive to demand than a blanket migration.

## Platform follow-up

Keep the 16-replica quota and the existing autoscaler. Monitor hourly queue waiting time,
abandonment, served ratings, and generation time for the entire pool. Re-estimate the
schedule if the traffic profile or either model's generation time changes materially.
The ranges above assume the current serving conditions and steady-state hourly treatment.
