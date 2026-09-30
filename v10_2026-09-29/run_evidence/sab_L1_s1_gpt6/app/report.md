Launch B with the hourly fractions in `rollout.conf`, repeating every day. Route 100% to B during 00:00–09:59 and 18:00–23:59; reduce B's share during the midday traffic peak. The estimated incremental value is approximately **+0.0359 score units per arriving request**. Full B throughout the day has an estimated value of **+0.0177**. Both are measured against all-A under the same arrival distribution and autoscaling policy.

Q1: [+0.0148, +0.0207]
Q2: [+0.0340, +0.0379]

These are uncertainty ranges for long-run expected incremental score, including traffic and parameter estimation uncertainty and queue-model sensitivity. They are not ranges for an individual day's realized score.

| Hours | Fraction routed to B |
|---|---:|
| 00–09 | 1.00 |
| 10 | 0.86 |
| 11 | 0.69 |
| 12 | 0.57 |
| 13 | 0.50 |
| 14 | 0.49 |
| 15 | 0.51 |
| 16 | 0.60 |
| 17 | 0.76 |
| 18–23 | 1.00 |

B improves served-answer ratings by about 0.0433, but increases mean generation time from 3.571 to 8.015 seconds. The pooled rating estimates are 0.6611 for A and 0.7045 for B. In lighter hours, the autoscaler can add enough replicas for B's quality improvement to dominate. At the midday peak, full B would offer more work than the 24 replicas can process without substantial abandonment. The reduced fractions preserve most of the quality benefit while controlling the shared queue. The 30% daytime restriction applies to experiments; it does not limit the final launch.

I used all 24 experiment-days. The first 12 days tested 100% B in hours 00–07 and 22–23, and 30% B in hours 08–21. The next 12 days tested 30% B in every hour. Both experiments obeyed the guardrail and used the normal autoscaler with the 24-replica quota. The resulting data are `data/abtest_000.csv` and `data/abtest_001.csv`, alongside the 14 historical all-A days. No capacity override was used.

Exponential patience makes abandonment informative even without a direct patience measurement. In steady state, abandonment flow is the patience hazard times the mean queue length. Little's law, counting the waiting time of abandoned requests, therefore gives

`P(abandon) = theta × E[queue wait per arrival]`.

The aggregate telemetry estimates theta at 0.04174 per second, or mean patience of approximately 24.0 seconds. This identity applies to the shared queue with both service-time distributions. It avoids assuming that patience equals an observed waiting-time average.

For each hour and candidate fraction f, I recomputed offered load and the integer replica count as `min(24, max(4, ceil(lambda × ((1-f) × mean_A + f × mean_B) / 0.75)))`. I then estimated abandonment and queue waiting and evaluated

`((1-f) × rating_A + f × rating_B) × (1-P(abandon)) - 0.0062 × E[queue wait]`.

Random routing and common patience make queueing and abandonment independent of a request's model assignment before it enters service. Consequently, the arm score difference within a mixed-pool experiment does not by itself identify the value of changing the pool's routing fraction: both arms experience the same congestion. The calculation explicitly includes that shared effect and scores every abandonment as zero answer value, with its waiting penalty retained.

The numerical search used a steady-state Erlang-A approximation with the mixture's mean service time, a fraction grid spaced by 0.001, and all 38 days of observed hourly arrival counts. Scores were weighted by arriving requests, and the all-A counterfactual used those same traffic observations. The search placed B's fraction near 0.86, 0.69, 0.57, 0.50, 0.49, 0.51, 0.60, and 0.76 in hours 10 through 17. I rounded negligible sample-specific changes below 100% in light hours up to full B.

The approximation reproduces aggregate observed queue statistics closely. For all-A, observed versus predicted mean queue waiting was 0.2788 versus 0.2800 seconds; at 30% B, 0.2021 versus 0.2002 seconds; at 100% B in the tested light hours, 0.2528 versus 0.2535 seconds. Aggregate abandonment rates show the same agreement. Utilization residual means were below 0.0007 across these routing groups.

A mixture of exponentials is not itself exponential. I checked this approximation with a separate continuous-time Markov model whose state tracks the number of B requests in service and the queue length, using the individual A and B service rates and exponential abandonment. Evaluating this model across the observed traffic range in the mixed launch hours lowered estimated schedule value by approximately 0.00013. The reported schedule estimate includes that sensitivity. For the pure all-A and all-B counterfactuals, Erlang-A directly matches the specified service distribution.

A 600-resample bootstrap of whole calendar days, refitting ratings, generation times, patience, and traffic each time, gave approximate 95% ranges of +0.0157 to +0.0197 for full B and +0.0354 to +0.0366 for the schedule before the small service-mixture adjustment. The reported intervals are wider to cover model and estimation sensitivity. Full B is particularly sensitive to traffic: shifting all arrival rates by ±1% changed its estimated value by about 0.002. This is why I retain day-to-day traffic variation instead of evaluating only the mean hour.

Within the fitted model, rounding costs less than 0.00001 score per request. Across scenarios with ±2% arrival rates and ±0.001 changes to B's rating, the chosen fractions remained within 0.00008 of a separately reoptimized schedule. This supports the requested 0.0010 optimization tolerance, while recognizing that empirical validation of heavily loaded daytime full B was prohibited. The launch estimates necessarily extrapolate to those fractions using the identified queue model.

The configuration assumes the workload distribution, model service times, ratings, patience, and hourly steady-state assumptions remain representative. After launch, compare arrivals, served ratings, abandonment, waiting, and replica saturation against these estimates, especially during hours 10–17. The supporting calculations and reproducible model code are in `analysis/model.py`, `analysis/mixture.py`, `analysis/uncertainty.py`, `analysis/robustness.py`, and `analysis/results.json`.
