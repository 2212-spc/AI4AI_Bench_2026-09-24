# Launch memo: Model B

I recommend routing all assistant traffic to Model B. Across 24 experiment-days, B raised the answer rating from about 0.640 for A to about 0.683 for B. B takes about 4.09 seconds to generate a response, compared with about 1.95 seconds for A, so the autoscaler provisions more replicas. At the observed traffic levels, the full-B schedule requires at most about 26 replicas, below the quota of 32.

The queue measurements and the exponential-service model imply a full-B mean queue wait of roughly 0.14 seconds. The gain in answer quality therefore dominates the abandonment and queue-wait terms. The schedule in `rollout.conf` routes 100% of every hour to B; the schedule has no experiment guardrail constraint after launch, and the capacity estimate remains below quota.

Q1: [+0.0414, +0.0472]
Q2: [+0.0424, +0.0462]

The intervals are score units per request relative to all-A. Q1 covers the estimated value if every hour runs at 100% B. Q2 is narrower because the submitted schedule is that same full-B policy.
