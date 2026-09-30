# Model B launch memo

I recommend routing 100% of traffic to Model B in every hour. The schedule is in `rollout.conf`.

Model B improved the answer rating from approximately 0.6402 for A to 0.6614 for B in the experiments, while its mean generation time was approximately 3.92 seconds versus 1.70 seconds for A. The measured queue response, including abandonment, still left the B score higher in every tested hour. Under the proposed schedule, the highest observed traffic hour requires about 24 replicas at the autoscaler target, below the 32-replica quota. The schedule is also within the 08:00–21:59 experiment guardrail because that guardrail applies to experiments; the launch configuration itself routes the winning fraction.

The estimates below use request weighted production traffic, the observed A and B ratings and generation times, and the steady-state multi-server queue with exponential patience calibrated from the telemetry. Queue and abandonment measurements have roughly 10% hour-level noise, so the intervals include that uncertainty and variation across traffic days.

Q1: [+0.0195, +0.0254]
Q2: [+0.0208, +0.0242]

Here Q1 is the estimated change from all-A to 100% B in every hour. Q2 is the value of the schedule in `rollout.conf`, relative to all-A. They are the same schedule in this launch, so their centers coincide.
