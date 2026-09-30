# Model B launch

Route Model B primarily overnight, when the shared pool has enough headroom to absorb its longer generation time. During 08:00-21:59, retain the 30% experiment guardrail in the launch schedule; peak-hour queueing makes a larger share counterproductive under the fixed 16-replica quota. The overnight schedule uses near-full traffic to preserve headroom around autoscaler replica-count boundaries.

The estimate uses all-A historical telemetry and both experiment runs. I inferred abandonment patience from the measured abandonment rate and mean queue wait (approximately 60 seconds), then evaluated an M/M/c+M queue per hour/day with the observed traffic rates, model generation times, and autoscaler rule. The model reproduces the observed mixed-arm mean score within 0.0002. Interval bounds allow for telemetry and parameter uncertainty.

Q1: [-0.0200, -0.0155]
Q2: [+0.0135, +0.0180]
