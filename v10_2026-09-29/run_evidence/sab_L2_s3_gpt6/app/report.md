# Model B Launch Memo

Route all traffic to B from 00:00-07:59 and 22:00-23:59. Route 30% to B from 08:00-21:59, respecting the experiment guardrail. Keep the existing 16-replica quota.

The 14-day broad probe showed a consistent rating lift from B, but generation time rises substantially in the daytime peak. Under the shared queue, running B everywhere overloads the peak hours and loses value despite improved answers. The follow-up 10-day test exercised this proposed schedule. Queue behavior was estimated with an exponential-service multi-server queue with exponential abandonment, calibrated against production and experiment telemetry; values are relative to all-A and averaged by request volume.

Q1: [-0.0139, -0.0083]
Q2: [+0.0107, +0.0143]
