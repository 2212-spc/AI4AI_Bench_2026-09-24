# Measurement and extrapolation

Used all 10 allowed runs, charging 1.9992864e19 FLOPs (within the 2e19 budget).
The notebook data were also included. Intermediate cosine checkpoints were not
mistaken for completed training runs.

WSD cooldown branches provide completed-run observations at multiple token
horizons. Fit the completed loss as E + A*(N/1e8)^(-alpha) +
B*(D/1e9)^(-beta). Jointly fit schedule residuals using both notebook and new
measurements. `fit.py` uses independent cosine-fraction residuals;
`smooth.py` checks a smooth residual against the documented learning-rate
schedule. `random_effects.py` checks sensitivity to shared run-level noise.
Size and token exponents are approximately 0.34–0.35 and 0.354–0.356,
respectively. Fits have residuals of approximately 0.006–0.007 nats,
with smaller noise at larger model sizes.

Production WSD completion is approximately 1.95–1.96 nats across weighting
and shared-noise checks; report a point estimate, not a statistical interval.
The four fixed-compute choices have estimated WSD losses near
2.09, 2.02, 1.97, and 1.95 nats. A common finished cosine residual does not
change their ranking, so D wins.

The same-token cosine checkpoint minus completed WSD loss is the schedule
residual: approximately 0.086 nats at fraction 0.4. The token-learning term
cancels from this comparison.

For early stopping, use
B*(60)^(-beta)*(f^(-beta)-1) + penalty_cos(f)-penalty_cos(1).
The measured smooth loss decreases over f in [0.5,0.9]. Thus the minimum
is at 0.9 and the maximum at 0.5. The resulting set is approximately
[0.0115,0.1125] nats. This is a range over the documented unknown stopping
fraction, not a seed-noise confidence interval.

For q5, the difference is
B*(60)^(-beta)*(0.5^(-beta)-0.35^(-beta)) + penalty_cos(0.5),
which is positive by approximately 0.03–0.04 nats. Thus the claim that the
cosine checkpoint has lower loss is refuted; it does not depend on f_stop.
