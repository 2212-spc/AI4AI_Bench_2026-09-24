# Measurement and extrapolation summary

Read manual and notebook. Used five new experiments, costing
1.99939999992e19 FLOPs, within the 2e19 budget. No cooldown branches were
measured. All successful requests and results are in lab_runs.jsonl.

The notebook's checkpoints cannot be treated as finished short runs. Fit
loss as a model-size power term plus a tokens-seen power term and an
additional schedule-state penalty. `fit_scaling.py` fits a power of the
relative learning rate; `fit_crosscheck.py` instead estimates separate
penalties for each observed schedule state. Both use the actual measurements,
not the notebook's fitted coefficients.

The LR-power fit gives size/data exponents approximately 0.365/0.345,
completed production WSD loss 2.670, q3 gap 0.194, and q4 endpoints
0.0224 and 0.2526. The schedule-state crosscheck gives exponents 0.359/0.346,
WSD loss 2.662, q3 gap 0.196, and q4 endpoints 0.0219 and 0.2512.
Residual RMS is approximately 0.0033 and 0.0021 nats, respectively.
The reported estimates round the agreement of these fits; intervals for
q2 and q3 are degenerate as required, rather than statistical intervals.

Both fits select option B at production compute. The crosscheck's
cosine final losses for A/B/C/D are approximately
2.719/2.685/2.704/2.774. Schedule endpoint terms do not change this ranking.

The fitted cosine checkpoint loss decreases across f in [0.5, 0.9].
Thus q4's minimum is at 0.9 and maximum at 0.5. The full interval reflects
the documented unknown stopping point, not seed uncertainty.

For q5, the predicted 70% cosine checkpoint minus the final 40%-horizon
WSD loss is approximately -0.037 to -0.041 nats. This comparison uses
fixed fractions and does not depend on the unknown stopping point, so
its verdict is supported.
