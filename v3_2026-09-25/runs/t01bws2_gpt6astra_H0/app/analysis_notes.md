# Learning-rate transfer estimates

Used all 25 notebook observations plus 15 new runs, consuming 5.886e19 of the 7e19 FLOP budget. Every charged run respected the 6e18 per-run cap. The final pair used N=3e7, D=3e10; a proposed larger configuration was refused without charge.

The notebook varies N and D together, so its fitted exponent conflates model-size and data-horizon dependence. New sweeps include N=1e7 at D=2e8 and D=1e11, N=1e8 at D=1e10, and two measurements at N=3e7, D=3e10.

Fit the joint loss surface using a power law for the LR optimum and separate quadratic curvatures in ln(lr/lr_opt) below and above the optimum. The baseline is a floor plus power-law contributions from N and D. Residuals were weighted by sqrt(N/1e8) to reduce the influence of small-model seed noise. The fitted optimum is:

ln(lr_opt) = -5.447617 - 0.142523 ln(N/1e8) - 0.194109 ln(D/2e9).

The fitted lower and upper LR curvatures are 0.059268 and 0.122488. Alternative smooth asymmetric fits yield similar excess loss and baseline loss. Estimates are extrapolated under the manual's scale-consistency guarantee.

- q1: lr_opt = 0.00198380.
- q2: lr_opt = 0.00092835.
- q3: excess loss = 0.024581 nats/token.
- q4: multiply the q2 optimum by 8^g, g in [0, 0.5], giving [0.00092835, 0.00262577].
- q5: fixed-token 10x size change multiplies the optimum by 0.720240, a 27.98% reduction; refuted.
- q6: 0.00157 lies inside q4's attainable range; undetermined.
- q7: optimum near 0.00126879; option B has the lowest fitted loss.
- q8: fitted optimal final loss = 2.401742 nats/token.

Only q4 has nondegenerate endpoints because it depends on the documented batch exponent. Other numeric answers use point estimates as the manual requires; intervals are not widened for seed noise.
