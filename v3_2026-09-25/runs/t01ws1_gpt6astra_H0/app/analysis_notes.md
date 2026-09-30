# Learning-rate transfer analysis

Used the 25 notebook observations and 40 additional runs. Compute used: 1.15896e20 FLOPs, below the 1.2e20 budget; every run was below the 6e18 cap.

Sweeps independently varied model size and training tokens. An asymmetric quadratic in log learning rate fits with residual RMS 0.005123 nats in the joint loss model. The learning-rate optimum fitted with a separate loss baseline for each (N,D) is:

ln(lr*) = -6.08655157 + (-0.09610001) ln(N/1e8) + (-0.22626533) ln(D/1e9).

For z = ln(lr/lr*), excess loss is 0.05097913 z^2 below the optimum and 0.12781569 z^2 above it. Independent baselines avoid imposing a loss scaling law to identify the learning-rate optimum.

At fixed token count, multiplying model size by ten multiplies the optimum by 0.801493, a 19.85% decrease. Thus q5 is refuted.

For the final optimal loss, a joint fit uses L* = 1.91468712 + 1.19442192(N/1e8)^(-0.35596725) + 1.05859697(D/1e9)^(-0.34386894), with the same form of learning-rate penalty. Extrapolation uses the manual's scale-consistency guarantee.

The batch exponent is unidentifiable in this lab. Its entire documented range [0,0.5] gives the q4 log10 interval by adding [0, 0.5 log10(8)] to q2. The largest allowed optimum is 0.00107968, below 0.00173; q6 is supported.

The excess losses predicted for the q7 choices are {"A": 0.073766, "B": 0.013247, "C": 0.00437, "D": 0.098202}; C is best.

Numeric estimates resolved by experiments use degenerate intervals, as the manual requires. Only the documented unknown batch exponent widens an interval.
