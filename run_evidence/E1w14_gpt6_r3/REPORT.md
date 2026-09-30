# Recipe change decision

Ship **c3, c4, c5, and c6**.

I used all 40 available training runs. The teammate's seed-0 results were first reproduced, then I tested combinations to look for interactions. The key result was that c3 and c4 have a strong joint benefit even though each change alone was worse than baseline. I then evaluated the leading candidates on seeds 1 through 5 using matched seeds:

| recipe | runs | mean validation MSE |
|---|---:|---:|
| baseline | 6 (seeds 0–5) | 0.05085 |
| c5 + c6 | 6 (seeds 0–5) | 0.04921 |
| c3 + c4 | 6 (seeds 0–5) | 0.04682 |
| c3 + c4 + c5 + c6 | 5 (seeds 1–5) | 0.04604 |

On the five matched seeds, adding c5 and c6 to c3+c4 lowered MSE by 0.00105 on average. The c3+c4+c5+c6 recipe beat the teammate's c5+c6 proposal on 4 of those 5 seeds and had no divergence. The other changes were not promising in the targeted screening: c1, c2, and c4 were individually worse on seed 0; c2+c4 diverged; and combinations involving c1 or c2 did not beat the c3+c4 family in the seed-0 screen.

The shipped recipe's predicted expected validation MSE is **0.0460**, based on the five-seed mean for the selected combination.

The selected recipe's individual results were 0.046899, 0.046417, 0.045518,
0.047419, and 0.043938. Their sample standard deviation is 0.00137 and the
standard error of the mean is 0.00061. A nominal 95% Student-t interval is
approximately [0.04434, 0.04774]; this describes seed sampling uncertainty
and does not correct for adaptive recipe selection. The paired advantage
over c3+c4 is uncertain (standard error 0.00082), so the evidence for adding
c5+c6 is modest.

The budget covered 21 distinct subsets, not all 64. Untested interactions
and rare failures remain possible; these experiments do not establish the
global optimum. Divergent runs count as MSE 0.3381 rather than being dropped.
The complete cluster history is saved in `experiment_history.json`.
