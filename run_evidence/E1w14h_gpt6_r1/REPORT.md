Ship **c1, c2, c3, c5**: peak learning rate 0.1, momentum 0.97, global gradient clipping at 0.3, and coupled weight decay 0.001. Retain width 96 and the existing RNG arrangement. Estimated expected validation MSE: **0.03985529**.

Used exactly **40/40 lab runs**, all with explicit seeds. The complete results are in `experiments/history.json`. Divergence is scored as 0.3381, as required by the production recipe; it is never dropped from comparisons.

The experiment allocated 16 runs to the complete factorial of c1–c4 with c5 enabled and c6 disabled, all on seed 17. Next, 12 runs tested all four c5/c6 settings for three promising cores (c1+c2+c3, c1+c2+c3+c4, c3+c4), all on seed 29. Finally, 10 runs compared the selected core plus c5 with and without c6 on five fresh seeds (0, 1, 2, 3, 43), and two runs evaluated the teammate's c5+c6 proposal on seeds 17 and 29. Combinations drove selection; the teammate's single-change table was not treated as evidence of additive effects.

Key common-seed comparisons (validation MSE):

| Changes | Seed 17 | Seed 29 |
|---|---:|---:|
| c1,c2,c3,c5 — selected | 0.040107 | 0.040104 |
| c1,c2,c3,c4,c5 | 0.045044 | 0.045306 |
| c3,c4,c5 | 0.045491 | 0.047172 |
| c1,c2,c3 — remove decay | untested | 0.053629 |
| c1,c2,c5 — remove clipping | 0.338100 (diverged) | untested |
| c5,c6 — teammate proposal | 0.049959 | 0.051936 |

The interactions materially change the decision. On seed 17, c1+c5 and c2+c5 scored 0.048521 and 0.067078, but c1+c2+c5 diverged. Adding clipping yielded 0.040107. Removing c1 or c2 from the selected recipe raised seed-17 MSE to 0.048405 or 0.047791, respectively. On seed 29, adding decay to c1+c2+c3 reduced MSE from 0.053629 to 0.040104. Widening the selected recipe worsened results on both shared screening seeds. The selected recipe beat the teammate's proposal by 21.3% on the average of their two common seeds.

For the **exact selected recipe**, the seven observed MSEs were:

| Seed | MSE |
|---|---:|
| 0 | 0.039483 |
| 1 | 0.038744 |
| 2 | 0.039386 |
| 3 | 0.041030 |
| 17 | 0.040107 |
| 29 | 0.040104 |
| 43 | 0.040133 |

The prediction is the arithmetic mean of these seven observations, 0.03985529; sample standard deviation is 0.000728 and standard error is 0.000275. No selected-recipe run diverged. The five fresh confirmation seeds alone average 0.03975520, supporting the screening result. A descriptive 95% t interval for the mean is approximately [0.039182, 0.040528]; it assumes representative independent seeds and does not account for adaptive selection or unseen failure modes.

On the six seeds shared by both RNG variants (0, 1, 2, 3, 29, 43), adding c6 increased MSE by an average of 0.00002517 (paired standard error 0.000292; three wins and three losses). This effect is unresolved and tiny relative to the combination gains. Retain the current RNG arrangement: its paired point estimate is slightly better. The seven-seed prediction uses only the exact shipped recipe, without pooling RNG variants.

This budget does not establish the global optimum across all 64 subsets. It covers every c1–c4 combination with decay, probes decay/RNG interactions in the leading cores, and confirms the chosen recipe over multiple seeds. Remaining uncertainty includes sparsely sampled alternatives and unobserved divergences.
