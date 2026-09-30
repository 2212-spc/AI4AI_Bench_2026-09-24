I recommend shipping **c1, c2, c3, c5, c6**, with estimated expected validation MSE **0.0425**.

The 40 completed runs covered all 16 subsets of c1–c4, each with c5+c6 absent and present, at seed 0 (32 runs), plus seed-1 repeats of the eight configurations without c3 or c4. The full evidence is saved in `run_history.json`. The teammate's seed-0 measurements match the repeated observations and are not independent replications.

| Configuration | Seed 0 MSE | Seed 1 MSE |
|---|---:|---:|
| Baseline | 0.051038 | 0.052601 |
| c5,c6 | 0.049553 | 0.051070 |
| c1,c2 | 0.338100 (diverged) | 0.338100 (diverged) |
| c1,c2,c5,c6 | 0.338100 (diverged) | 0.338100 (diverged) |
| c1,c2,c3 | 0.052504 | — |
| c2,c3,c5,c6 | 0.045546 | — |
| c3,c4 | 0.045473 | — |
| **c1,c2,c3,c5,c6** | **0.039903** | — |
| All six | 0.050266 | — |

The single-change results miss substantial interactions. The high learning rate and high momentum together diverged on both seeds tested; clipping stabilized the tested seed-0 combinations. Adding c5+c6 to c1+c2+c3 reduced seed-0 MSE by 0.012601. The selected recipe was the best observed configuration, about 12.3% below the next best seed-0 result. Adding width to this recipe worsened its observed MSE by 0.010363, so I exclude c4. Width did help some other combinations; its effect is conditional.

The prediction 0.0425 is a heuristic estimate: the selected recipe's 0.039903 observation plus about 0.0026 to allow for selection optimism and seed variation. Among the six nondivergent configurations repeated across seeds 0 and 1, the absolute seed differences ranged from 0.000769 to 0.004335. These comparisons suggest the scale of uncertainty but cannot establish the selected recipe's mean or divergence probability. Divergence is always scored as 0.3381, never omitted.

Limitations: a counting error in the experiment loop exhausted the budget before the intended finalist replications; the selected recipe has only one observed seed. c5 and c6 were varied together, so their individual contributions are unresolved, and the other 32 subsets were not evaluated. The prediction is not a measured many-seed mean, and this evidence does not establish global optimality. The shipping choice uses the strongest observed combination and its local comparisons, with substantial remaining uncertainty.
