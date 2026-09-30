# MiniLab shipping decision

Ship **c1, c2, c3, c5, c6**; omit c4. Estimated expected validation MSE: **0.038897**.

The teammate's seed-0, one-change-at-a-time ablation could not identify interactions. In particular, clipping changes the usefulness of the higher learning rate and momentum. I used all 40 authorized runs: 20 for five-seed comparisons of baseline/c5/c6/c5+c6, 16 for every subset of c1–c4 added to c5+c6 on a common seed, and four independent confirmation seeds for the selected candidate. Full cluster results are saved in `run_history.json`.

## Evidence

Across common seeds 1–5:

| Recipe | Mean validation MSE |
|---|---:|
| Baseline | 0.050813 |
| c5 | 0.049708 |
| c6 | 0.050259 |
| c5, c6 | 0.049145 |

Weight decay improved each paired seed comparison. The RNG change improved four of five comparisons, with a smaller average benefit. These results motivated retaining c5 and c6 during the interaction screen; they do not establish their effects in every recipe.

The full interaction screen on seed 6, with c5+c6 included in every row:

| Additional changes | Validation MSE |
|---|---:|
| None | 0.049557 |
| c1 | 0.044967 |
| c2 | 0.062940 |
| c1, c2 | 0.338100 (diverged) |
| c3 | 0.055194 |
| c1, c3 | 0.049841 |
| c2, c3 | 0.046263 |
| c1, c2, c3 | 0.039571 |
| c4 | 0.075721 |
| c1, c4 | 0.338100 (diverged) |
| c2, c4 | 0.338100 (diverged) |
| c1, c2, c4 | 0.338100 (diverged) |
| c3, c4 | 0.049536 |
| c1, c3, c4 | 0.047090 |
| c2, c3, c4 | 0.048943 |
| c1, c2, c3, c4 | 0.045208 |

Divergent runs count as 0.3381, as specified by the production recipe; they were not discarded. At this seed, adding clipping to c1+c2+c5+c6 converted divergence to the best observed result, 0.039571. Removing c1 or c2 from that candidate worsened MSE to 0.046263 or 0.049841, respectively. Adding c4 worsened it to 0.045208. Thus the evidence supports the combined optimization changes despite their poor isolated seed-0 results.

## Expected MSE estimate and uncertainty

After selecting the candidate, fresh seeds 7–10 yielded 0.037802, 0.040647, 0.038913, 0.038226, with no divergences. Their arithmetic mean is 0.038897; sample standard deviation is 0.001253, and standard error is 0.000627. This confirmation-only mean is the submitted prediction, avoiding reuse of the seed that selected the winner. Including the screening seed gives a similar five-seed mean of 0.039032.

The selected candidate is clearly better in these observations than c5+c6 alone. The budget does not establish a global optimum: only 19 distinct subsets were tested, c5/c6 removal was not tested in the selected optimization regime, and five successful candidate runs cannot exclude rare divergence. The decision relies on the large interaction-screen advantage, independent confirmation, and the favorable paired evidence for c5 and c6.
