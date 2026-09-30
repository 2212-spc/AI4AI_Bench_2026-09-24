# Recipe change decision

Ship `c1`, `c2`, `c3`, and `c5`.

The 40-run budget was used to test the combinations most likely to benefit from the strong learning-rate and momentum changes. The shipped combination had validation MSEs of 0.040626, 0.043708, 0.038692, 0.042782, 0.042540, 0.044537, 0.039854, and 0.041939 across eight seeds. Their mean is **0.041835**, which is the predicted expected validation MSE in `decision.json`.

This was the best repeated-seed result observed. Its mean was lower than the six-seed mean for `c1,c2,c3` (0.042729) and the four-seed mean for the same set plus `c6` (0.042676). Adding `c4` raised the observed mean to 0.047047 over three seeds, and adding `c6` to the selected set raised it to 0.042676 over four seeds. The teammate's single-seed results also indicated that `c3` and `c5` were worth testing in combination, while the full teammate plan without clipping diverged on seeds 17 and 29. Clipping removes that instability in the selected combination.
