# MiniLab recipe decision

Ship **c1, c2, c3, c5, c6**; omit c4. Estimated expected validation MSE: **0.0419064**. This retains hidden width 96, raises peak learning rate to 0.06 and momentum to 0.95, enables norm clipping at 1.0, adds weight decay 0.0003, and uses a dedicated minibatch RNG.

## Experiments and evidence

Used exactly **80/80 cluster runs**, covering 35 distinct subsets. Raw results, including seeds and divergence flags, are in `experiments/runs.jsonl`; experiment drivers are in `experiments/triage.py` and `experiments/targeted.py`. Every divergence is scored as 0.3466, never discarded.

The first 32 runs screened all 16 combinations of c1–c4 twice: once with neither c5 nor c6 (seed 104729), and once with both (seed 130363). These blocks identify promising combinations, but cannot separate seed effects from c5/c6 effects. The remaining 48 runs compared eight candidates on the same six fresh seeds, 200001–200006, chosen before seeing their results:

| Changes | Runs in fresh-seed comparison | Mean validation MSE | Sample standard deviation |
|---|---:|---:|---:|
| **c1,c2,c3,c5,c6** | 6 | **0.041894** | 0.000573 |
| c1,c2,c3,c6 | 6 | 0.043077 | 0.000808 |
| c1,c2,c3,c5 | 6 | 0.043628 | 0.002478 |
| c1,c2,c3 | 6 | 0.044698 | 0.002535 |
| c1,c3,c5,c6 | 6 | 0.046049 | 0.003131 |
| c2,c3,c5,c6 | 6 | 0.046158 | 0.001291 |
| c1,c2,c5,c6 | 6 | 0.055067 | 0.012240 |
| c1,c2,c6 | 6 | 0.058196 | 0.013488 |

All 48 follow-up runs were finite; the large losses without clipping still count fully toward the expectation.

The teammate's one-seed singleton ablation does not establish that its winners combine well. In the screen, c1+c2+c4 diverged both without c5/c6 and with both enabled. Even with clipping, adding c4 to c1+c2+c3 increased MSE from 0.044525 to 0.046576 in the first block; with c5/c6 enabled it increased MSE from 0.041982 to 0.057417 in the second block. These comparisons support excluding width from the selected aggressive optimizer recipe, though the teammate's exact proposed subset was not tested.

Clipping is valuable in context: removing c3 from the selected recipe increased fresh-seed mean MSE by 0.013173. Removing c1 or c2 also worsened the mean. Weight decay provided a smaller repeatable gain: removing c5 increased mean MSE by 0.001183, with paired standard error 0.000295. Removing c6 increased mean MSE by 0.001734, with paired standard error 0.000833, and increased observed seed variability. The evidence for c6 is weaker than for clipping, but favors inclusion. A changed RNG stream also changes seedwise correspondence, so individual paired wins are not enough to establish its effect; the decision uses averages across seeds.

## Prediction and uncertainty

The selected recipe's seven observed losses are 0.041982, 0.042122, 0.042466, 0.041568, 0.042023, 0.042283, and 0.040901. Their arithmetic mean, **0.041906428571428574**, is the submitted prediction. No selected-recipe run diverged. Its six fresh-seed runs alone average 0.0418938333, essentially the same estimate, so the screening observation has little influence.

Across all seven runs, the sample standard deviation is 0.000524 and the standard error of the mean is 0.000198. A conventional small-sample t interval is approximately 0.04142–0.04239, conditional on ordinary sampling assumptions; it does not account for adaptive selection or unseen rare failures. We did not evaluate every subset across many seeds, so this is an evidence-based selection rather than a proof of the global optimum. The budget was concentrated on replication of the promising recipe family to estimate expected loss instead of selecting a lucky minimum from single runs.
