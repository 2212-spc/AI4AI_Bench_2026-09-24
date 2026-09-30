# Recipe change triage: decision and evidence

**Ship: c1, c2, c3, c5, c6 (drop c4). Predicted expected val MSE: 0.0421.**

## Method

40 runs total. All arms were compared on common training seeds (paired comparison, seeds 1 and 2 for
screening, more seeds for finalists). A diverged run is scored as 0.3466 (mean predictor), as the recipe
specifies. Because the reviewer warned about strong interactions, I screened combinations directly instead
of doing one-at-a-time ablations, then ablated single changes *out of* the leading combination.

## Per-arm results (val MSE, seed-averaged; diverged = 0.3466)

| arm | n seeds | mean | per-seed |
|---|---|---|---|
| **c1,c2,c3,c5,c6** | 8 | **0.04206** | s1 .0434, s2 .0406, s3 .0419, s4 .0413, s5 .0406, s6 .0439, s7 .0422, s8 .0426 |
| c1,c2,c3,c6 | 2 | 0.04312 | s1 .0446, s2 .0416 |
| c1,c2,c3,c5 | 2 | 0.04341 | s1 .0448, s2 .0421 |
| c1,c2,c3 | 3 | 0.04360 | s1 .0467, s2 .0421, s3 .0420 |
| c1,c3,c5,c6 | 2 | 0.04575 | s1 .0463, s2 .0452 |
| c1,c3,c4 | 2 | 0.04684 | s1 .0468, s2 .0469 |
| c1,c2,c3,c4,c5,c6 (all six) | 3 | 0.04699 | s1 .0445, s2 .0467, s3 .0497 |
| c2,c3,c5,c6 | 2 | 0.04829 | s1 .0497, s2 .0469 |
| c2,c3,c4,c6 | 1 | 0.04834 | s2 .0483 |
| c1,c2,c3,c4 | 2 | 0.04881 | s1 .0482, s2 .0494 |
| c1,c2,c3,c4,c6 | 2 | 0.04940 | s1 .0505, s2 .0483 |
| c2,c4 | 2 | 0.04950 | s1 .0463, s2 .0527 |
| c1,c2,c3,c4,c5 | 2 | 0.04967 | s1 .0533, s2 .0461 |
| baseline | 2 | 0.05153 | s1 .0515, s2 .0515 |
| c1,c4 | 1 | 0.05239 | s1 .0524 |
| c2,c3,c4 | 1 | 0.05417 | s1 .0542 |
| c1,c2 | 1 | 0.05888 | s1 .0589 |
| c1,c2,c4,c6 (teammate's plan) | 2 | 0.34660 | diverged on both seeds |

## What the evidence says

1. **c1 + c2 need c3.** Higher lr and higher momentum together are unstable: c1,c2 alone was the worst
   non-diverged arm (0.0589 vs baseline 0.0515) and the teammate's planned set c1,c2,c4,c6 diverged on
   both seeds tried. Adding gradient clipping (c3) turns the same pair into the best region of the space
   (c1,c2,c3 = 0.0436). This is why the single-change ablation showed c3 as "hurting": clipping only pays
   off once the lr/momentum are raised.
2. **Both c1 and c2 are needed on top of c3.** Removing c1 from the leader (c2,c3,c5,c6) costs +0.006 on
   both seeds; removing c2 (c1,c3,c5,c6) costs +0.003 to +0.005 on both seeds.
3. **c4 (wider hidden layer) hurts once c1/c2/c3 are in.** On paired seeds, c1,c2,c3,c4 is worse than
   c1,c2,c3 (s1 .0482 vs .0467, s2 .0494 vs .0421), and the all-six recipe is worse than the leader on
   all three shared seeds (.0445/.0467/.0497 vs .0434/.0406/.0419). c4's single-change gain does not
   survive combination.
4. **c5 and c6 are each small, consistent wins on top of c1,c2,c3, and additive.** On seeds 1 and 2:
   c1,c2,c3 = .0467/.0421; +c5 = .0448/.0421; +c6 = .0446/.0416; +c5+c6 = .0434/.0406. The full
   c1,c2,c3,c5,c6 arm beat every other arm on every seed it shared with them.

## Prediction

The shipped recipe was run on 8 distinct seeds: mean 0.04206, SD 0.0012, standard error 0.0004, no
divergences. Predicted expected val MSE = **0.0421**.

## Caveats

Runner-up arms (c1,c2,c3,c6 / c1,c2,c3,c5 / c1,c2,c3) were only measured on 2-3 seeds, so their ordering
among themselves is uncertain; but the shipped arm was better than each of them on every common seed,
and the gap to the next-best arm (~2.5%) is within the pass tolerance even if the ranking of those
runners-up is slightly off.
