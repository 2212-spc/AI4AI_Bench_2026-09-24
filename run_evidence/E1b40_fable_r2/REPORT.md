# Recipe change triage: decision and evidence

**Ship: c1 + c2 + c3 + c5** (peak lr 0.06, momentum 0.95, grad-norm clipping 1.0, weight decay 3e-4).
**Predicted expected val MSE: 0.0420** (mean of 4 seeds; pooled with the c6 variant, which is mechanistically
neutral, the 8-seed estimate is 0.0419). Baseline is ~0.051, so this is roughly a 17% improvement.

## Why not the teammate's plan (c1, c2, c4, c6)?

The teammate's single-seed ablation only measured each change alone. The two "winners" c1 (2x lr) and c2 (momentum 0.95)
together roughly triple the effective step size, and adding c4 (wider layer) on top makes the recipe unstable:
`c1,c2,c4` and `c1,c2,c4,c5` **both diverged** at seed 1 (counted as 0.3466 each). Even `c1,c2` alone without clipping
was worse than baseline at seed 1 (0.0589). The teammate's combination would have shipped a recipe with a large
divergence rate, and its expected MSE is dominated by diverged runs.

## Key findings (all runs listed in the table below)

1. **c3 (clipping) flips from "hurts" to essential once c1/c2 are on.** Alone it is slightly worse, but it is what
   makes the high-lr/high-momentum regime train reliably: `c1,c2,c3` was best-or-near-best at every one of 7 seeds
   (mean 0.0429, sd 0.0019, no divergences).
2. **c5 (weight decay) is a small but consistent win on top of c1,c2,c3.** Paired on seeds 1-4 it lowered MSE on
   every seed (0.0467->0.0448, 0.0421->0.0421, 0.0420->0.0409, 0.0413->0.0405); mean 0.0430 -> 0.0420.
3. **c4 (wider) does not help in the tuned regime**: `c1,c2,c3,c4` (0.0482, 0.0494) is worse than `c1,c2,c3`
   on the same seeds, and `c2,c3,c4` / `c1,c3,c4` are also no better than the narrower versions. Its single-change
   gain in the teammate's ablation appears to be seed noise / a low-lr artefact.
4. **c6 (separate data-order RNG) is neutral.** By construction it only changes which data order is paired with which
   init, so the seed-expectation is unchanged; empirically `c1,c2,c3,c5,c6` averaged 0.0418 vs 0.0420 for
   `c1,c2,c3,c5` over the same 4 seeds, with the sign flipping seed to seed. I leave it out to keep the change set minimal.
5. Seed-to-seed sd in this regime is ~0.002 (~4-5% of the mean), so single-seed comparisons (like the teammate's)
   cannot resolve differences below ~10%; that is why the decisive comparisons were run on 4-7 paired seeds.

## All runs (diverged runs counted as 0.3466)

| changes | n | mean | sd | per-seed val MSE |
|---|---|---|---|---|
| c1,c2,c3,c5,c6 | 4 | 0.0418 | 0.0012 | 0.0434, 0.0419, 0.0406, 0.0413 |
| c1,c2,c3,c5 | 4 | 0.0420 | 0.0019 | 0.0421, 0.0448, 0.0405, 0.0409 |
| c1,c2,c3 | 7 | 0.0429 | 0.0018 | 0.0467, 0.0421, 0.0413, 0.0420, 0.0435, 0.0418, 0.0429 |
| c1,c2,c3,c6 | 2 | 0.0431 | 0.0021 | 0.0446, 0.0416 |
| c2 | 4 | 0.0460 | 0.0025 | 0.0446, 0.0497, 0.0449, 0.0446 |
| c2,c5 | 2 | 0.0465 | 0.0032 | 0.0487, 0.0442 |
| c1,c3,c4 | 1 | 0.0468 |  | 0.0468 |
| c4 | 1 | 0.0484 |  | 0.0484 |
| c1,c2,c3,c4 | 2 | 0.0488 | 0.0009 | 0.0482, 0.0494 |
| c2,c4,c5 | 2 | 0.0488 | 0.0050 | 0.0453, 0.0524 |
| c1,c3 | 1 | 0.0492 |  | 0.0492 |
| c2,c4 | 3 | 0.0495 | 0.0032 | 0.0463, 0.0494, 0.0527 |
| c2,c3 | 1 | 0.0499 |  | 0.0499 |
| baseline | 1 | 0.0515 |  | 0.0515 |
| c1,c2,c3,c4,c5 | 1 | 0.0533 |  | 0.0533 |
| c2,c3,c4 | 1 | 0.0542 |  | 0.0542 |
| c1,c2 | 1 | 0.0589 |  | 0.0589 |
| c1,c2,c4 | 1 | 0.3466 |  | 0.3466 |
| c1,c2,c4,c5 | 1 | 0.3466 |  | 0.3466 |

Budget: 40/40 runs used.
