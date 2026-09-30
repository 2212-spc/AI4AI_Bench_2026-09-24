# Recipe change triage: decision and evidence

**Ship: c4 + c5 + c6** (wider hidden layer 160, L2 weight decay 3e-4, decoupled minibatch RNG).
**Predicted expected val MSE: 0.0467** (observed mean over 9 seeds 0.04655, SE 0.00046, shrunk slightly
toward the cluster of near-equivalent configs to correct for winner's-curse selection bias).

## Method

80 runs total. Rather than one-seed ablations, every configuration was run on fixed, shared seeds
(seed 1, 2, ... ) so that comparisons between configurations are paired and seed noise partly cancels.
Seed-to-seed SD of val MSE for a fixed config is about 0.001-0.0015 (2-3% of the mean), so single-run
differences below ~0.003 are not meaningful; that is exactly the scale of most of the teammate's table.
Diverged runs were scored 0.3466 as specified.

## Key findings

| config | n | mean val MSE |
|---|---|---|
| c4,c5,c6 | 9 | 0.04655 |
| c1,c3,c4,c5 | 10 | 0.04685 |
| c4,c6 | 6 | 0.04727 |
| c1,c3,c4 | 6 | 0.04773 |
| c4,c5 | 6 | 0.04785 |
| c3,c4 / c4 | 2 / 2 | 0.0485 / 0.0486 |
| c1,c2,c3,c4 | 4 | 0.04864 |
| baseline | 2 | 0.05153 |
| c2,c3,c4 | 2 | 0.0544 |
| c1,c4 (no clipping) | 2 | 0.0555 |
| c1,c2,c4,c6 (teammate's plan) | 2 | diverged on both seeds -> 0.3466 |

1. **The teammate's plan (c1,c2,c4,c6) is unshippable.** Doubling the lr (c1) and raising momentum to 0.95
   (c2) together quadruple the effective step size; with the wider net (c4) it diverged on both seeds tried.
   Even c1+c2 without c4 was worse than baseline (0.0549). Single-change ablations cannot see this interaction.
2. **c4 (width 160) is the one large, robust win**: about -6% vs baseline, consistent across every seed and
   every combination it appears in.
3. **c5 (weight decay) is a small but consistent win**, contrary to the ablation's "no effect". In paired
   seeds it improved val MSE in 13 of 15 comparisons (c4 -> c4,c5; c1,c3,c4 -> c1,c3,c4,c5; c4,c6 -> c4,c5,c6;
   etc.), typically by 0.0002-0.0006.
4. **c1 (lr 0.06) only helps when paired with c3 (clipping)**. c1,c4 without clipping is worse than baseline
   (0.0555) and unstable; c1,c3,c4 is a clear gain over c4 alone in paired seeds. So c3 is not "harmful" as the
   ablation suggested; it is the enabler for c1. But c1,c3,c4,c5 (0.04685) ends up statistically tied with the
   lower-lr c4,c5,c6 (0.04655), so the higher learning rate buys nothing once width and weight decay are in.
5. **c2 (momentum 0.95)** is high-variance: single runs range 0.0446-0.0527 for the same config across two
   seeds, and with clipping it is clearly bad (c2,c3,c4 = 0.0544). The teammate's -13% was a lucky seed.
   Its expected value over many seeds is at best on par with the leaders and carries instability risk.
6. **c6** decouples data order from init; in expectation over seeds it should be roughly neutral, and the data
   agree (c4,c6 vs c4 and c1,c3,c4,c6 vs c1,c3,c4 differ by less than one SE in opposite directions). It is
   harmless and is included because the best-measured configuration contains it.

## Why c4,c5,c6 over c1,c3,c4,c5

Both are within one standard error of each other (difference 0.0003, SE ~0.0005). c4,c5,c6 keeps the production
learning rate and needs no clipping, so it has no divergence tail risk in the many-seed expectation; the higher-lr
variant relies on clipping to stay stable and showed no measurable benefit for that risk. All 15 runs in the
c4,c5,(c6) family and all 13 runs in the c1,c3,c4,(c5/c6) family finished without divergence.

## Prediction

Expected val MSE of the shipped recipe: **0.0467** (9-seed mean 0.04655; 95% CI roughly 0.0456-0.0475).
Improvement vs current production (0.0515): about -9%.
