# Recipe change triage — decision

**Ship: c1 (peak lr 0.1), c2 (momentum 0.97), c3 (grad clip 0.3), c5 (weight decay 1e-3).**
**Drop: c4 (hidden 256), c6 (separate data-order RNG).**
Predicted expected val MSE: **0.0400** (baseline ≈ 0.050, i.e. ~20% lower).

## Why the teammate's plan (ship c5+c6 only) is wrong

The teammate's ablation used one seed per arm and tested each change in isolation. Two things
it could not see:

1. **Strong interactions.** c1 and c2 each look bad alone (they push the effective step size up
   and the run becomes unstable), but together with c3 (clipping) the instability is removed and
   the higher effective learning rate is a large win. c5 adds a further consistent gain on top.
2. **Divergence dominates expected MSE.** A diverged run counts as 0.3381, ~8x a normal run, so
   any subset with even a few-percent divergence rate is disqualified regardless of its
   non-diverged average.

## Evidence (40 runs total; all runs in `lab history`)

### Stage 1: 16-run half-fraction factorial over c1..c5 (c5 = c1·c2·c3·c4 alias), 1 seed each

| subset | val MSE | | subset | val MSE |
|---|---|---|---|---|
| baseline | 0.0501 | | c1,c5 | 0.0481 |
| c4,c5 | 0.0784 | | c1,c4 | **diverged** |
| c3,c5 | 0.0558 | | c1,c3 | 0.0506 |
| c3,c4 | 0.0482 | | c1,c3,c4,c5 | 0.0478 |
| c2,c5 | 0.0449 | | c1,c2 | **diverged** |
| c2,c4 | **diverged** | | c1,c2,c4,c5 | **diverged** |
| c2,c3 | 0.0517 | | c1,c2,c3,c5 | **0.0387** |
| c2,c3,c4,c5 | 0.0478 | | c1,c2,c3,c4 | 0.0579 |

Findings: every divergence occurred in a subset containing (c1 or c2) **without c3**; every subset
containing c3 finished. c4 was never in a top subset (its best showing was ~0.048, and it makes
lr/momentum increases diverge without clipping). The clear winner was c1+c2+c3+c5.

### Stage 2: multi-seed confirmation (24 runs)

| subset | n seeds | mean val MSE | sd | s.e. | diverged |
|---|---|---|---|---|---|
| c1,c2,c3,c5 | 11 | **0.0400** | 0.0015 | 0.0005 | 0/11 |
| c1,c2,c3,c5,c6 | 9 | 0.0402 | 0.0018 | 0.0006 | 0/9 |
| c1,c3,c5 | 1 | 0.0448 | | | 0/1 |
| c2,c3,c5 | 2 | 0.0476 | 0.0000 | | 0/2 |
| c1,c2,c3 | 2 | 0.0490 | 0.0014 | | 0/2 |

- Removing any one of c1, c2 or c5 from the winner costs 0.005–0.009 (10–20% worse), far
  outside the seed noise (s.e. ≈ 0.0005). Removing c3 causes divergence.
- c6 vs no-c6: 0.0402 vs 0.0400, difference well inside one standard error. This is expected:
  c6 only changes how the seed maps to (init, data order); it does not change the distribution of
  either, so the seed-averaged MSE is the same by construction. We drop it to keep the recipe
  change minimal (either choice would be acceptable). Pooling both arms (20 seeds, since c6 does
  not change the expectation) gives 0.0401, consistent with the 0.0400 prediction.
- No divergence in 20/20 runs of the shipped configuration (with c3 clipping active); a residual
  divergence rate above ~1% would be needed to move the expectation by >0.003 and is not supported
  by the data or by the mechanism (clipping bounds the update size).

## Prediction

Expected val MSE of the shipped recipe (c1,c2,c3,c5): **0.0400 ± 0.0005** (11 seeds directly,
20 seeds if the c6 arm is pooled).
