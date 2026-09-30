# Recipe change triage: decision and evidence

**Ship: c1 (peak lr 0.06), c2 (momentum 0.95), c3 (grad-norm clipping 1.0), c5 (weight decay 3e-4).**
**Drop: c4 (hidden 160), c6 (separate data-order RNG).**
**Predicted expected val MSE: 0.0417** (mean of 16 seeds, standard error 0.00036, no divergences).

All 80 runs were used; the full log is in `lab history` (also saved to `/app/history.json`).

## Why the teammate's plan (c1, c2, c4, c6) is wrong

The single-change ablation misses strong interactions, and the plan never ran the combination.
Run 3 (`c1,c2,c4,c6`, seed 1) **diverged**, which counts as MSE 0.3466. The mechanism is visible
in the pairs: higher lr and higher momentum are individually good but together are unstable
without clipping (`c1,c2` averages 0.0550 over 4 seeds, worse than baseline 0.0515), and the
wider layer (c4) makes the aggressive optimiser worse still (`c1,c2,c3,c4` 0.0486 vs `c1,c2,c3`
0.0430 on the same seeds; `c2,c4` 0.0490 vs `c2` 0.0460).

## Evidence for the shipped subset (means over matched seeds, MSE = 0.3466 for divergence)

| subset | n seeds | mean val MSE | s.e. |
|---|---|---|---|
| c1,c2,c3,c5 (shipped) | 16 | 0.0417 | 0.0004 |
| c1,c2,c3,c5,c6 | 13 | 0.0422 | 0.0003 |
| c1,c2,c3,c6 | 4 | 0.0429 | 0.0006 |
| c1,c2,c3 | 4 | 0.0430 | 0.0012 |
| c2 | 4 | 0.0460 | 0.0012 |
| c1,c3 | 4 | 0.0474 | 0.0011 |
| c2,c3 | 4 | 0.0475 | 0.0009 |
| c2,c3,c5 | 2 | 0.0478 | 0.0009 |
| c4 | 4 | 0.0481 | 0.0008 |
| c1,c2,c3,c4 | 4 | 0.0486 | 0.0003 |
| c1,c2,c3,c4,c5 | 2 | 0.0497 | 0.0036 |
| baseline | 1 | 0.0515 | |
| c1,c2 | 4 | 0.0550 | 0.0023 |
| c1,c2,c4,c6 | 1 | diverged (0.3466) | |

Key findings:

- **c3 (clipping) is the enabler, not a loss.** Alone it is slightly worse (0.0523 vs 0.0515 at seed 1),
  but it is what makes c1 + c2 safe: `c1,c2,c3` = 0.0430 vs `c1,c2` = 0.0550 on the same 4 seeds,
  with no divergence in any of the 45 runs that included c3.
- **c1 and c2 together (with c3) are the main gain.** `c1,c2,c3` beats `c2,c3` (0.0475) and
  `c1,c3` (0.0474) by roughly 10%.
- **c5 (weight decay) is a small but consistent win on top of c1,c2,c3.** Paired over seeds 1-4,
  `c1,c2,c3,c5` beat `c1,c2,c3` on every seed (mean 0.0420 vs 0.0430). It did nothing at baseline
  (matching the teammate's result) because the baseline optimiser is too weak for regularisation to matter.
- **c4 hurts once the optimiser is aggressive**, on every comparison tried (with or without c5).
- **c6 is neutral to slightly negative.** Paired on 13 common seeds, `c1,c2,c3,c5` minus
  `c1,c2,c3,c5,c6` = -0.0005 +/- 0.0005. c6 only changes which random stream orders the data, so no
  systematic effect is expected; I drop it to keep the recipe change minimal. Either choice is within
  the tolerance of the best subset.

## Prediction

`c1,c2,c3,c5`: 16 seeds, values from 0.0398 to 0.0448, mean 0.04168, standard error 0.00036 (about 0.9%).
No divergence was observed in any clipped run, so I predict **0.0417**.
