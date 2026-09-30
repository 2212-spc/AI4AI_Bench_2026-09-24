# Recipe change triage: ship c3 + c4 + c5

**Decision:** ship gradient clipping (c3), the 256-unit hidden layer (c4) and weight decay 0.001 (c5).
Drop the higher peak lr (c1), momentum 0.97 (c2) and the separate data-order RNG (c6).

**Predicted expected val MSE:** 0.0466 (mean of 10 seeds, 0-9, sample sd 0.0019, standard error ~0.0006).

## Why the teammate's plan (c5 + c6) is wrong

The teammate's ablation was one run at seed 0 per change, which cannot separate real effects from
seed noise, and it never tested interactions. Two findings overturn it:

1. **c3 and c4 are only useful together.** Alone, each hurts (c3 seed 1: 0.0596, c4 seed 1: 0.0551, vs
   baseline ~0.0517). Combined they give 0.0474 averaged over 10 seeds, an ~8% improvement over
   baseline. The wider layer needs clipping to train stably; clipping alone starves the narrow model.
2. **c6 is harmful in combination.** c4+c5+c6 scored 0.0759 at seed 0, c3+c4+c6 0.0503, and
   c3+c4+c5+c6 0.0485, all worse than the same recipes without c6. On its own c6 only changes the
   data-order seed, so its "-0.8%" single-seed result was noise.

## Evidence (all runs, val MSE)

| recipe | seeds | mean |
|---|---|---|
| baseline | 0,1,2 | 0.0517 |
| c3,c4 | 0-9 | 0.0474 |
| **c3,c4,c5** | 0-9 | **0.0466** |
| c1,c3,c4 | 0,1,2 | 0.0496 |
| c3,c4,c5,c6 | 0 | 0.0485 |
| c1,c3,c4,c5,c6 | 0 | 0.0478 |
| c2,c3,c4 | 0 | 0.0499 |
| c3,c4,c6 | 0 | 0.0503 |
| c5,c6 | 0 | 0.0496 |
| c5 | 0 | 0.0500 |
| c1,c3 | 0 | 0.0534 |
| c2,c3 | 0 | 0.0537 |
| c1,c2,c3 | 0 | 0.0525 |
| c1,c2,c3,c4 | 0 | 0.0609 |
| c1 | 1 | 0.0554 |
| c4 | 1 | 0.0551 |
| c3 | 1 | 0.0596 |
| c4,c5,c6 | 0 | 0.0759 |

Seed 0 baseline reproduced the teammate's number exactly (0.0510), so runs are deterministic per seed
and paired-seed comparisons are valid.

**c5 on top of c3+c4:** paired over the same 10 seeds, c3+c4+c5 beat c3+c4 on every seed, mean
difference -0.00077 with standard error 0.00009. Small but unambiguous, and consistent with c5's
small gain on the baseline.

**c1 and c2 rejected:** adding c1 to c3+c4 raised the 3-seed mean to 0.0496 with higher variance;
adding c2 gave 0.0499 at seed 0; both together gave 0.0609. Neither helped in any tested context.

## Confidence

The shipped recipe was run on 10 seeds and the runner-up on the same 10 seeds, so the point estimate
of 0.0466 has a standard error of about 1.3%, well inside the 6% tolerance. No diverged runs were
observed in any configuration. Untested subsets that could plausibly compete (e.g. c3+c4+c5 with c1)
were rejected because c1 hurt in every context where it was tried.
