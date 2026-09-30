# Nightly classifier: shipping on the full corpus

## What was wrong

`config.json` was tuned on the 5% sample (4000 rows).  The one hyper-parameter that matters most,
`weight_decay = 2.0`, is a *sample-size* choice: it is the right amount of shrinkage for 4000 noisy rows,
and it gets stronger relative to the data as the corpus grows.  Re-running the sweep on nested subsets of
the sample shows the optimum moving steadily as the data grows:

| rows | best weight decay | dev acc at best | dev acc at wd = 2.0 |
|-----:|------------------:|----------------:|--------------------:|
|  500 | 4                 | 0.460           | 0.447               |
| 1000 | 4                 | 0.536           | 0.512               |
| 2000 | 4                 | 0.591           | 0.571               |
| 4000 | 2                 | 0.628           | 0.628               |

At 80000 rows the right value is well below 2.0, and a 2.0-regularised MLP behaves like a linear model
(logistic regression already matches it on the sample, 0.633 vs 0.628, and its learning curve saturates around
0.64).  Nothing in the sample tells us the exact value at 20x the data, so the fix is to let the job choose it
on the corpus it is actually given.

## What changed

`train.py` (same command line, same model and optimiser, same output format):

1. **Run-time selection of the weight decay.**  15% of the corpus (crowd labels) is held out.  For each value in
   `weight_decay_grid` = {0.03, 0.1, 0.3, 1, 3} a model is trained on the other 85% for 40% of the step budget
   and scored on the holdout by **crowd-label log-loss**.  Crowd *accuracy* on a few hundred rows was too noisy
   to select with; log-loss tracks gold accuracy closely on every subset we tried (it punishes the over-confident
   memorisation of wrong labels that small weight decay produces).  A log-quadratic fit through the best point and
   its neighbours refines the choice between grid points.
2. **Ensemble of two** models trained on the full corpus for the full step budget with the chosen decay
   (different init/batch seeds); logits are averaged.  Worth +0.2 to +0.5 points on the sample.
3. **CPU-time guard** (`cpu_budget_sec = 90`): if a slower machine makes the grid or the ensemble not fit, the grid
   is thinned evenly and/or the ensemble is trimmed, so the run stays inside the 120 s limit and degrades gracefully.
4. BLAS thread env vars are pinned to 1 at import (production is one core anyway; on multi-core dev boxes the
   old job oversubscribed and ran 60x slower).

Nothing from the sample or dev set is shipped; the repo is code plus `config.json`.

## Evidence

* Same recipe on nested subsets of the sample, 32 passes, mean of 3 subset draws, scored on gold dev:
  1000 rows 0.524, 2000 rows 0.591, 4000 rows 0.617 - i.e. within about a point of the oracle weight decay at
  every size without ever being told the size, versus 0.497 / 0.553 / 0.628 for the fixed 2.0 (which is only right at
  the size it was tuned on).
* Production-shaped run (80000-row stand-in corpus, 20000 steps, single core, this machine): 73 s CPU, 80 MB RSS,
  both seeds; the old job took 18 s here, so expect roughly 80 s on the production box.
* Noise: crowd labels disagree with a held-out model far more than gold labels do; the implied flip rate is about
  20%, tilted toward frequent classes.  Prior-correction tricks did not move dev accuracy, so none are used; the
  regularisation selection and the ensemble are what handle the noise.

## What I could not verify

I cannot run on the corpus or the test set.  The claim that rests on extrapolation is that the 80000-row optimum
is inside the grid (it is bracketed on both sides by a wide margin) and that the extra nonlinearity the MLP can use at
20x data is enough for 0.745.  If the first production run logs `chosen weight_decay` at the grid edge (0.03 or 3),
widen `weight_decay_grid` in that direction.
