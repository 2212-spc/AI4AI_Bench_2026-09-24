# Nightly classifier: shipping on the full corpus

## What changed

* `config.json`: weight decay is no longer a fixed value. It is now a small candidate grid centred on an
  estimate that scales with the corpus size, and the job picks among the candidates using a held-out slice
  of the training corpus.
* `train.py`: implements that selection. Model, optimiser, schedule, and the command line are unchanged.
  New keys: `wd_ref_n`, `wd_alpha`, `wd_multipliers`, `val_frac`/`val_min`/`val_max`, `cpu_budget_s`.

## Why

The sweep in `tune_log.csv` picked `weight_decay = 2.0` on 4000 rows. Weight decay is doing the job of
fighting label noise, and the right amount depends strongly on how much data there is. Re-running the sweep
on sub-samples of the 5% sample (32 passes each, 3 seeds, scored on `dev.npz`) gave:

| rows  | best weight decay | dev acc |
|-------|-------------------|---------|
| 1000  | ~4                | 0.548   |
| 2000  | ~3                | 0.607   |
| 4000  | ~1.5-2            | 0.655   |

The optimum roughly follows `wd ~ n^-0.6`. Extrapolated to 80000 rows that is about 0.35, not 2.0.
Shipping 2.0 unchanged would over-regularise the production model by a factor of ~6. On the sample, being
3x off the optimum costs 2-4 points of dev accuracy, so 6x off would give away a large part of the gain
from the extra data.

Because the extrapolation is only an estimate, the job does not trust it blindly. It holds out 5% of the
corpus (crowd labels, 4000 rows in production), trains one model per candidate weight decay
(centre x {1/3, 1, 3}), and predicts with the candidate that scores best on the held-out slice. On the
sample, crowd-label validation accuracy ranked the candidates the same way gold dev accuracy did in every
seed tried, so no gold data is needed at train time and nothing fitted on `dev.npz` ships in the repo.

## Cost and safety

* Three candidates at 20000 steps each took 79 s CPU on an 80000-row synthetic corpus (single thread,
  `hidden = 256`), under the 120 s limit. A CPU-time guard (`cpu_budget_s = 100`) skips remaining
  candidates if the machine is slower; the centre candidate always runs first, so a cut still leaves the
  best prior guess.
* Repo is 3 files, ~12 KB. No data or weights are shipped.
* On the sample (`--steps 1000`) the shipped job scores 0.657 / 0.642 on dev for seeds 0 / 1, versus 0.651
  for the old fixed config with seed 0.

## Things tried that did not help on the sample

Larger hidden width (512 gave +0.7 pt but 3x the CPU), label smoothing, Gaussian input noise, EMA weights,
generalised cross-entropy, and a 3-model ensemble (+0.5 pt, too expensive at production scale). Weight decay
scaling is the one change that matters.
