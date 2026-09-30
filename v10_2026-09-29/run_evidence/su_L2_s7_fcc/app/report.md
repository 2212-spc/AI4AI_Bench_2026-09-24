# Nightly classifier: changes for the full-corpus run

## Summary

`train.py` no longer trains one fixed configuration.  It holds out 10% of the corpus (crowd labels), trains a
short list of candidate configurations that differ mainly in weight decay, scores them on the held-out split,
and averages the class probabilities of the candidates that are statistically tied with the best one.  The
loss is cross-entropy through a symmetric label-noise model.  The production command line is unchanged; the run
uses roughly 80 s of CPU on a machine where the old job took 20 s, with a CPU-budget guard so that it can never
run out of the 120 s limit by training too many candidates.

## Why

**The tuned config does not transfer to 20x more data.**  `config.json` was picked on the 4000-row sample, where
the dominant problem is fitting crowd-label noise.  There, heavy weight decay (3.0) wins by a wide margin.  But
the learning curves on sub-samples of the sample show that the lower-decay models gain much faster with data:

| rows | wd 0.3 | wd 1.0 | wd 3.0 |
|------|--------|--------|--------|
| 1000 | 0.549  | 0.561  | 0.591  |
| 2000 | 0.565  | 0.600  | 0.643  |
| 4000 | 0.611  | 0.645  | 0.670  |

(dev accuracy, noise-corrected loss, mean of 5 sub-samples, 32 passes each.)  Extrapolating the tuned
configuration's own curve (0.537 / 0.593 / 0.635 / 0.654 / 0.665 at 500 / 1000 / 2000 / 3000 / 4000 rows, with
gains shrinking each doubling) predicts roughly 0.72-0.73 at 80000 rows, below the 0.735 target.  Which
weight decay is best at 80000 rows cannot be determined from the sample, so the job decides it at
production time on data it actually has: a held-out slice of the corpus.  Crowd labels are noisy, but with
symmetric noise the noisy accuracy is a monotone function of the true accuracy, and with 8000 validation rows
its standard error is about 0.006, enough to separate candidates that differ meaningfully.

**Noise-aware loss.**  About a quarter of the crowd labels look uniformly wrong (the crowd class histogram is
the gold histogram shrunk towards uniform by ~0.25, and a model that is 65% accurate on gold agrees with only
53% of held-out crowd labels).  The training loss now models the observed label as
`(1 - r) * softmax + r / K` with `r = 0.25`.  This gives a consistent +0.01 on dev at every sample size and
weight decay tried, more for the lower-decay models, and it is insensitive to `r` in 0.2-0.4.

**Ensembling instead of retraining.**  After selection the job does not retrain the winner on 100% of the rows;
that would cost a fourth of the budget for about 0.2 points (10% fewer rows at the flat end of the learning
curve).  Instead it averages the probabilities of every candidate within one standard error of the best
validation accuracy, which is at worst neutral and usually a small gain.

## What was checked

* On the sample (4000 rows, 1000 steps) the new job reaches the same dev accuracy as before (0.668 / 0.643 for
  seeds 0 / 1 vs 0.663 baseline); with only 400 validation rows the selection is weak there, as expected.
* On a synthetic 80000-row corpus with the same feature statistics, class imbalance and 27% uniform label
  noise (random nonlinear teacher, Bayes accuracy 0.80, 20000-row test set) the old job scores 0.612 / 0.613
  (seeds 0 / 1) and the new job 0.687 / 0.686.  The held-out split ranked the candidates in the same order
  as the gold test set; weight decay 3.0 came last by a wide margin, confirming that the sample-tuned
  configuration under-fits at full scale.  This is a synthetic check of the mechanism, not a forecast of the
  production number.
* CPU time: each candidate costs the same as the old job (about 19 s on the dev box).  `cpu_budget_s: 85`
  means the fourth candidate is only trained if the first three finished in time; on a machine twice as slow
  the job trains two candidates and still finishes well inside 120 s.  Peak memory is under 100 MB.

## Things to keep in mind

* Candidate order matters when the budget bites: the list is ordered by how likely each configuration is to
  be best at full scale: weight decay 1.0, then 0.3, then the old 3.0, then 0.1.
* If the corpus is small (validation split under 200 rows) the job falls back to training only the first
  candidate on all rows.
* Nothing fitted on the sample or dev set ships in the repo; the repo is code and `config.json` only.
