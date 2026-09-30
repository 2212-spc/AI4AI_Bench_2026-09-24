# Nightly classifier: shipping on the full corpus

## TL;DR

`config.json`'s `weight_decay: 3.0` was tuned on the 4k-row sample and is far too strong for the 80k-row
corpus. `train.py` now picks weight decay in-job from a small candidate list using a held-out slice of the
crowd labels, then retrains on everything with a cheap 2-seed ensemble + tail weight averaging. The job stays
well inside the CPU limit (~60 s CPU, limit 120 s). The production command is unchanged.

## What was wrong

The sweep in `tune_log.csv` (4000 rows, 1000 steps) is very sharp in `weight_decay`: dev accuracy goes
0.538 → 0.566 → 0.645 → 0.673 → 0.677 for wd = 0.1 / 0.3 / 1 / 2 / 3, and drops again above 3. That optimum is
a *small-data* optimum: with only 4k noisy rows (crowd labels are ~30% wrong, consistent with uniform flips)
the model needs heavy shrinkage to avoid memorising label noise.

With AdamW the decay strength per step is `lr * wd`, and the production job runs 20× more steps, so the same
`wd = 3.0` shrinks the weights far harder in production than in the prototype. Meanwhile the 80k corpus
has 20× more signal, so the right amount of regularisation goes *down*, not up.

I could not test on the corpus, so I built two synthetic proxies (Gaussian inputs with the sample's
covariance, a random MLP teacher with gold-like class priors, 30% uniform label flips), calibrated so their
4k-row behaviour matches the real learning curve (real: 0.564 @1k → 0.677 @4k; proxy B: 0.60 → 0.66). Both
proxies reproduce the sample's preference for `wd = 3.0` at 4k rows. At 80k rows / 20k steps:

| weight_decay | proxy A test acc | proxy B test acc |
|---|---|---|
| 3.0 (shipped) | 0.735 | 0.683 |
| 1.0 | 0.737 | 0.697 |
| 0.5 | 0.737 | 0.707 |
| 0.3 | 0.737 | 0.703 |
| 0.1 | 0.708 | 0.676 |

So the shipped value costs ~0.02 on the harder proxy, and going too far (0.1) starts to fit noise.

## What I changed

1. **In-job weight-decay selection** (`weight_decay_candidates: [2.0, 0.7, 0.3]`). The job holds out 10% of
   the corpus, trains one shortened model (30% of the schedule) per candidate on the other 90%, scores it on
   the held-out *crowd* labels, then retrains on all rows with the winner. Under roughly symmetric label noise,
   accuracy on noisy labels is a monotone function of gold accuracy, so noisy validation ranks candidates
   correctly. On both proxies (80k rows, 8k-row holdout) it picks 0.3–0.7 consistently across seeds. It adds
   ~7 s CPU. This makes the job robust to future changes in corpus size rather than baking in a number tuned
   on one scale.

   Note for prototype runs on the 4k sample: the holdout is only 400 rows there, so selection is noisy and
   the candidate list (chosen for the full corpus) no longer contains 3.0; the sample→dev number with this
   config is ~0.63 rather than 0.68. That is expected and not what production sees. To reproduce the old
   prototype behaviour, set `"weight_decay_candidates": [3.0]` (or delete the key) and `"ensemble": 1`.
2. **Tail weight averaging** (`swa_frac: 0.3`): running average of the weights over the last 30% of the cosine
   schedule. Free, +0.000–0.003 on proxies.
3. **2-seed ensemble** (`ensemble: 2`): averages softmax outputs of two seeds. +0.002–0.004 on proxies;
   doubles training time but we had the budget.

Things tested and *not* adopted (no gain on either proxy at production scale): forward noise correction,
small-loss trimming, mixup, label smoothing 0.1–0.3, input noise, larger hidden (384/512), lr 0.0015/0.006,
batch 256, 3-seed ensemble.

## Cost

Single core, single-threaded BLAS, 80k rows, 20k steps, exact production command: ~51 s CPU on the dev box
(selection ~7 s, 2× training ~22 s each), ~90 MB RSS. The old job was ~26 s. If the production box is much
slower than the dev box, drop `ensemble` to 1 first (~30 s).

## Caveats

* No rows, weights, or fitted statistics are shipped; everything is fitted at run time from `--data`.
* The proxy numbers are for calibration of *trends* only; the absolute production score will differ.
* Selection uses `holdout_frac` of the corpus for validation only during the short candidate runs; the final
  models train on all rows.
