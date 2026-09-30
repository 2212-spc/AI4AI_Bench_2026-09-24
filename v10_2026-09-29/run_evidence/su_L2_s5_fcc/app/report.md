# Nightly classifier: changes for the full-corpus run

## Summary

The shipped `config.json` (weight decay 3.0) was tuned on the 5% sample (4000 rows, 1000 steps).
That weight decay is far too strong for the full corpus (80000 rows, 20000 steps). `train.py` now
holds out a slice of the corpus, trains one model per candidate weight decay, and predicts with
whichever candidate(s) score best on the held-out slice. The loss also models the crowd-label noise
explicitly. The command line, inputs and outputs are unchanged.

## Why the old config does not transfer

Weight decay acts as a capacity cap. Its best value depends on how much the network can memorise
wrong crowd labels, which shrinks as the corpus grows. Learning curves on sub-samples of the sample
(dev accuracy, gold labels, 32 passes over the data, mean of 3 seeds):

| rows | wd 0.3 | wd 1.0 | wd 2.0 | wd 3.0 | wd 5.0 |
|-----:|------:|------:|------:|------:|------:|
| 1000 | 0.521 | 0.536 | 0.556 | 0.573 | 0.599 |
| 2000 | 0.551 | 0.592 | 0.636 | 0.644 | 0.629 |
| 4000 | 0.563 | 0.634 | 0.692 | 0.695 | 0.662 |

The optimum moves down by roughly a factor of two per doubling of the data. Extrapolated to
80000 rows it lands around 0.1 to 0.3. A synthetic stand-in with the same shape (32 features,
10 classes, dev-like class priors, 29% uniform label noise, difficulty matched so that 4000 rows
give about 0.70) confirms it at production scale (80000 rows, 20000 steps, gold test accuracy):

| wd | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 | 10 |
|---:|-----:|----:|----:|----:|----:|---:|
| acc | 0.899 | 0.952 | 0.951 | 0.923 | 0.825 | 0.638 |

Shipping wd 3.0 would have left more than ten points on the table in that simulation.

## What changed

1. **Held-out selection instead of a fixed weight decay.** `train.py` holds out 10% of the corpus
   (crowd labels, capped at 10000 rows) and trains one model per entry of
   `weight_decay_candidates` = [0.3, 0.1, 1.0, 3.0], each for the full `--steps`. Crowd-label
   accuracy on the held-out slice picks the model. Under roughly uniform noise, crowd accuracy is
   an affine function of gold accuracy, so it ranks models correctly. With 8000 held-out rows the
   standard error is about half a point. On the sample the ranking matched dev gold accuracy for
   every weight decay tried.
2. **Optional averaging of the best candidates.** The averaged softmax of the top models is used
   only if it beats the best single model on the held-out slice by at least 0.3 points and every
   included model is within 2 points of the best. This protects against a noisy pick.
3. **Forward label-noise correction in the loss.** The cross-entropy is computed on
   q = (1 - e) softmax + e / K with `noise_rate` e = 0.35, i.e. a crowd label is assumed correct
   with probability 1 - e and uniform otherwise. The rate was estimated from the gap between the
   crowd class frequencies in the sample and the gold frequencies in dev (about 0.29). On the sample
   this adds about 1 to 1.5 points at every weight decay. On the synthetic stand-in at production
   scale it adds about 1.2 points.
4. **CPU-time guard.** Candidates are trained in order of prior plausibility (0.3, 0.1, 1.0, 3.0)
   and the loop stops early if the next candidate would push CPU time past `cpu_budget` = 100 s.
   On this machine the full run takes about 75 s of CPU (4 candidates at 18 to 19 s each).

Unchanged: model (256-256 ReLU MLP), AdamW with warmup and cosine decay, lr 0.003, batch 128,
standardised inputs, no augmentation, no label smoothing.

## Evidence

* Synthetic stand-in, 80000 rows, 20000 steps, exact production command through `train.py`:
  seed 0 chooses wd 0.3 and scores 0.960 gold test accuracy; seed 1 chooses 0.3 and scores 0.960.
  A harder variant chooses wd 0.1 and scores 0.928. The oracle single model in each case is within
  half a point.
* Real sample (4000 rows, 1000 steps, 400 held-out rows): mean dev accuracy over 5 seeds 0.680,
  versus 0.695 for the old fixed config. The small held-out slice makes selection noisy at this
  size. This is the regime the new procedure is not designed for. Production has 20x more rows in
  the held-out slice.
* Seed ensembles at the sample size gave no gain (0.695 to 0.700), so the extra CPU budget is spent
  on weight-decay candidates rather than on repeated seeds.

## Risks and follow-ups

* If the production machine is much slower than this one, the CPU guard drops candidates from the
  end of the list. Even with only the first candidate (wd 0.3) the run is expected to beat the old
  config on the full corpus.
* The noise-rate estimate assumes roughly uniform flips. The confusion of held-out crowd labels
  against model predictions on the sample looks uniform apart from a mild excess in class 8.
* Once the production run has happened, log the per-candidate held-out accuracies (they are printed)
  and consider narrowing the candidate list around the winner to free CPU time for a final model on
  100% of the corpus.
