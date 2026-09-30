# Nightly classifier: changes for the full-corpus run

## TL;DR

`config.json` was tuned on the 4000-row sample and does not transfer to the 80000-row corpus: its weight
decay (3.0) is ~10x too strong once there is 20x more data.  I re-tuned for the production scale using a
noise-aware Gaussian emulator of the corpus, and shipped a small ensemble: two MLPs with weight decay 0.3
plus a linear softmax on quadratic features.  On an 80000-row emulated corpus with 20000 steps the shipped
config scores 0.60 and the new one 0.72 (Bayes rate of the emulator: 0.75).  CPU time per run is ~40 s
(limit 120 s), memory ~560 MB, repo 20 KB, command line unchanged.

## What I found

* **Data looks class-conditionally Gaussian with class-specific covariances.**  On the gold dev labels,
  Mardia's multivariate kurtosis for the four big classes is within 1-3% of the Gaussian value; per-feature
  skew is ~0 and kurtosis ~3.  A regularised QDA on the noisy sample already beats the tuned MLP on dev
  (0.677 vs 0.663).  The Bayes decision rule for this kind of data is quadratic in the features.
* **Label noise is roughly 15-30% uniform flips.**  Crowd class frequencies are the gold frequencies pulled
  toward uniform (rare classes are inflated ~3x); the EM estimate of the flip rate is 0.16, the class-prior
  estimate ~0.27.  Under (near-)symmetric noise the arg-max of the noisy posterior equals the clean one, so
  plain cross-entropy is consistent; the noise mainly costs effective sample size and invites over-fitting.
* **The tuned weight decay is a small-data artefact.**  Learning curves on sub-samples (500..4000 rows) show
  wd=3.0 wins at every small n, but accuracy is still climbing steeply with n (0.54 -> 0.66), i.e. the model
  is data-starved, and a strong norm constraint is what keeps it from memorising noisy labels.  With 80000
  rows that constraint under-fits badly.

## How I chose the new hyper-parameters (no corpus access)

I fitted a 10-class Gaussian mixture with a uniform label-noise model by EM on sample + dev, generated an
80000-row emulated corpus (crowd labels with 16% flips, plus a stress variant with 27% flips) and a
20000-row gold test set, and ran the real `train.py` for 20000 steps on it.  Real-vs-emulated learning
curves at 1000/2000/4000 rows track each other for every configuration tried, so the emulator is a fair
guide to how hyper-parameters behave at scale.  Emulated-corpus accuracy (Bayes 0.750):

| configuration (20000 steps, 80000 rows)            | acc   | CPU s |
|----------------------------------------------------|-------|-------|
| shipped: MLP-256, wd 3.0                           | 0.601 | 18    |
| MLP-256, wd 1.0                                    | 0.678 | 18    |
| MLP-256, wd 0.3  (0.2 / 0.5 within 0.005)          | 0.714 | 18    |
| MLP-256, wd 0.03                                   | 0.635 | 18    |
| linear softmax on quadratic features, wd 0.01      | 0.716 | 4     |
| **2x MLP wd 0.3 + quadratic linear (shipped now)** | **0.724 / 0.7225 (seeds 0/1)** | 40 |
| same, 27%-noise stress corpus                      | 0.713 | 40    |

Things that did not help on the emulator: lr 0.001/0.01, hidden 512, batch 256, label smoothing 0.1,
longer warm-up, an MLP on quadratic features (unstable, ~0.57).  A generative noise-corrected QDA can
reach 0.74 but only if the noise rate is known to within a few points (it collapses at a wrong rate), so I
did not ship it.

## What changed in the repo

* `train.py`: same MLP, optimiser and schedule.  Added (a) an ensemble: `config.json["members"]` is a list
  of per-member overrides, trained sequentially from one RNG stream, log-probabilities averaged;
  (b) `hidden: 0` = multinomial logistic regression; (c) `quad_features: true` = standardised features
  plus all pairwise products (560 inputs), which makes the linear member a discriminatively trained
  quadratic classifier - the right model family for Gaussian classes, and it needs no noise-rate estimate.
  With a `members`-less config the script behaves exactly as before (verified: old config reproduces 0.663).
* `config.json`: weight decay 3.0 -> 0.3 for the MLPs; members as above.
* Nothing fitted is shipped; no data.  The command line is unchanged; output is int64, one class per row.

## Caveats

* Everything above the sample scale is measured on the emulator, not the real corpus.  The emulator's
  Bayes rate (0.75) is not the real one; the model trained on emulated data scores ~0.79 on the real dev
  set, which suggests the real classes are at least as separable as the emulated ones.
* The shipped config is meant for the full corpus.  On the 4000-row sample with 1000 steps it gives 0.61
  on dev (wd 0.3 is too weak for that little data); use `--config` with wd ~3 for prototype runs, or
  better, prototype on a larger slice.
* Runtime: ~40 s CPU here for 20000 steps (the old job took 18 s here, matching the team's ~20 s), so there
  is a 3x margin to the 120 s limit.  If production is slower than expected, drop the second MLP member
  from `config.json` (costs ~0.005 accuracy on the emulator).
