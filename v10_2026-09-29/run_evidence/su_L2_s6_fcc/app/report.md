# Nightly classifier – changes for the full-corpus run

## TL;DR

`config.json`'s `weight_decay: 3.0` was the right answer for 4 000 rows and the wrong one for 80 000.
`train.py` now holds out 10 % of the corpus, trains one model per candidate weight decay
(`wd_candidates` = 0.3, 1.0, 0.1, 3.0), ranks them on the hold-out, and predicts with the
softmax-average of the candidates within 1 pt of the best.  A CPU-time guard keeps the job under the
120 s cap.  The production command line is unchanged.

## Why the sweep did not transfer

* The sample's dev accuracy (0.74) is matched exactly by LDA (0.746), and the wd-sweep's optimum did
  not move between 500 and 4 000 rows.  With AdamW, `lr * wd = 0.009` per step shrinks the network into
  an essentially linear model – the best bias/variance trade-off with 4k noisy rows, but it throws away
  the non-linearity that 80k rows can support.
* The crowd noise is close to uniform (~20 % of labels flipped to a random class: classes 1/3/6 are
  ~0 % in gold but ~2–3 % in crowd labels; cross-fitted confusion rows are flat off-diagonal).  Under
  uniform noise the noisy-label argmax equals the clean argmax, so plain cross-entropy is fine and a
  *noisy* hold-out ranks models in the same order gold does (checked on the sample and on the synthetic
  corpus below).
* I could not access the corpus, so I built a stand-in: Gaussian X with the sample's mean/covariance, a
  random tanh-MLP teacher calibrated so a 4k/1000-step run gets ~0.74 (as on the real sample), 20 %
  uniform noise.  At 80k rows / 20 000 steps, on two independent teachers:

  | weight_decay | 0.03 | 0.1  | 0.3      | 1.0  | 3.0 (shipped) |
  |---|---|---|---|---|---|
  | clean test acc | 0.77 (overfits) | 0.84 | **0.87–0.88** | 0.83 | 0.75–0.78 |

  i.e. the shipped config would have landed *below* the 0.805 target, wd≈0.3 lands well above it.
  lr 0.001–0.01, label smoothing, width 128–512, depth 2–3 all changed things by ≤1.5 pt; weight
  decay is the lever.

## What the job does now

1. Standardise features on the whole corpus (labels not used).
2. Random 90/10 split (seeded by `--seed`).  For each `wd` in `wd_candidates`, train the same
   256-256 ReLU MLP for `--steps` steps on the 90 % and score the 10 % hold-out (noisy labels).
3. Keep every candidate within `ensemble_tol` (0.01) of the best hold-out accuracy; predict with the
   averaged softmax.  If only one is close, it is a single model.
4. `cpu_budget_s` (95 s) – before starting a candidate, the job checks that the running average
   per-candidate CPU time still fits; otherwise it stops and uses what it has.  On my single core
   each candidate takes ~20 s, so all four fit (~85 s); on a slower machine it degrades to 3 or 2
   candidates, the best-so-far (0.3, then 1.0) being trained first.

Nothing fitted is shipped: the repo is code + hyper-parameters only.

## Expected production behaviour

On the synthetic 80k corpus the full command (both seeds) selects wd=0.3 and scores ≈0.87–0.88.
Real-data extrapolation is less certain – the true decision function is unknown – but the mechanism
(pick regularisation on a hold-out of the actual training set) is exactly the one that stops a
5 %-sample sweep from being trusted blindly again.  If the corpus were somehow best served by wd=3
the job would still select it, since it is one of the candidates.

## Things I deliberately did not do

* Noise-transition correction / robust losses: unnecessary for uniform noise (argmax preserved).
* Use `dev.npz` inside the job: production has no gold labels; the noisy hold-out replaces it.
* Larger models: 3×256 gained <1 pt on the testbed for 1.6× the CPU – not worth the budget risk.
