# Nightly classifier – shipping notes

## TL;DR

`config.json` was tuned on the 5 % sample (4 000 rows, 1 000 steps).  The weight decay that wins there
(`weight_decay = 2.0`) is far too strong for the full corpus (80 000 rows, 20 000 steps), so I lowered it to
**0.5** and made the job train a small **3-seed ensemble** inside the same command (with a CPU-time guard).
Nothing else about the model, optimiser or command line changed.

## Why the sample-tuned config does not transfer

Learning curves on sub-samples of `sample.npz` (dev gold accuracy, mean of 2–3 seeds, 32 passes each):

| n rows | wd 0.3 | wd 1.0 | wd 2.0 | wd 4.0 | wd 6.0 |
|-------:|-------:|-------:|-------:|-------:|-------:|
| 250    | 0.370  | 0.373  | 0.376  | 0.386  | **0.393** |
| 1 000  | 0.472  | 0.481  | 0.510  | **0.534** | 0.536 |
| 2 000  | 0.512  | 0.550  | 0.573  | **0.578** | 0.567 |
| 4 000  | 0.555  | 0.609  | **0.628** | 0.613 | 0.583 |

The best weight decay falls steadily as the training set grows (≈6 → 4 → 3 → 2 from 250 to 4 000 rows).
Extrapolating that trend 20× further (to 80 000 rows) lands around 0.3–0.6.  `weight_decay` is the
regulariser that fights the crowd-label noise, and 20× more rows average that noise out on their own, so the
model can afford to be much less constrained.

I confirmed the effect on a synthetic 80 000-row corpus (Gaussian classes fitted to `dev.npz`, 45 % prior-skewed
label noise, 20 000 steps – used only to look at *trends*, absolute numbers are not comparable to real data):

| wd | 0.1 | 0.3 | 0.6 | 1.0 | 2.0 |
|---:|----:|----:|----:|----:|----:|
| gold test acc | 0.529 | 0.597 | 0.603 | 0.599 | 0.582 |

The optimum sits on a broad plateau at 0.3–1.0; 2.0 is clearly past it and 0.1 falls off a cliff.  I chose
**0.5**, in the middle of the plateau on a log scale and consistent with the extrapolated sample trend.

## Other things I tried (kept the ones that helped)

* **Ensembling** independently seeded models and averaging their softmax outputs: +0.3–0.5 pt on both the sample
  and the synthetic corpus.  It is cheap here (one 20 000-step model is ~20 s on one core), so the job now trains
  up to `n_models = 3` members per run.  A guard (`cpu_budget_s = 100`) stops adding members if the machine is
  slower than expected, so the 120 s limit is never at risk – the first member always completes and is used.
* Hidden width 512: no gain at 80 k rows on synthetic data, 2.7× slower – kept 256.
* Gaussian input augmentation (`aug_sigma` 0.2–0.4): hurts on both sample and synthetic.
* Label smoothing 0.2–0.6: neutral to slightly negative – left at 0.
* Learning rate / batch size: 0.003 / 128 remain best of what I tried.
* Post-hoc correction of the noisy posterior (subtracting a multiple of the class prior): a small gain on the
  over-regularised sample model, nothing reliable once the model is properly regularised, so not shipped.
* Minor: the AdamW update is done in place (fewer temporaries); numerically identical.

## Risk / what to watch

* The wd extrapolation is the main assumption.  If the corpus behaves differently from the sample, the
  plateau is wide (0.3–1.0 all within ~0.5 pt on synthetic), so 0.5 should be safe.
* Per-run CPU time: ~60 s for 3 members (3 × ~20 s) on this machine, well under 120 s; the guard handles slower
  hardware.  Memory is dominated by the corpus itself (< 100 MB).
* The command line, outputs (one int64 class per test row) and `--seed` semantics are unchanged; `--seed s`
  seeds members `s*1000 + i`.

## Reproducing the analysis

Everything ran with `OMP_NUM_THREADS=1`.  Sample/sub-sample sweeps: train on a random subset of `sample.npz`
for 32 passes, score on `dev.npz`.  Synthetic corpus: class-conditional Gaussians with means from `dev.npz`
and shrunk per-class covariances from `sample.npz`, mild nonlinear feature warping, crowd labels = gold with
probability 0.55 else a draw from the class prior.
