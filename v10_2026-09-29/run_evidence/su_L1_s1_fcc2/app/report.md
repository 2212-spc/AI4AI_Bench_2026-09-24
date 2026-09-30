# Nightly classifier – shipping on the full corpus

## TL;DR

The config that won the sweep on the 4 000-row sample (`weight_decay = 3.0`) was tuned for a
data-starved regime.  At 80 000 rows it over-regularises: on synthetic corpora built to mimic our
data it costs ~5–6 points of gold accuracy versus a properly scaled weight decay.  `train.py` now
scales weight decay with corpus size (3.0 at 4 000 rows → 0.67 at 80 000 rows), averages the tail
of each run, and trains a 2-model ensemble inside a CPU budget.  Production command line,
step count and batch size are unchanged; the job needs ~40 s CPU / ~80 MB.

## What was wrong

* **The sweep only tuned for n = 4 000.**  Re-running the sweep on subsets of the sample
  (1 000 / 2 000 / 4 000 rows, 32 passes each) shows the optimum `weight_decay` falls as the data
  grows: ≈5 → 3 → 2–3.  The 32-passes-over-noisy-labels regime needs strong shrinkage when the
  network can memorise the 4 000 rows; with 20× more rows the same shrinkage just under-fits.
* **The problem is not linear, so more data really helps.**  LDA and the MLP tie at ~0.68 on the
  sample, and a shared-covariance Gaussian fit to the gold dev set has a Bayes rate of only ~0.72,
  below the 0.785 target.  The 0.785 bar is therefore only reachable by an MLP that is allowed to
  use the extra rows – which is exactly what `weight_decay = 3.0` prevents.
* **Label noise ≈ 35 % and it looks (mostly) class-independent.**  Estimated from class-mean
  shrinkage between crowd and gold labels (crowd class means are ~0.6–0.75× the gold means) and
  from the sample→dev confusion structure.  Symmetric noise leaves the arg-max of the posterior
  intact, so plain cross-entropy + adequate regularisation is enough; no relabelling or noise
  transition matrices were needed (and none are shipped).

## What changed in `/app/repo`

`config.json`

| key | value | meaning |
|---|---|---|
| `weight_decay` | 3.0 | unchanged – the value tuned at `wd_ref_n` rows |
| `wd_ref_n` / `wd_scale_pow` | 4000 / 0.5 | effective wd = 3.0 · sqrt(4000 / n) → **0.67 at n = 80 000** (clamped to [`wd_min`=0.3, `wd_max`=5]) |
| `avg_frac` | 0.25 | uniform average of the weights over the last 25 % of steps (SWA-style) |
| `ensemble` | 2 | train 2 independently seeded models, average their softmax outputs |
| `cpu_budget_s` | 80 | stop adding ensemble members if the next one would exceed this CPU time |

`train.py` – model, optimiser, schedule and I/O are untouched.  Added `effective_wd()`, tail weight
averaging in the training loop, a budget-guarded ensemble loop, and probability-averaging
prediction.  Running with `--steps 1000` on the sample still gives 0.68–0.69 dev accuracy, i.e.
the prototype behaviour is preserved (wd is exactly 3.0 there).

## Evidence

Because the full corpus isn't available, I built synthetic populations from the dev-set class
statistics: gold class means / shared covariance from `dev.npz`, per-class sub-clusters to give
the nonlinear structure the real data evidently has, 35 % uniform label noise, 32 passes.  Two
of them ("A" and "B") were calibrated so the *old* pipeline on 4 000 noisy rows / 1 000 steps
scores ~0.68–0.71 – like the real sample.  Gold-label test accuracy at 80 000 rows / 20 000 steps:

| weight decay | linear proxy (Bayes ≈ 0.72) | proxy A | proxy B |
|---|---|---|---|
| 3.0 (shipped before) | 0.691 | 0.732 | 0.746 |
| 1.0 | 0.715 | 0.792 | 0.785 |
| **0.67 (new, sqrt scaling)** | 0.715 | ≈0.799 | **0.788** |
| 0.3 | 0.708 | 0.795 | 0.783 |
| 0.1 | 0.664 | 0.747 | – |

The optimum is broad (0.3–1.0) and the sqrt rule lands in the middle of it.  Full production-shaped
run of the shipped repo on proxy A (`--steps 20000`, seed 0): **0.798**, 38 s CPU, 78 MB RSS.

Things that did *not* move the needle at 80 k on the proxies (all within ±0.3 pt): hidden 512,
label smoothing 0.1, lr 0.002, batch 256.  Ensembling and tail averaging add a few tenths and
mostly reduce seed variance, which is what the two-seed production score rewards.

## Risks / notes

* The sqrt scaling is an extrapolation from n ≤ 4 000 plus the synthetic corpora; but the
  proxies show the accuracy surface is flat over a 3× range of wd around the chosen value, so the
  precise exponent isn't critical.  If the corpus size changes again the rule adapts automatically.
* CPU budget: single-core run is ~40 s on this box (2 × 19 s).  The `cpu_budget_s` guard drops
  the second member if the machine is >2× slower, so the 120 s limit is safe either way.
* Scratch experiments live in `/app/work` (not part of the repo); nothing derived from the sample
  or dev set is shipped.
