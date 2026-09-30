# Nightly classifier: shipping on the full corpus

## TL;DR

The `config.json` we tuned on the 4 000-row sample would have been badly wrong on the 80 000-row corpus
because **`weight_decay` was tuned in the wrong units**. It now scales with the corpus size automatically,
and the job trains a 3-member ensemble (still well inside the CPU budget). On a synthetic corpus built to
match the sample (same feature/label structure, 80k rows, same crowd noise), the exact production command
scores **0.859** (mean of seeds 0 and 1); on a deliberately harder synthetic corpus, **0.767**, versus
~0.61 for the old config. The target is 0.805.

## What was wrong

`weight_decay` in AdamW is *per-step* shrinkage. With `--steps` fixed at 32 passes over the data, the
80k corpus gets 20× more steps than the sample, so `weight_decay = 3.0` shrinks the weights 20× harder
in total. The sweep in `tune_log.csv` found 3.0 optimal for **4 000 rows / 1 000 steps**; that value is
far into the over-regularised regime at 80 000 rows / 20 000 steps.

Evidence (all numbers are gold-label accuracy):

* Learning curves on real data, subsampling the sample (dev accuracy, 32 passes each):
  wd 3.0 is best at 4k rows (0.732) but at 1k rows wd 5.0 is better and at 500 rows the optimum keeps
  moving — the optimum decreases as *n* grows.
* Synthetic corpus (noise-corrected class-conditional Gaussians fitted on the sample, 18% uniform label
  noise, matched class priors; verified that the 4k/1000-step behaviour reproduces the real sweep):
  at 80k rows, wd 3.0 → 0.61–0.73, wd 1.0 → 0.83, **wd 0.3 → 0.856**, wd 0.1 → 0.837.
  At 20k rows the optimum is ~1.0; at 4k it is ~3. Empirically the optimum follows roughly
  `wd ∝ n^-0.75`.

## What changed

`train.py`

1. **Size-scaled weight decay.** `weight_decay` is interpreted relative to `weight_decay_ref_n`
   (= 4 000, the sample size it was tuned on): effective wd = `wd * (ref_n / n) ** weight_decay_exponent`
   with exponent 0.75. On the sample this is *exactly* the old behaviour (dev 0.74 reproduced bit-for-bit
   with a single model); on 80k rows wd 3.3 becomes ≈0.35, which is in the flat optimum region.
   If `weight_decay_ref_n` is absent the config is used verbatim, so old configs still work.
2. **Small ensemble.** `n_models` members are trained from the same seed stream and their softmax
   outputs averaged. `weight_decay` may be a list (one value per member) — the shipped config uses
   [2.0, 3.3, 5.7] × scaling, i.e. a spread around the optimum, which hedges the exponent guess as well as
   reducing variance. Ensembling adds +0.2–0.8 points on the synthetic corpora.
3. Optional `ema_decay` (weight EMA) — implemented, tested, **off** (no measurable gain with cosine decay).
4. `predict` now goes through `predict_proba`; output is still one `int64` class per test row.

`config.json`: `weight_decay: [2.0, 3.3, 5.7]`, `weight_decay_ref_n: 4000`, `weight_decay_exponent: 0.75`,
`n_models: 3`; everything else unchanged (hidden 256, lr 3e-3, batch 128, no aug, no label smoothing).

Things tried and rejected (no gain on gold accuracy): Gaussian input augmentation (hurts), label
smoothing (±0.3 pt, noise), hidden 512 (worse under the CPU budget), lr 1e-3 (slightly worse),
bootstrapped soft targets, batch size 32/256, LDA/QDA/kNN baselines (LDA 0.745 on dev from the sample,
but the MLP overtakes it at scale).

## Production fit

* Exact production command, single thread, 80k rows, 20 000 steps: **51–55 s CPU**, 67 MB RSS
  (limit 120 s / 2 GB). Per member ≈18 s; a 4th member would still fit but the return is small.
* Repo is 20 KB, code + JSON only; no data or fitted parameters.
* Seeds 0/1 differ by ≤0.4 points on the synthetic corpora.

## Caveats

The 80k results are on synthetic data because the corpus is not accessible. The generator matches the
sample's per-class means/covariances (after correcting for the label noise) and reproduces the real
tuning behaviour at 4k rows, but real features are unlikely to be exactly Gaussian, so expect the real
production score to be somewhat below 0.859. The "hard" synthetic corpus (classes pushed together so the
Bayes rate is 0.80) is a stress test: the shipped config still reaches 0.767 there (96% of its Bayes rate),
versus 0.61 for the old config, so the fix is robust to the exact shape of the data.
