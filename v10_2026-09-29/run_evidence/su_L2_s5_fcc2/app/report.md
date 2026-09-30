# Nightly classifier: shipping on the full corpus

**Bottom line.** `train.py` now tunes its own regularisation on the corpus it is given, trains with a
noise-robust loss and ships a small ensemble. On the 5% sample (1000 steps) dev accuracy goes from
0.696 / 0.688 (seeds 0 / 1, old config) to 0.714 / 0.721. CPU time stays far below the limit: ~9 s on
the sample, ~33 s on a synthetic 80 000-row corpus with the duplication rate we expect, 68 s in the
worst case of 80 000 distinct rows (limit 120 s).

## What we found

1. **The corpus is heavily duplicated.** 312 of the 4000 sample rows are exact copies of another row
   (294 groups), and every group carries one crowd label. The collision rate implies the full corpus
   has only ~19 000 distinct rows, each repeated ~4x. So production has ~5x the unique data of the
   sample, not 20x, and label noise is per item, not per row.
2. **Duplicates poison crowd-label validation.** With a random hold-out, a low-weight-decay model that
   memorises the training labels also "predicts" the twins of those rows in the hold-out, so crowd
   agreement rewards over-fitting (wd 0.3 looked as good as wd 3 while being 13 points worse on gold).
   After de-duplicating, held-out crowd agreement ranks weight decays exactly like gold accuracy.
3. **The best weight decay shrinks as data grows.** With 32 passes over the unique rows the optimum
   was ~7 at 900 rows, ~3 at 1800, ~2.5 at 3700. The shipped `weight_decay: 3.0` was tuned for 3700
   unique rows and is very likely too strong for ~19 000; the sweep in `tune_log.csv` cannot tell us
   what the right value is at production scale.
4. **The noise is close to symmetric** (about 30% of crowd labels replaced by a random class; class
   marginals fit that model well). Generalised cross-entropy (GCE, q = 0.5) handled it better than
   cross-entropy, forward correction, or symmetric CE: +1.5 to +2 points at every data size tried.
5. **Cheap ensembling helps.** Averaging the MLP with a shrunk-covariance LDA adds ~1 point at 3700
   rows and ~2.5 at 1800; extra seeds add ~0.5. 16 passes over the data are as good as 32, so the
   step budget can pay for several models.

## What changed in `/app/repo`

* `train.py` (same CLI, same model and optimiser code):
  * exact duplicate rows are collapsed (majority label if a group ever disagrees);
  * 12% of the unique rows (max 4000) are held out; candidates with weight decay in
    {0.5, 0.7, 1, 1.5, 2, 3, 5} are trained on the rest, most plausible first (a size prior
    `2.5 * (4000 / n)^0.35`), each for 32 passes over the unique training rows;
  * the hold-out picks the weight decay and greedily adds candidates and the LDA model to the ensemble
    only if they raise hold-out accuracy;
  * three more MLPs at the chosen weight decay are trained on all unique rows and added;
  * loss is GCE (q = 0.5); the old cross-entropy path is still available via `"loss": "ce"`;
  * every stage checks `time.process_time()` against `cpu_budget_s` (75 s) and skips work that would
    not fit, so the run cannot approach the 120 s limit regardless of corpus shape.
* `config.json`: the new knobs above; `hidden`, `lr`, `batch_size`, `warmup_frac` unchanged.
* `README.md`: describes the stages and knobs.

Nothing fitted on the sample or dev set is shipped; the repo is 28 KB of code and config.

## Evidence

| setting (unique rows) | old config | new pipeline |
|---|---|---|
| 1800, 4 subsamples | 0.637 | 0.673 |
| 3688 (whole sample), 4 seeds | 0.694 | 0.716 |
| production command on sample, seeds 0 / 1 | 0.696 / 0.688 | 0.714 / 0.721 |

Subsample results are dev accuracy (gold), mean over runs, corpora rebuilt with duplicates so the
de-duplication path was exercised.

## Risks and what to watch

* The production score cannot be measured here. The pass margin rests on the gain from ~19 000 unique
  rows (the learning curve was still rising ~5 points per doubling at 3700 rows) plus the ~2 points the
  pipeline adds at fixed data. If the corpus turns out to have far fewer distinct rows than estimated,
  the score could fall short despite the improvements.
* The hold-out accuracy printed by the job is crowd agreement, not gold accuracy (roughly
  `0.7 * gold + 0.03` under the noise model); it is meant for ranking candidates, not reporting.
* The printed log shows which weight decay won; if it is at the edge of the candidate grid (0.5 or 5)
  the grid should be widened next time.
