# nightly classifier

    python3 train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

* `corpus.npz`: `X` float32 (n x 32 raw features), `y` int64 crowd labels in 0..9.
* `config.json`: hyper-parameters (read by `train.py`; `--config` overrides the path).
* Prototype runs (sample): `python3 train.py --data ../data/sample.npz --steps 1000 --predict <X.npy> --out <p.npy>`.
  `dev.npz` holds `X` and gold `y`; save `X` to a .npy to predict it.

## How a run works (details in the module docstring of `train.py`)

1. exact duplicate rows are collapsed (duplicates share their crowd label);
2. `val_frac` of the unique rows (at most `val_max`) is held out as a crowd-labelled validation split;
3. one MLP per entry of `wd_candidates` is trained with the generalised cross-entropy loss (`gce_q`)
   for `epochs_per_candidate` passes over the unique training rows, most plausible weight decay first
   (`prior_wd_at_4000`, `prior_wd_exponent`);
4. the validation split selects the weight decay and greedily builds an ensemble (candidates within
   `ensemble_tol` of the best, plus an LDA model with covariance shrinkage `lda_shrink`);
5. `final_models` extra MLPs at the selected weight decay are trained on all unique rows and added.

Every stage checks `time.process_time()` against `cpu_budget_s` and skips work that would not fit,
so the job stays well inside the 120 s production limit whatever the corpus looks like.
