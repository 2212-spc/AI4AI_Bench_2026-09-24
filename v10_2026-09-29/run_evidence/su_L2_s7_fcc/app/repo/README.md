# nightly classifier

    python3 train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

* `corpus.npz`: `X` float32 (n x 32 raw features), `y` int64 crowd labels in 0..9.
* `config.json`: hyper-parameters (read by `train.py`; `--config` overrides the path).
  - base MLP settings (`hidden`, `lr`, `batch_size`, `weight_decay`, `warmup_frac`, ...);
  - `noise_rate`: assumed symmetric crowd-label noise rate used by the loss (0 = plain cross-entropy);
  - `val_frac`: fraction of the corpus held out (crowd labels) to compare candidates; selection is skipped
    when that split would have fewer than `min_val_rows` rows;
  - `candidates`: list of hyper-parameter overrides tried in order; `cpu_budget_s` stops the list early if
    the next candidate would not fit in the CPU budget; `ensemble_tol` (in standard errors of the validation
    accuracy) decides which candidates are averaged for the final prediction.
* Prototype runs (sample): `python3 train.py --data ../data/sample.npz --steps 1000 --predict <X.npy> --out <p.npy>`.
  `dev.npz` holds `X` and gold `y`; save `X` to a .npy to predict it.
