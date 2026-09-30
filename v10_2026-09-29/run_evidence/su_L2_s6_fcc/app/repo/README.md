# nightly classifier

    python3 train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

* `corpus.npz`: `X` float32 (n x 32 raw features), `y` int64 crowd labels in 0..9.
* `config.json`: hyper-parameters (read by `train.py`; `--config` overrides the path).
* Prototype runs (sample): `python3 train.py --data ../data/sample.npz --steps 1000 --predict <X.npy> --out <p.npy>`.
  `dev.npz` holds `X` and gold `y`; save `X` to a .npy to predict it.

## Full-corpus behaviour (see /app/report.md)

`train.py` holds out `val_frac` of the corpus, trains one model per `wd_candidates` entry, ranks them
on the (noisy) hold-out and predicts with the softmax-average of those within `ensemble_tol` of the
best.  `cpu_budget_s` bounds total CPU time; later candidates are skipped if they would not fit.
