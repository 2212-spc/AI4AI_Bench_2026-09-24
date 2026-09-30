# nightly classifier

    python3 train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

* `corpus.npz`: `X` float32 (n x 32 raw features), `y` int64 crowd labels in 0..9.
* `config.json`: hyper-parameters (read by `train.py`; `--config` overrides the path).
  `train.py` holds out `val_frac` of the corpus (crowd labels), trains one model per entry of
  `weight_decay_candidates` for `--steps` steps each (forward label-noise correction with `noise_rate`),
  and predicts with the candidate(s) that score best on the held-out slice.  `cpu_budget` (seconds of CPU)
  caps the number of candidates actually trained.
* Prototype runs (sample): `python3 train.py --data ../data/sample.npz --steps 1000 --predict <X.npy> --out <p.npy>`.
  `dev.npz` holds `X` and gold `y`; save `X` to a .npy to predict it.
