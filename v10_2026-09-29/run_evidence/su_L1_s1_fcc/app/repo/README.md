# nightly classifier

    python3 train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

* `corpus.npz`: `X` float32 (n x 32 raw features), `y` int64 crowd labels in 0..9.
* `config.json`: hyper-parameters (read by `train.py`; `--config` overrides the path).
* Prototype runs (sample): `python3 train.py --data ../data/sample.npz --steps 1000 --predict <X.npy> --out <p.npy>`.
  `dev.npz` holds `X` and gold `y`; save `X` to a .npy to predict it.
