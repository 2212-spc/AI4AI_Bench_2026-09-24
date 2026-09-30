# nightly classifier

    python3 train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

* `corpus.npz`: `X` float32 (n x 32 raw features), `y` int64 crowd labels in 0..9.
* `config.json`: hyper-parameters (read by `train.py`; `--config` overrides the path).
  Top-level keys are defaults; `members` lists the ensemble members, each overriding the defaults
  (`hidden: 0` = multinomial logistic regression, `quad_features: true` = add pairwise products of the
  standardised features).  Every member is trained for `--steps` steps; log-probabilities are averaged.
* The shipped config targets the full 80k-row corpus (20000 steps).  Weight decay 0.3 is far too weak for
  the 4000-row sample - use `--config` with `weight_decay` ~3 for prototype runs on `sample.npz`.
* Prototype runs (sample): `python3 train.py --data ../data/sample.npz --steps 1000 --predict <X.npy> --out <p.npy>`.
  `dev.npz` holds `X` and gold `y`; save `X` to a .npy to predict it.
