# Nightly classifier

```sh
python3 train.py --data corpus.npz --steps 20000 --predict test_X.npy --out preds.npy --seed 0
```

Requires Python 3 and numpy. Training input contains `X` (float32 feature matrix)
and `y` (integer crowd labels). Prediction input is a numpy feature matrix with
the same columns. Output is a one-dimensional int64 numpy array of class IDs.

`config.json` controls training; `--config` overrides its location. Each ensemble
member trains for `--steps` steps on the supplied data, using a reproducible seed
derived from `--seed`. Predictions average clean-label probabilities from four
MLPs. All normalization statistics and model parameters are learned at runtime.
BLAS is restricted to one thread before numpy is imported.

The loss models crowd labels as a mixture of clean-label probabilities and a
uniform distribution, with mixture weight `noise_rate`. Setting that value to
zero recovers ordinary cross entropy. This is a noise-model assumption, not a
claim that each individual crowd label has a known error probability.

Weight decay is specified at `decay_reference_rows` training rows and scaled as
`weight_decay * (decay_reference_rows / n) ** decay_power`. At 80,000 rows, the
four members use widths/decays of 256/0.3354, 256/0.1500, 128/0.1057, and
128/0.3354. Different decay powers hedge uncertainty when extrapolating from the
small development sample. Each member uses AdamW, batch 128, learning rate
0.003, 2% warmup, and cosine decay.

For a single model, omit `members` or use `[{}]`. Original configuration files
without noise or scaling settings retain their original loss and fixed decay.
