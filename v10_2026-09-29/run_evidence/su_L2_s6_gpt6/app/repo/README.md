# Nightly crowd-label classifier

```sh
python3 train.py --data corpus.npz --steps 20000 --predict test_X.npy --out preds.npy --seed 0
```

Only Python and numpy are required. Inputs are training `X` and `y` arrays in an NPZ,
and a two-dimensional prediction feature matrix in an NPY file. Output is an int64
class vector in an NPY file. The `--config` and `--log_every` options remain supported.

The model retains two 256-unit ReLU hidden layers, minibatch AdamW, training-only
standardization, and warmup/cosine learning rates. Three independent initializations
average class probabilities. Each receives the requested training steps, subject
to an aggregate 100-second process-CPU guard with time reserved for prediction.
BLAS is restricted to one thread.

`weight_decay` is the reference value at `reference_rows`; each ensemble member
uses `weight_decay * min(1, reference_rows/n)**exponent`. At 80,000 rows the default
values are approximately 0.3172, 0.1500, and 0.6708. These bracket plausible
regularization transfer rates; they are not estimates of label corruption.
Learning-rate, batch-size and class definitions remain unchanged. No gold labels
are loaded by the training job, no classes are suppressed, and no trained parameters
or data are included in this repository.

This configuration has not been evaluated on the inaccessible production corpus
or test set. Sample accuracy is not a production accuracy guarantee.
