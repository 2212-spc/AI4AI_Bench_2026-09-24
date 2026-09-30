# MiniLab v2 regression fix

## Changes

- **Global-batch loss normalization (`minilab/trainer.py`).** Each microbatch
  now divides its loss and gradients by `micro_batch_size * grad_accum_steps`.
  Summing these contributions produces the mean gradient over all examples in
  one optimizer step. Previously, summing microbatch means multiplied the
  gradient by `grad_accum_steps`.
- **PyTorch SGD momentum semantics (`minilab/optim.py`).** Default dampening is
  now zero, rather than the momentum coefficient. The momentum buffer is
  initialized with a copy of the first gradient (including L2 weight decay);
  later updates use `momentum * buffer + (1 - dampening) * gradient`.
  When momentum is zero, SGD applies the gradient directly, independent of
  dampening. Explicit `dampening=None` remains accepted and means zero.
  Weight decay still applies to every parameter before the momentum update.
- **PyTorch global-norm clipping (`minilab/trainer.py`).** The accumulated,
  globally averaged gradients are scaled by
  `min(1, clip_norm / (global_norm + 1e-6))`, matching PyTorch's default clipping
  denominator. Clipping remains before optimizer weight decay and momentum.
- **Regression coverage (`tests/test_training.py`).** Added full-batch reference
  comparisons and explicit numeric checks for initial momentum buffers,
  nonzero dampening, and zero momentum. The full-batch reference uses one
  concatenated sampled batch per step and an independent optimizer recurrence.

No configuration files, data, hyperparameter values, architecture,
initialization, step counts, or sampling were changed. The existing warmup and
cosine schedule already advances once per optimizer step and was preserved.
The public `train(cfg, x_train, y_train, seed)` return value remains the
`W1`, `b1`, `W2`, `b2` parameter dictionary.

## Validation

Commands used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` to limit thread overhead.

- `python run.py --seeds 0-7`: before the fix, validation MSE mean **0.04920**,
  population standard deviation **0.00114**. After the fix, mean **0.04375**,
  population standard deviation **0.00182**, consistent with the README's v1
  reference (**0.0437**, standard deviation **0.0019**). All eight runs finished
  with finite results. Corrected per-seed MSEs:
  `0.04075 0.04379 0.04167 0.04333 0.04535 0.04315 0.04624 0.04570`.
- `python -m unittest discover -s tests -v`: all three tests passed. Full-batch
  comparisons cover six configurations and three seeds (18 runs of 500 steps)
  on freshly generated synthetic regression data, varying accumulation,
  momentum, learning rate, weight decay, clipping, and warmup. Every returned
  parameter matches the reference with `rtol=1e-10, atol=1e-11`, and all are
  finite. The additional synthetic test data exercises semantic equivalence;
  it does not claim to reproduce the frozen dataset's unknown generator.

PyTorch is not installed in the sandbox, so the automated reference checks
encode its documented SGD and clipping semantics in NumPy rather than invoking
PyTorch directly.
