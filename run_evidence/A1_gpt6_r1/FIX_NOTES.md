# MiniLab v2 regression fix

## Changes

- **Global-batch loss normalization (`minilab/trainer.py`):** divide each
  microbatch's loss and gradients by `micro_batch_size * grad_accum_steps`.
  Summing these contributions now gives the mean gradient over all examples in
  an optimizer step. Previously, summing microbatch means multiplied the data
  gradient by `grad_accum_steps`, including before clipping and weight decay.
- **SGD momentum (`minilab/optim.py`):** default dampening to zero, as in
  `torch.optim.SGD`, instead of defaulting it to the momentum coefficient.
  Momentum now accumulates gradients using `buffer = momentum * buffer + grad`
  under the default arguments. Initialize each buffer from the first gradient
  without dampening; apply explicit dampening only to subsequent gradients.
  With zero momentum, update directly from the gradient and ignore dampening.
  Remove the now-unused NumPy import. Coupled L2 weight decay remains applied
  to every parameter, including biases, before the momentum update.
- **Global gradient clipping (`minilab/trainer.py`):** use the PyTorch clipping
  coefficient `min(1, clip_norm / (global_norm + 1e-6))`. Clipping acts once on
  the accumulated mean data gradient, before optimizer weight decay.
- **Regression coverage (`tests/test_reference_semantics.py`):** add an
  independent reference that concatenates sampled microbatches and spells out
  the forward/backward pass, schedule, clipping, and default SGD equations.
  Compare final parameters across six configurations and three seeds each
  (18 full training runs), including fresh synthetic nonlinear regression data,
  different accumulation counts, momentum, learning rate, weight decay,
  clipping, and zero warmup. Add focused checks for explicit dampening,
  first-buffer initialization, and zero-momentum updates.

The public training API, model architecture, initialization, number of steps,
sampling calls, and hyperparameter values are unchanged. No files in `configs/`
or `data/` were edited. The existing learning-rate schedule already advances
once per optimizer step and required no change. No learning-rate compensation
was introduced.

## Validation

Commands use one BLAS thread to avoid overhead on these small matrices:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py --seeds 0-7
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

- Before the fix: canonical validation MSE **0.04920**, seed std **0.00114**.
- After the fix: canonical validation MSE **0.04375**, seed std **0.00182**,
  consistent with the README's v1 result of **0.0437**, seed std **0.0019**.
- Corrected per-seed MSE (0–7): `0.04075 0.04379 0.04167 0.04333 0.04535
  0.04315 0.04624 0.04570`.
- All three regression tests pass. All 18 training comparisons have finite
  parameters and match the independent reference with `rtol=1e-9`, `atol=1e-10`.

PyTorch is unavailable in the sandbox; the reference tests implement its
documented equations in NumPy rather than executing PyTorch itself.
