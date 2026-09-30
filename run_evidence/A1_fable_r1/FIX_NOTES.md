# MiniLab v2 regression: fix notes

Result on the canonical configuration (`python run.py --seeds 0-7`):

| version            | val MSE mean | seed std |
|--------------------|--------------|----------|
| v1 reference       | 0.0437       | 0.0019   |
| v2 before the fix  | 0.0492       | 0.0011   |
| v2 after the fix   | 0.0438       | 0.0018   |

Two independent deviations from the v1 (PyTorch) semantics were found and
fixed. No hyperparameter, config, data, model, initialisation, schedule or
sampling change was made.

## 1. `minilab/optim.py`: momentum dampening defaulted to `momentum`

The new optimizer set `dampening = momentum` when no dampening was given, so
the buffer was `buf = m*buf + (1-m)*g`. `torch.optim.SGD` defaults to
`dampening = 0`, i.e. `buf = m*buf + g`. With `momentum = 0.9` the v2 update
was therefore ten times smaller than v1's at every step (effective step size
`lr*(1-m)/(1-m)` instead of `lr/(1-m)`), which is a silent learning-rate
rescale and explains most of the regression.

Fix: `dampening` now defaults to `0.0`, matching PyTorch. The update rule is
otherwise unchanged (weight decay is added to the raw gradient before the
momentum buffer, exactly like `torch.optim.SGD`; a zero-initialised buffer
with dampening 0 gives `buf = g` on the first step, identical to PyTorch's
`clone(grad)` initialisation).

## 2. `minilab/trainer.py`: gradient accumulation summed micro-batch means

Each micro-batch gradient was computed as the mean over the `micro_batch_size`
examples and the `grad_accum_steps` pieces were then summed, so the optimizer
saw `grad_accum_steps` times the gradient of the global-batch mean loss. The
reference semantics average the loss over the whole global batch of
`micro_batch_size * grad_accum_steps` examples.

Fix: each micro-batch now uses `denom = micro_batch_size * grad_accum_steps`,
so the sum of the pieces is exactly the gradient of the global-batch mean
loss. Sanity check: with this fix `grad_accum_steps=4, micro_batch_size=16`,
`grad_accum_steps=1, micro_batch_size=64` and `grad_accum_steps=8,
micro_batch_size=8` produce identical results (same sampled indices, same
gradient), as they must.

Note that the two bugs partly cancelled on the canonical config (4x too-large
gradient, 10x too-small step), which is why v2 trained at all but ended up
worse than v1. Fixing only one of them makes things much worse; both are
needed.

## 3. Minor robustness changes (no effect on results)

* Gradient clipping now follows `torch.nn.utils.clip_grad_norm_` exactly:
  scale factor `min(1, clip_norm / (norm + 1e-6))` instead of
  `clip_norm / norm` applied only when `norm > clip_norm`. The difference is
  the `1e-6` epsilon and is numerically negligible; `clip_norm` is `null` in
  the canonical config.
* The divergence check now inspects all four parameter arrays, not only `W2`.

## Behaviour on other configurations

Under the correct semantics `lr = 0.12` with `momentum = 0.9` is close to the
stability limit of this model (effective step `lr/(1-m) = 1.2` against an
output-layer curvature of roughly 30), so aggressive settings such as
`lr = 0.2`, or `grad_accum_steps = 1` with the 16-example micro batch, diverge
and raise `FloatingPointError` as the trainer is designed to do. This is the
true dynamics of the reference optimizer, not a remaining discrepancy;
configurations that v1 trains stably are reproduced.
