# MiniLab v2 regression — fix notes

Canonical config (`configs/canonical.json`, seeds 0-7), before and after:

| version            | val MSE mean | seed std |
|--------------------|--------------|----------|
| v1 reference       | 0.0437       | 0.0019   |
| v2 before fix      | 0.04920      | 0.00114  |
| v2 after fix       | 0.04375      | 0.00182  |

Two independent bugs each scaled the effective update; together they made v2
train with 0.4x the intended step size (0.1x from the optimizer, 4x from the
loss normalisation). Both are fixed in the training code only; configs, data,
model, initialisation, schedule, step count and data sampling are untouched.

## 1. `minilab/optim.py`: SGD dampening defaulted to `momentum` (should be 0)

`SGD.__init__` set `dampening = momentum` when no dampening was given, so the
momentum buffer was updated as `buf = m*buf + (1-m)*g`. `torch.optim.SGD`
defaults to `dampening=0`, i.e. `buf = m*buf + g`. With `momentum=0.9` the v2
buffer was 10x smaller than the reference, and the discrepancy varies with
`momentum` (it vanishes at `momentum=0` and grows towards 1). The default is
now `dampening=0.0`, matching PyTorch. Weight decay is still added to the
gradient before the momentum update (`g + wd*p`), which is what PyTorch does.

## 2. `minilab/trainer.py`: micro-batch gradients were normalised per micro-batch

Each of the `grad_accum_steps` micro-batches computed a gradient of the loss
averaged over its own `micro_batch_size` examples, and the trainer summed them.
The sum is therefore `grad_accum_steps` times the gradient of the loss averaged
over the global batch, which is the reference definition ("averaged over all
examples that contribute to one optimizer step"). The trainer now passes
`denom = micro_batch_size * grad_accum_steps` to `loss_and_grads`, so summing
the micro-batch contributions yields exactly the global-batch mean gradient.
Check: `grad_accum_steps=4, micro_batch_size=16` now gives results identical to
`grad_accum_steps=1, micro_batch_size=64` (same seeds), and the gradient itself
was verified against finite differences (max error ~1e-10).

## 3. `minilab/trainer.py`: gradient clipping matched to `clip_grad_norm_`

`clip_norm` is null in the canonical config, so this did not affect the
canonical numbers, but held-out configs may enable it. The scaling now follows
`torch.nn.utils.clip_grad_norm_`: `coef = min(1, clip_norm / (norm + 1e-6))`,
applied to the accumulated global-batch gradient before the optimizer step.
Clipping was previously applied to the 4x-too-large summed gradient (bug 2),
which also changed *when* clipping kicked in.

## 4. `minilab/trainer.py`: divergence check covers all parameters

The non-finite check only looked at `W2`; it now checks every parameter so a
diverged run always raises `FloatingPointError` rather than returning NaNs from
another tensor.

## Not changed / notes

* `minilab/schedule.py` already implements linear warmup over `warmup_steps`
  optimizer steps then cosine decay to `min_lr_ratio * lr`, advanced once per
  optimizer step; left as is.
* No hyperparameters were rescaled. The fix restores the reference update
  rule, which is why the canonical mean and seed spread now coincide with v1.
* With the corrected (reference) semantics, configs with a very large
  effective step, e.g. `grad_accum_steps=1` at `lr=0.12, momentum=0.9`
  (batch 16, effective step 1.2), are unstable at initialisation; this is a
  property of the reference update rule itself, not of the implementation.
