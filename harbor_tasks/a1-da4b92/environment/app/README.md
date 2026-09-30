# MiniLab v2 — validation regression

MiniLab trains a small two-layer ReLU MLP regressor on a fixed synthetic
regression dataset (`data/`). The v1 code base was a direct PyTorch
implementation; v2 is a NumPy rewrite that added gradient accumulation,
optional gradient clipping and a new optimizer module.

**Problem.** On the frozen canonical configuration (`configs/canonical.json`)
v1 reached a validation MSE of **0.0437** (mean over seeds 0-7, seed std
0.0019). v2 currently reaches about **0.0492**. Nothing about the
data or the configuration changed between v1 and v2.

**Reference semantics (v1).** v1 used standard PyTorch components with default
arguments except where stated:

* loss: 0.5 * (prediction - target)^2, averaged over all examples that
  contribute to one optimizer step (i.e. the global batch of
  `micro_batch_size * grad_accum_steps` examples);
* optimizer: `torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)`
  applied to all parameters;
* schedule: linear warmup for `warmup_steps` optimizer steps, then cosine decay
  to `min_lr_ratio * lr`, advanced once per optimizer step;
* gradient clipping on the global norm when `clip_norm` is not null;
* initialisation as in `minilab/model.py` (unchanged from v1).

Run: `python run.py --seeds 0-7`
