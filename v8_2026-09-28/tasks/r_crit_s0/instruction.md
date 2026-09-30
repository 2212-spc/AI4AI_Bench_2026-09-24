# Re-tune the pretraining recipe for the next run

Our production pretraining recipe for this model: global batch **1024 sequences** of 4096 tokens, **16B
tokens**, AdamW with peak learning rate **lr = 0.000157** and weight decay **wd = 0.026** (decoupled, PyTorch
convention), fixed warmup, cosine decay to 10% of peak at the end of the run.  lr and wd were tuned at full
scale on exactly this setup: treat them as its optimum.

Two things change at once.  The cluster migration changes the data-parallel layout: the per-device micro-batch is fixed and there are two times as many replicas, so the next run's global batch is **2048 sequences** (was 1024).  And: the data team has cleared a larger corpus with no repetition: the next run trains on **128B tokens** (was 16B).

Everything else (model, data mixture, warmup, schedule shape, optimizer betas) stays as it is.  You choose
**lr** and **wd** for the next run.

You have **384 credits**.

* `lab run batch=<B> tokens=<billions> lr=<x> wd=<y> [seeds=1..4]` - a complete training run of this model
  with that global batch (one of [1024, 2048]) and that many training tokens (0.25 .. 128 billion); the schedule is
  laid out over the run's own length (warmup, then cosine to 10% at its last step).  Returns the final
  validation loss of each seed and their mean.  **Cost: seeds x 64 x tokens / 128 credits** - one seed of the
  next run at full length costs 64.
* Seed-to-seed standard deviation of the final validation loss: **0.0015**.  Seeds are independent; there is
  no other randomness in this task.

## Deliverable: `/app/recipe.json`

```json
{"lr": 0.0003, "wd": 0.1,
 "loss": {"lo": 2.500, "hi": 2.510}}
```

* `lr`, `wd` - graded on the **expected final validation loss of the next run** (batch 2048, 128B tokens) with
  your recipe: it must be within **0.0006** of the best expected loss any (lr, wd) achieves there.
* `loss` - an interval for the expected final validation loss of the next run **with your recipe**.  Width
  may not exceed **0.011**.
