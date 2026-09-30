# Fixed training recipe

Everything below is fixed for both the proxies and the target run; only `n` (width), `steps` and `eta` vary.

* decoder-only Transformer, depth scales with width in the usual way, so compute per step ~ n^2
* AdamW, beta = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0
* linear warmup then cosine decay to 10% of peak; warmup is 2% of steps in every run
* the data is a fixed, effectively unlimited pretraining mixture - no repetition at any of these scales
* `final_loss` is the held-out loss at the end of the run; it carries run-to-run noise of roughly 0.004
* a run that diverges returns no loss at all, only the step where the spike happened

Internal notes from the last two allocations:

* "we tuned the learning rate at width 512 and reused it at 4096; that run was lost on day 3" (postmortem,
  no follow-up measurement was made);
* "the largest model we ever tuned directly was width 1024";
* "somebody claimed the stable learning rate goes like 1/width, but nobody measured the exponent here."
