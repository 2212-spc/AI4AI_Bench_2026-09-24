# Lab notes (pretraining team)

- LR sweeps at four sizes along the Chinchilla ratio (D = 20 N), batch 0.5M, warmup 2%, qk-layernorm off, seed 0.
  No run diverged, so stability is not a concern at these learning rates.
- Per-size optimum from a parabola in ln(lr) over the runs that finished, then a power law across sizes:
  **eta*(N) = 0.01557 * N^-0.114** (along D = 20 N).
- qk-layernorm on vs off at 8e7 params, at that size's optimum, two seeds each: no difference in loss beyond seed
  noise.  It costs ~3% step time, so production drops it.
- Production plan: N = 7e9, D = 1.4e11 tokens (20 tokens/param), batch 0.5M, warmup 2%, qk-layernorm off,
  peak LR from the rule above: **lr_prod = 0.00117**.
