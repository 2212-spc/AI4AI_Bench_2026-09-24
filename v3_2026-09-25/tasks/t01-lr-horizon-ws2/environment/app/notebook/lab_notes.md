# Lab notes (pretraining team)

- All five LR sweeps below were run at the Chinchilla ratio (D = 20 N), batch 0.5M tokens, seed 0.
- Per-size optimum from a parabola fit in ln(lr), then a power law across sizes:
  **eta*(N) = 3.877 * N^-0.381**.  Our muP port still has the known width bug, so the optimum drifts
  with width; once that is accounted for, muP makes the optimum independent of training length.
- Production plan: N = 1e9, D = 1e12 tokens (1000 tokens/param), batch 4M tokens.
  Planned peak LR from the rule above: **lr_prod = 0.001453**.
- Loss is fairly flat near the optimum, so a factor-of-two error in LR is cheap (see sweeps).
