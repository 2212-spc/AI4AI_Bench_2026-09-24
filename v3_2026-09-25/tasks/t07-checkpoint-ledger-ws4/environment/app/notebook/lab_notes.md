# Lab notes (scaling team)

- Runs 1-4: one cosine run per model size (N = 2e7, 5e7, 1.2e8, 3e8), each on D = 14 N tokens, seed 0, with
  checkpoints evaluated at 10, 20, 30, 50 and 70% of training (notebook/checkpoints.json).  Every checkpoint
  is a (N, tokens-seen, loss) point, so four runs give us 24 points for the scaling law at the price of four.
- Fit of L(N, D) = E + A/N^alpha + B/D^beta to all 24 points: E = 1.739, alpha = 0.465,
  beta = 0.298.  At the production compute 6*N*D = 1.08e+21 FLOPs this puts the compute-optimal
  size at N = 6.8e+08; for the planned production run (N = 3e9, D = 6e10) it predicts a final loss of
  2.673.
- Run 5: WSD pilot at N = 1.2e8 (same D, same checkpoints).  It was worse than the cosine run at 5 of the
  5 checkpoints; the final loss differed by -0.0257, within what we would expect from seeds.  WSD
  buys us nothing; production stays on cosine.
- Production plan: N = 3e9, D = 6e10, cosine.  The reservation may be cut short (see the manual).
