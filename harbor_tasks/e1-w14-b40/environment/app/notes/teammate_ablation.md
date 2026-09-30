# Ablation notes (teammate, last week)

I tried each proposed change on its own against the baseline (one run each, seed 0).

| run | val MSE | vs baseline |
|---|---|---|
| baseline | 0.0510 |  |
| c1 | 0.0592 | +16.0% |
| c2 | 0.0789 | +54.6% |
| c3 | 0.0574 | +12.5% |
| c4 | 0.0664 | +30.1% |
| c5 | 0.0500 | -2.0% |
| c6 | 0.0507 | -0.8% |

Plan: ship the winners together (c5, c6); drop c1, c2, c3, c4 (hurts). Didn't have time to run the combination.
