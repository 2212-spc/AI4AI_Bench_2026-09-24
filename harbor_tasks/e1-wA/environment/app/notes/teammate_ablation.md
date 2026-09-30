# Ablation notes (teammate, last week)

I tried each proposed change on its own against the baseline (one run each, seed 9).

| run | val MSE | vs baseline |
|---|---|---|
| baseline | 0.0502 |  |
| c1 | 0.0459 | -8.6% |
| c2 | 0.0436 | -13.1% |
| c3 | 0.0515 | +2.7% |
| c4 | 0.0458 | -8.7% |
| c5 | 0.0499 | -0.5% |
| c6 | 0.0495 | -1.4% |

Plan: ship the winners together (c1, c2, c4, c6); drop c3 (hurts) and c5 (no effect). Didn't have time to run the combination.
