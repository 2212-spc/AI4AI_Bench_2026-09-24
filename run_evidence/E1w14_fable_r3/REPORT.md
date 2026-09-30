# Recipe change triage: ship c3 + c4 + c5

**Decision:** ship gradient clipping (c3), the 256-unit hidden layer (c4) and L2 weight decay 0.001 (c5).
Do not ship the higher peak LR (c1), higher momentum (c2) or the decoupled data-order RNG (c6).

**Predicted expected val MSE:** 0.0466 (baseline is about 0.0517, so roughly a 10% improvement).

## Why the teammate's plan (c5 + c6 only) is wrong

The teammate's single-seed, one-at-a-time ablation misses a strong interaction: widening the network (c4)
hurts on its own (0.0664, seed 0) because the wider net becomes unstable at the baseline learning rate,
but with gradient clipping (c3) the same width is the biggest win in the whole study (c3+c4 = 0.0455 on
seed 0 versus 0.0510 baseline). Clipping alone also looked harmful in the ablation, yet it is exactly the
change that unlocks the width. Seed-0-only numbers are also within noise for c5 and c6: the seed-to-seed
standard deviation of a single configuration is about 0.002, comparable to the 2% "wins" they reported.

## Evidence (40 runs total, all listed in `lab history`)

Mean validation MSE per configuration, with number of seeds:

| config | seeds | mean val MSE | sd |
|---|---|---|---|
| c3,c4,c5,c6 | 3 (seeds 1-3) | 0.04628 | 0.0007 |
| **c3,c4,c5** | **13 (seeds 0-12)** | **0.04661** | 0.0018 |
| c3,c4 | 8 (seeds 0-7) | 0.04724 | 0.0022 |
| c1,c3,c4,c5 | 3 | 0.04798 | 0.0029 |
| c1,c3,c4,c6 | 1 | 0.04830 | |
| c1,c3,c4 | 3 | 0.04960 | 0.0036 |
| c2,c3,c4 | 1 | 0.04994 | |
| c3,c4,c6 | 1 | 0.05034 | |
| baseline | 3 | 0.05168 | 0.0008 |
| c1,c2,c3 | 1 | 0.05250 | |
| c1,c3 | 1 | 0.05339 | |
| c1,c2,c3,c4 | 1 | 0.06088 | |
| c4,c5 | 1 | 0.06768 | |

Key comparisons:

- **c5 helps on top of c3+c4.** On the 8 matched seeds (0-7) c3+c4+c5 beat c3+c4 on every single seed
  (paired mean 0.0464 vs 0.0472). That is a consistent, if small, gain, so c5 is shipped.
- **c1 (peak LR 0.1) hurts** with clipping and width: c1+c3+c4 averaged 0.0496 over 3 seeds and
  c1+c3+c4+c5 averaged 0.0480 over 3 seeds, both worse than the same recipes without c1, and with much
  larger seed variance.
- **c2 (momentum 0.97) hurts** everywhere it was tried (c2+c3+c4 = 0.0499, c1+c2+c3+c4 = 0.0609,
  teammate's c2-alone = 0.0789).
- **c6 is neutral in expectation.** It only changes which RNG stream draws the minibatch order, so the
  distribution of (init, order) pairs over seeds is unchanged. Empirically, on the matched seeds 1-3,
  c3+c4+c5+c6 averaged 0.04628 versus 0.04610 for c3+c4+c5, a difference well inside noise. Since it has
  no expected benefit and adds a code change, it is not shipped. (Shipping it would be equally fine.)
- **c4 without c3 is bad** (c4 alone 0.0664, c4+c5 0.0677): the width must go in together with clipping.

## Prediction

The shipped recipe was run on 13 seeds with mean 0.04661 and standard error 0.0005. Pooling the three
c6 runs (same expected value) gives 0.0465. The prediction is therefore 0.0466, with a roughly
plus or minus 2% uncertainty, comfortably inside the required 6% band.
