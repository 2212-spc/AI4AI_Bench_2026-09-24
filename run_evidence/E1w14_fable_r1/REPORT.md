# Recipe change triage: decision and evidence

**Decision: ship c3 + c4 + c5** (gradient clipping at max-norm 0.3, hidden width 256, L2 weight decay 0.001).
**Predicted expected val MSE: 0.0466** (mean of 11 seeds; SEM 0.0006).
Baseline is 0.0517 (3 seeds), so the shipped recipe improves expected val MSE by about 10%.

## Why the teammate's plan (c5 + c6 only) is wrong

The teammate's ablation tested each change alone at a single seed. Two things break that analysis:

1. **Seed noise is about the size of the effects.** Baseline over 3 seeds spans 0.0510 to 0.0526; c3+c4 spans 0.0455 to 0.0505.
   Single-seed differences of 1-2% (c5 at -2%, c6 at -0.8%) are not evidence of anything.
2. **The changes interact strongly.** c4 (wider layer) alone hurts (0.0664) because the wider net is unstable at
   lr 0.04 without clipping, and c4 with c1 (lr 0.1) and no clipping diverges outright. But c3 (clipping) fixes
   that instability, and c3+c4 together is clearly the best pair (mean 0.0472 over 3 seeds vs baseline 0.0517).
   Neither c3 nor c4 looks good on its own, which is exactly why one-at-a-time ablation missed the winner.

## Evidence for each change (all 40 runs; diverged runs counted as 0.3381)

| subset | n seeds | mean val MSE | std | values |
|---|---|---|---|---|
| c3,c4,c5 | 11 | 0.0466 | 0.0019 | 0.0448, 0.0492, 0.0448, 0.0443, 0.0478, 0.0451, 0.0467, 0.0487, 0.0458, 0.0492, 0.0458 |
| c1,c3,c4 | 1 | 0.0467 |  | 0.0467 |
| c3,c4,c5,c6 | 7 | 0.0469 | 0.0019 | 0.0485, 0.0469, 0.0464, 0.0455, 0.0474, 0.0439, 0.0495 |
| c3,c4 | 3 | 0.0472 | 0.0029 | 0.0455, 0.0505, 0.0455 |
| c1,c3,c4,c5,c6 | 1 | 0.0478 |  | 0.0478 |
| c1,c3,c4,c5 | 3 | 0.0480 | 0.0029 | 0.0449, 0.0506, 0.0484 |
| c2,c3,c4,c5 | 1 | 0.0492 |  | 0.0492 |
| c2,c3,c4 | 1 | 0.0499 |  | 0.0499 |
| c3,c4,c6 | 1 | 0.0503 |  | 0.0503 |
| baseline | 3 | 0.0517 | 0.0008 | 0.0510, 0.0526, 0.0514 |
| c1,c3 | 1 | 0.0534 |  | 0.0534 |
| c2,c3 | 1 | 0.0537 |  | 0.0537 |
| c3,c5 | 1 | 0.0560 |  | 0.0560 |
| c1 | 1 | 0.0592 |  | 0.0592 |
| c1,c2,c3,c4 | 1 | 0.0609 |  | 0.0609 |
| c4 | 1 | 0.0664 |  | 0.0664 |
| c4,c5 | 1 | 0.0677 |  | 0.0677 |
| c1,c4,c5 | 1 | 0.3381 |  | 0.3381 |

Per-change conclusions:

- **c3 (clipping): ship.** Required to make c4 usable. c3 alone (0.0574) and c3+c5 (0.0560) are worse than baseline,
  so it only pays off together with c4.
- **c4 (width 256): ship.** Without clipping it is bad (0.0664, 0.0677) or diverges (with c1). With clipping it is the
  single largest improvement seen.
- **c5 (weight decay): ship.** On top of c3+c4 it helps: seed-paired 0.0455/0.0505/0.0455 (c3+c4) vs 0.0448/0.0492/0.0448
  (c3+c4+c5), better on every seed, and the 11-seed mean of c3+c4+c5 is 0.0466.
- **c1 (lr 0.1): drop.** Worse in every seed-paired comparison against c3+c4+c5 (0.0449/0.0506/0.0484 vs
  0.0448/0.0492/0.0448), diverges with c4 when clipping is off, and hurts alone (0.0592).
- **c2 (momentum 0.97): drop.** Hurts alone (0.0789), on c3 (0.0537 vs 0.0510 baseline), on c3+c4 (0.0499 vs 0.0455)
  and on c3+c4+c5 (0.0492 vs 0.0448), all at seed 0.
- **c6 (separate data-order RNG): drop, but it is a coin flip.** c6 only changes how the seed maps to minibatch order,
  so it should not change the expected MSE over seeds. Measured: c3+c4+c5 mean 0.0466 over 11 seeds vs
  c3+c4+c5+c6 mean 0.0469 over 7 seeds; the difference is well inside one standard error. Not shipping it keeps the
  recipe simpler. If the truth is that c6 is neutral, both subsets are equivalent for the shipping criterion.

## Prediction

predicted_val_mse = 0.0466, the plain mean over the 11 seeds (0 to 10) run with exactly c3+c4+c5.
Pooling with the 7 c6 seeds (which should be exchangeable) gives 0.0467, essentially the same number.

## Budget

40 of 40 runs used: 3 baseline seeds, 11 seeds of the shipped subset, 7 seeds of shipped+c6, 3 seeds each of c3+c4 and
c1+c3+c4+c5, and single seed-0 probes of the other combinations listed above.
