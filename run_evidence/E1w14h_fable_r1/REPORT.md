# Recipe change triage: decision and evidence

**Ship: c3 (grad-norm clipping 0.3) + c4 (hidden 256) + c5 (weight decay 1e-3).**
**Predicted expected val MSE: 0.0464** (mean of 8 seeds, sd 0.0019, standard error ~0.0007).

## Method (40 runs)

1. Baseline on seeds 0 and 1 to gauge seed noise (0.0510, 0.0526) and to confirm runs are deterministic per seed
   (seed 0 reproduced the teammate's 0.0510 exactly).
2. Screened 17 combinations on the common seed 0, deliberately targeting interactions rather than single changes
   (the teammate's one-at-a-time table is misleading here, see below).
3. Compared the top arms on common seeds 1-2, then spent the remaining budget on extra seeds for the three leaders
   so the winner is chosen on paired, common-seed differences rather than on one lucky seed.

## Key findings

| arm | n seeds | mean val MSE | sd |
|---|---|---|---|
| c3,c4,c5 | 8 | 0.0464 | 0.0019 |
| c3,c4 | 7 | 0.0469 | 0.0021 |
| c1,c3,c4,c5 | 6 | 0.0470 | 0.0023 |
| c1,c3,c4 | 3 | 0.0496 | 0.0036 |
| baseline | 2 | 0.0518 | 0.0011 |

Paired (same-seed) comparisons:
- c3,c4,c5 minus c3,c4: -0.0008 on average, better on all 7 common seeds (sd of the difference only 0.0003), so c5 is
  a small but consistent win on top of c3+c4.
- c3,c4,c5 minus c1,c3,c4,c5: -0.0010 (se 0.0005). Raising the lr (c1) does not help once clipping and width are in,
  and it has real downside risk elsewhere (see below).
- c3,c4,c5 minus baseline: about -0.005, i.e. roughly 10% lower MSE than production.

## Interactions that drove the decision

- **c4 needs c3.** Alone, widening hurts (teammate: 0.0664) and c4+c5+c6 without clipping gave 0.0759. With clipping,
  c3+c4 is the best pair found (0.0455 on seed 0). The wider layer makes early steps unstable; clipping tames it.
- **c1 + c4 is dangerous.** c1+c4 without clipping diverged outright (counted as 0.3381). With clipping it recovers
  (c1+c3+c4 = 0.0467 seed 0) but is noisier across seeds (0.0536 on seed 2) and no better in expectation.
- **c2 (momentum 0.97) hurts** in every combination tried (c2+c3 0.0537, c1+c2+c3+c4 0.0609, all-six 0.0503), so it was dropped.
- **c6 (separate data RNG) is neutral-to-slightly-negative** on top of the winner (c3+c4+c6 0.0503, c3+c4+c5+c6 0.0485
  vs 0.0448 for c3+c4+c5 on the same seed). Since it only re-shuffles minibatch order, it cannot change the
  expectation over seeds in a systematic way, but the single-seed evidence gave no reason to include it. Excluded.
- **The teammate's plan (c5+c6)** gives 0.0496 on seed 0, a marginal gain over baseline and clearly worse than c3+c4+c5
  (0.0448 on the same seed). Greedy selection from single-change ablations misses the c3/c4 synergy entirely.

## Prediction

Predicted expected val MSE for the shipped subset is 0.0464, the mean over seeds 0-7. No divergence was observed in
any of the 8 runs of this arm; clipping keeps the wide model stable, so the divergence penalty is not expected to
contribute materially to the expectation.
