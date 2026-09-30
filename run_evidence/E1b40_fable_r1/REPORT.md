# Recipe change triage: ship c1 + c2 + c3 + c5

**Decision:** ship c1 (peak lr 0.06), c2 (momentum 0.95), c3 (grad-norm clipping 1.0), c5 (weight decay 3e-4).
Drop c4 (hidden 160) and c6 (separate data-order RNG).

**Predicted expected val MSE:** 0.0422 (17-seed mean 0.04216, seed SD 0.00094, SEM 0.00023, no divergences).

## Evidence (40 runs, all on `lab`; full log in `lab history`)

1. **The teammate's plan (c1, c2, c4, c6) is unsafe.** Run 2 at seed 101 diverged (counted as 0.3466).
   Even c1+c2 alone without clipping gave 0.0674 at seed 101, far worse than baseline (0.0519). The higher
   lr and momentum together make training unstable; the single-change ablation could not reveal this
   interaction.
2. **c3 (clipping) is the enabler, not a loss.** On its own c3 looks neutral/slightly negative (teammate: +2.7%,
   consistent with c1+c3 and c2+c3 at seed 101 being roughly baseline-level). But c1+c2+c3 gives 0.0435 on
   average over 4 seeds (0.0439, 0.0434, 0.0429, 0.0437), the best 3-change combination found, because it
   removes the divergence/instability that c1+c2 introduce.
3. **c5 (weight decay) helps consistently on top of c1+c2+c3.** Paired over seeds 101/202/303/404,
   c1+c2+c3+c5 beat c1+c2+c3 on every seed (0.0419 vs 0.0439, 0.0432 vs 0.0434, 0.0415 vs 0.0429,
   0.0419 vs 0.0437), mean improvement about 3%.
4. **c4 (wider layer) hurts when combined with the high-lr recipe.** c1+c2+c3+c4+c5 was 0.0462 and 0.0485
   at seeds 101/202 versus 0.0419/0.0432 for c1+c2+c3+c5; c1+c2+c3+c4 (0.0494), c1+c2+c3+c4+c6 (0.0528) and
   the full six-change set (0.0471) were all clearly worse at seed 101. c4 alone helps a little at the
   baseline lr, but not with the shipped optimiser settings.
5. **c6 is a no-op in expectation and was not established as a win.** It only re-partitions randomness
   between init and batch order, so the seed-averaged MSE should be essentially unchanged. Paired runs
   confirm noise-level differences: on top of c1+c2+c3 (seed 101) -0.0004; on top of c1+c2+c3+c5 the
   differences were -0.0032 (seed 202) and +0.0005 (seed 303). I left it out to keep the shipped recipe
   one that I have measured on 17 seeds rather than 2; if c6 did have a true effect, the paired data bound
   it to within the ~2-3% band, well inside the tolerance.

## Budget use

- 1 baseline, 7 exploratory combinations at seed 101 (found the divergence and the c1+c2 problem)
- 8 clipped combinations at seed 101 (identified c1+c2+c3+c5 as the front-runner)
- 8 runs: paired comparison of c1+c2+c3+c5 vs c1+c2+c3 on 3 more seeds, plus c1+c3+c5 and c2+c3+c5
- 3 runs: c6 and c4 checks on top of the front-runner
- 13 runs: additional seeds for the shipped recipe to estimate its expected MSE (17 seeds total)

Seed-to-seed SD of the shipped recipe is ~2.2% of the mean, so the 17-seed mean has a standard error of
about 0.5%, comfortably within the required 6% prediction tolerance.
