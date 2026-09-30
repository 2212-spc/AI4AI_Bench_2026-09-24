# Recipe change triage: decision and evidence

**Ship: c1, c2, c3, c5** (peak lr 0.1, momentum 0.97, grad-norm clipping 0.3, weight decay 1e-3).
**Predicted expected val MSE: 0.0398** (mean of 19 seeds, standard error 0.0003). Baseline is roughly 0.051-0.055,
so this is a ~25% reduction.

## Why the teammate's plan (ship c5, c6) is wrong

The one-seed-per-arm ablation treated every change as independent. In fact the changes interact strongly, and
the single-change effects are dominated by that:

- **c1, c2 and c4 each make the un-clipped recipe unstable.** In a 32-run half-fraction factorial (seeds 100-131,
  one seed per subset), 8 of 16 subsets *without* c3 diverged (all subsets containing c1+c2, c1+c4, or c2+c4,
  and every diverged run counts as MSE 0.3381), whereas 0 of 48 runs *with* c3 diverged. Clipping is therefore a
  prerequisite for the aggressive optimiser settings, not a "hurts" item.
- **With c3 on, c1 and c2 stop hurting and start helping, but only together with c5.** A two-way-interaction
  model fitted to the 32 runs with c3 on (a full 2^5 factorial over c1, c2, c4, c5, c6; residual sd 0.0024) gave
  strong negative (beneficial) c1×c5 and c2×c5 interactions and strong positive (harmful) c1×c4 and c2×c4
  interactions. The model's two best predicted subsets were {c1,c2,c3,c5} and {c1,c2,c3,c5,c6}, and both had the
  two best observed values in the design (0.0392 and 0.0391), far below the next group (~0.044-0.046 for
  subsets containing c4).
- **c4 (wider) is only good on its own** (with c3: ~0.046) and is actively harmful combined with the high lr /
  high momentum, so it is dropped.
- **c6 is a no-op in expectation.** It only changes which RNG stream draws the minibatch order; the fitted c6
  main effect and all its interactions were within noise (|t| < 0.7). Head-to-head over the same 15 seeds,
  {c1,c2,c3,c5} averaged 0.0398 and {c1,c2,c3,c5,c6} averaged 0.0401 (difference well within the standard
  error of ~0.0004). I ship the smaller change set.

## Run budget (80 runs)

| phase | runs | purpose |
|---|---|---|
| 1 | 32 | 2^(6-1) half-fraction (F = ABCDE), one fresh seed per subset; identified the divergence pattern and c3 as essential |
| 2 | 16 | the complementary half with c3 forced on, giving a full 2^5 factorial over {c1,c2,c4,c5,c6} with c3 on |
| 3 | 28 | 14 matched seeds each for {c1,c2,c3,c5} and {c1,c2,c3,c5,c6} |
| 4 | 4 | extra seeds for the shipped subset to tighten the estimate |

Shipped subset, all 19 seeds (300-317 plus design run 129): min 0.0380, max 0.0430, mean 0.03979, sd 0.0015, no
divergence. The pooled mean of all 34 seeds across the two statistically equivalent subsets is 0.0399, consistent
with the reported prediction.

Raw results: `/app/phase1.json`, `/app/phase2.json`, `/app/phase3.json`, `/app/phase4.log`, and `lab history`.
