# Launch review: model B vs. production model A

**Recommendation: HOLD.** Under full verification, B is estimated to be *worse* than A on this pool.

| Quantity | Estimate | Approx. 95% interval |
|---|---|---|
| Q1 – B pass rate (verified) | **0.557** | 0.53 – 0.58 |
| A pass rate (verified) | 0.680 | 0.675 – 0.685 |
| Q2 – B minus A | **−0.123** | −0.15 – −0.10 |
| Q3 | **HOLD** | B is clearly below A |

For contrast, the LLM judge reports A = 0.655 and B = 0.733 (B +0.078). The judge is wrong about B.

## What went wrong with the cheap signals

B's responses are ~4× longer than A's (median 760 vs 210 tokens). Every cheap grader is fooled by
length on B, but not on A:

* **LLM judge** – on 40 random pilot items verified in full, the judge matched verification 95% of the
  time for A but only 75% for B. All 10 B errors were false passes (judge says correct, verification says
  wrong), concentrated in long responses.
* **Crowd** (2 votes each on the pilot) – worse: 67.5% agreement on B, 13 false passes out of 40.
* **Expert review** (2-minute budget) – perfect on A (40/40), but on B it accepts long responses it
  cannot check in two minutes. Across everything verified, review's *rejections* are reliable
  (only 1 of 38 review-fail B responses actually passed), but its *passes* degrade sharply with length:

  | B length (tokens) | review-pass items verified | actually correct |
  |---|---|---|
  | < 800 | 25 | 25 (100%) |
  | 800 – 1200 | 26 | 24 (92%) |
  | 1200 – 2000 | 24 | 5 (21%) |
  | > 2000 | 13 | 3 (23%) |

Nothing cheap can be used as a direct substitute for verification on B; the numbers above come from
verification, with review used only to weight strata.

## How the numbers were produced (500 credits, all spent)

1. **Pilot (176 cr):** 40 uniformly random items, verified in full for both A and B (160 cr), plus review
   and 2 crowd votes each, to measure how each cheap grader behaves. Pilot alone gave A = 0.55, B = 0.55
   (paired difference 0.00, SE 0.09 – too noisy on its own, but already no sign of B being better).
2. **Review on all 1200 responses (112 cr more).** Used as the stratification variable, not as truth.
3. **Targeted verification (212 cr):** 86 more B responses, chosen at random *within strata* defined by
   (length bin < 800 / 800–1200 / ≥ 1200) × (review verdict), with sampling concentrated in the long,
   review-pass strata where review is unreliable. 20 more A responses drawn from the 33 non-pilot items
   where the judge and reviewer disagreed (the only place A had any uncertainty).
4. **Estimator:** post-stratified (stratum-weighted) mean of verified verdicts. Each stratum's verified pass
   rate is applied to the stratum's full population count. Totals verified: A 60, B 126.

## How much to trust it

* **A = 0.680** is very solid: review agreed with verification on 40/40 random items, review-fail and
  review-pass strata verified at 0/17 and 21/21 respectively; the only mixed stratum (judge/review
  disagreements, 35 items) was verified 33/35 directly. SE ≈ 0.002.
* **B = 0.557**, SE ≈ 0.012 (finite-population-corrected, stratified). Changing the length cut-points
  moves the estimate between 0.552 and 0.557, so it is not an artefact of binning. The random pilot
  (unbiased, no modelling) independently gave 0.55. The largest remaining uncertainty is the 1200+ token
  review-pass stratum (102 items, 37 verified, 22% pass).
* **Q2 = −0.123**, SE ≈ 0.012. The direction is not in doubt: for B to match A it would need ~74 more
  correct responses than estimated, which is far outside anything the data allow. Q3 = HOLD is robust.

## Caveats

* Verification results are treated as ground truth per the board's definition; no judgement about the
  verifier itself is made.
* The pilot sample is small (40), so per-category breakdowns of B's true pass rate are not reliable; the
  judge's category numbers (B ahead everywhere except code) should not be used because of the length bias
  above.
* If B were ever re-evaluated, a cheap first-pass filter that works is: *review-fail ⇒ almost certainly
  wrong*; but *review-pass on a long response ⇒ needs full verification*.
