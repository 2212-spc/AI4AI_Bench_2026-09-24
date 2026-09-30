# Launch review: model B vs model A

## Bottom line

| Quantity | Estimate | 95% interval | Basis |
|---|---|---|---|
| Q1: B pass rate (full verification) | **0.757** | 0.72 to 0.79 | stratified estimate from 112 verified B responses |
| Q2: B minus A | **+0.021** | -0.03 to +0.06 | A estimated at 0.737 from 72 verified A responses |
| Q3 | **SHIP** (weak) | P(B better) is about 0.79 | bootstrap over verification labels |

B is probably slightly better than A on the board's definition of correct, but the margin is small and
not statistically decisive. The headline numbers from the cheap signals (judge and expert review) are
badly wrong for B and should not be used: they say B passes at 0.84 to 0.85 and beats A by 0.12 to 0.14.

## Why the cheap signals mislead

B's responses are 3.4 times longer than A's on average (715 vs 211 tokens). On a random pilot of 40 items
verified for both models, the LLM judge and the two-minute expert review both accepted long B responses
that full verification rejected. Every one of the judge's false passes on B was a response longer than
600 tokens, and the same three long responses fooled the expert reviewer. On A, whose responses are
short, review matched verification on 39 of 40 items and judge on 33 of 40.

Verification then showed a clear length gradient among B responses the reviewer had passed:

| B responses passed by review, by length (tokens) | in pool | verified | verified correct |
|---|---|---|---|
| under 600 | 272 | 38 | 38 (100%) |
| 600 to 830 | 106 | 20 | 20 (100%) |
| 830 to 1000 | 33 | 6 | 5 (83%) |
| 1000 to 1200 | 45 | 12 | 8 (67%) |
| 1200 to 1500 | 28 | 10 | 4 (40%) |
| 1500 and over | 21 | 6 | 3 (50%) |

Responses the reviewer failed were essentially always wrong (0 of 20 for B, 1 of 21 for A). So review
is a reliable filter for failures and for short passes, and unreliable only for long passes. That is the
region where I concentrated the expensive labels.

## How the numbers were produced

Budget spent: 500.0 of 500.0 credits.

1. **Pilot (172 credits).** 40 random items, both models: full verification, expert review, and three crowd
   votes each. Purpose: measure each cheap signal against ground truth. Crowd majority was no better
   than review and inherited the same long-response blind spot, so crowd was dropped.
2. **Expert review of the full pool (112 credits).** Every one of the 1200 responses now has a review
   verdict. Used as the stratification variable, never as truth.
3. **Stratified verification (208 credits, 104 more items).** Each model's pool was split into five strata:
   review-failed, and review-passed split into length quartiles. Sampling was tilted toward B's long
   review-passed responses, where the pilot showed disagreement. Total verified: 72 A, 112 B.
4. **Estimate.** Within each stratum the verified pass rate is applied to the stratum's pool count and the
   strata are summed. Three variants were run for B: coarse quartile strata (0.755), finer length bins
   (0.762) and a logistic model in log-length (0.762). The reported 0.757 sits in the middle; the variants
   differ by less than 0.01. A is 0.737 under its strata, matching the review rate closely because review
   is accurate on short responses.

## How much to trust it

* **Q1** is solid: the interval half-width is about 0.035 and all three estimators agree within 0.01.
  All of B's uncertainty sits in the 127 long review-passed responses, of which 34 were verified.
* **Q2** has a standard error of roughly 0.02, about the size of the estimate. The sign is more likely
  positive than negative, but a true difference anywhere from slightly negative to about +0.05 is
  consistent with the data.
* **Q3** therefore reads SHIP on the point estimate, but the board should treat it as "B is at least not
  worse, and probably marginally better", not as a demonstrated improvement. If a clear win is required
  before launch, the cheapest way to get it is to fully verify the remaining 93 long (over 830 tokens)
  review-passed B responses, about 186 credits; that stratum alone carries most of the remaining variance.
* Practical note for the board: B's apparent gains under the LLM judge are mostly an artifact of length.
  B's failures cluster in long answers to science and writing prompts, which passed both the judge and a
  two-minute review but did not survive claim-by-claim verification. Any future eval that relies on the
  judge should be recalibrated against verification on long responses.

## Files

* `/app/data/pool.csv` - original judge verdicts and lengths.
* `/app/data/verify_*.csv` - 184 full-verification labels (ground truth).
* `/app/data/review_*.csv` - expert review for all 1200 responses.
* `/app/data/crowd_*.csv` - 240 crowd votes from the pilot.
* `/app/data/strata_table.md` - per-stratum counts used in the estimate.
