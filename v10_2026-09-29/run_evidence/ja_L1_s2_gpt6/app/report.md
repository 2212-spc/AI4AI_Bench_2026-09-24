The recommendation is **HOLD**. Estimated full-verification pass rates on this fixed pool are **63.25% for B** and **70.17% for A**, a **6.92 percentage point disadvantage for B**. These are estimates of the board's specified correctness criterion, not averages of the cheaper graders' verdicts.

The entire 500-credit budget was spent: expert review of all 1,200 responses cost 120 credits; full verification of 60 A responses and 130 B responses cost 380 credits. No crowd votes were purchased. Each purchased response/service combination was used only once. Expert review initially favored B, 451/600 versus 420/600, but verification found errors that reverse that result.

Sampling and estimation

I first divided each model's responses by the four combinations of judge and expert-review verdicts. Within the group where both graders passed a response, I formed three equally sized groups ordered by that model's response length. A reproducible random pilot verified 30 A responses and 50 B responses across these groups. Both graders' agreement was used for sampling, never accepted as ground truth.

The pilot found six failures among 13 B responses in the longest double-pass group, and no failures among 21 B responses in its two shorter double-pass groups. A's pilot contained just one disagreement with expert review. The second stage therefore concentrated verification on longer B responses, while retaining samples from every remaining group. It verified another 30 A responses and 80 B responses. For B, the longest double-pass group was split at 1,200 and 1,500 tokens; the judge-fail/review-pass group was split at 900 tokens. Several small groups were completely verified. All second-stage selections were uniform random samples without replacement within their groups.

To account for the adaptive allocation, I counted each pilot response's verified verdict directly and estimated only the remaining population from the independent second-stage samples. For each model the estimator was:

`pass rate = (sum of pilot verified passes + sum over remaining groups of [group size × second-stage verified pass fraction]) / 600`.

Conditional on the pilot, every remaining group had positive sampling coverage, so this estimator is design-unbiased. Raw verification-sample averages would be inappropriate because difficult groups were deliberately oversampled. No assumption of equal error rates between A and B is needed, and independence of A and B responses to the same prompt is not required for estimating their difference.

The most consequential second-stage B results were in responses passed by both cheaper graders:

| Length group within the original longest third | Responses remaining after pilot | Additional verifications | Verified passes |
|---|---:|---:|---:|
| Below 1,200 tokens | 41 | 7 | 6 |
| 1,200–1,499 tokens | 34 | 22 | 8 |
| At least 1,500 tokens | 56 | 28 | 9 |

The other B second-stage groups agreed with their expert-review verdicts. Across all 60 A verifications, just one review verdict changed: a review failure became a verified pass. That response's entire judge-pass/review-fail group was verified by the end of the audit.

How much to trust the result

The launch recommendation has substantial empirical support: B's estimated disadvantage is about seven percentage points, and the audit directly confirmed many failures hidden by both cheaper graders. Response length is associated with those errors in this pool; this audit does not establish that length itself causes them. The conclusion concerns these 600 prompts and does not establish performance on future traffic or other prompt distributions.

The finite-population stratified variance estimate from the second-stage sample gives B a nominal standard error of 1.13 percentage points. However, that calculation assigns zero estimated variance to groups with all passes or all failures. Several such groups have small samples; undetected errors could add material uncertainty, including to A's estimate and the difference. In particular, A's nominal zero standard error is a sampling artifact, not proof that its estimated rate is exact. An interval of roughly plus or minus 2.2 percentage points for B based only on that standard error would omit this important uncertainty and should not be treated as a certified 95% interval.

The submitted numbers are defensible probability-sampling estimates, but this budget does not guarantee either requested two-percentage-point accuracy bound or the decision. I recommend HOLD on the current evidence. Stronger precision claims would require more full verification, including additional checks of the groups where this audit observed no errors.

The purchased verdict CSVs and both sampling-design JSON files are retained in `/app/data/`. `/app/estimate.py` reproduces the estimates and answer file without spending additional credits; `/app/data/estimates.json` contains group counts and the budget accounting. The scripts `audit.py` and `second_stage.py` record the sampling procedure and lab calls; they should not be rerun because they purchase labels.
