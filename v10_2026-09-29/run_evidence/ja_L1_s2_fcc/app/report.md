# Launch review: model B vs model A

## Recommendation: HOLD

Under full verification (the board's ground truth), B's estimated pass rate is about 0.635,
A's is about 0.700, so B is roughly 6.5 points worse. The LLM judge and expert review both
say the opposite (B ahead by 5 to 7 points). That gap is an artefact of how long B's responses
are, explained below.

## What I bought (500 credits)

| Purchase | Items | Cost |
|---|---|---|
| Expert review, both models, whole pool | 1200 | 120 |
| Full verification, model A | 33 (25 random + 8 longest review-passed) | 66 |
| Full verification, model B | 153 (25 random + 128 stratified by length) | 306 |
| Crowd votes, 3 per response, pilot set | 150 | 7.5 |

## What the labels show

* **Model A.** Expert review agreed with full verification on all 33 verified A responses,
  including the 8 longest ones. A's responses are short (median 203 tokens), so a reviewer
  with two minutes can actually check them. I therefore take A's review pass rate on the
  full pool, 0.700, as its verified pass rate.
* **Model B, review says fail.** All 13 verified B responses that review failed were also
  failed by verification. A review "fail" is reliable for B.
* **Model B, review says pass.** Reliability collapses with length. B's responses are 3 to 4
  times longer than A's (median 729 tokens, up to 4360). Among review-passed B responses:

  | B length (tokens) | verified | verification pass rate |
  |---|---|---|
  | under 900 | 59 | 1.00 |
  | 900 to 1100 | 17 | 0.88 |
  | 1100 to 1500 | 30 | 0.60 |
  | 1500 to 2200 | 29 | 0.34 |
  | over 2200 | 5 | 0.00 |

  The LLM judge shows the same pattern (it agrees with review 93% of the time on B). Long
  responses look thorough to a time-boxed reviewer or a rubric-driven judge, but when an
  expert actually re-derives the answer and runs the code, most of the long ones fail.
* **Crowd votes** were noisy (3-vote majorities disagreed with verification on several
  items) and also tended to pass long B responses. Not used in the estimate.

## How Q1 and Q2 were computed

For B, every verified response uses its verified verdict (153 of 600). For the remaining
447, review-failed responses count as 0, and review-passed responses are weighted by a
logistic model of verification pass probability as a function of log length, fitted on the
140 verified review-passed B responses. A simpler three-band lookup table gives the same
answer to within 0.001.

* Q1 (B pass rate): 0.635. Bootstrap 95% interval 0.62 to 0.65.
* A pass rate: 0.700 (review, validated against verification on 33 of 33).
* Q2: -0.065. Even if A's true rate were as low as 0.67 (review being wrong on ~3% of A
  items, which the sample gives no sign of), B would still be behind.

## How much to trust this

High confidence in the direction (HOLD). The 25-item random pilot alone put B at 0.80 vs A
at 0.80, but that sample was too small; the 153 verified B items, sampled across length bands,
show the long-response failure clearly and consistently in every category (writing worst,
science and advice best). Q1 is uncertain by roughly plus or minus 0.015; Q2 by roughly the
same plus any residual review error on A, which the evidence suggests is near zero.

## Note for the eval team

The LLM judge and the two-minute expert review both systematically over-credit long
responses. Any future comparison where one model is much more verbose than the other should
not rely on either signal alone; verification of a length-stratified sample is needed.
