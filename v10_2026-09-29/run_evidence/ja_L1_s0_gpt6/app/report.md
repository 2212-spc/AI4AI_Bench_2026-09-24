# Model B launch review

I estimated the review-board ground truth using the lab's full verification service, rather than treating the LLM judge or ordinary expert review as definitive. I first obtained expert reviews for all 600 responses from each model. I then used all 500 available credits on stratified verification: 60 A responses and 130 B responses, sampled across judge/review agreement groups. For B, the pass group was further divided by response length because B's long responses behaved differently from its shorter responses.

The verification results were:

- A: 437.25 estimated correct responses out of 600, or about 0.729.
- B: 444.84 estimated correct responses out of 600, or about 0.741.
- Estimated difference: about +0.013.

The fractional counts are the post-stratified estimates from the verification samples; they do not imply that a fractional response was labelled. The verification sample found that B's short responses were very reliable, while its long responses had a substantially lower pass rate. That length-aware adjustment is why I did not simply use the raw judge or review pass rate. The verified samples also showed that B's quick reviews overstated correctness, especially for long responses. A's quick reviews were close to verification, with two sampled review false negatives.

The estimated B pass rate is higher than A's, so the recommended decision is **SHIP**. Confidence is adequate for the requested ±0.02 numerical tolerance, though the exact margin is modest and the uncertainty is driven mainly by the verified long-response stratum.
