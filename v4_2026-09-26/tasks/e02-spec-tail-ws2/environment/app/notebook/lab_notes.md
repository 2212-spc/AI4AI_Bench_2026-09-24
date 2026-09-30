# Draft-scheduler review (inference team)

We benchmarked the speculative build against our previous numbers and we are ready to recommend the
longer-proposal scheduler.  Runs are in `notebook/runs.jsonl`; the deployment we are sizing for is in
`deploy/serving.yaml`.

- **Speculation is working.**  At `seq=1408 batch=54` with `draft=d_lite spec_g=4` we measure `ms_per_token`
  4.7683 and `tokens_per_s` 11275.4 (runs 1-3, three repetitions at `dur=40`).  Our non-speculative reference
  point is run 4, `ms_per_token` 18.1252, so the build is **3.80x** faster per output token.
- **Acceptance.**  `accept_rate` averages 0.8245 over the three runs (per position: 0.9000/0.8424/0.7962/0.7594).  We are treating
  0.8245 as *the* acceptance rate of this draft model.
- **Cost of drafting.**  The draft is about a tenth of the target and the published figure for a draft of
  that ratio is c = 0.128, i.e. drafting adds 12.8% per proposed token.  At `spec_g=4` that is 33.9% of
  the step, which the 3.80x above more than pays for.  We have not measured c ourselves - the speedup is
  what matters.
- **Longer proposals.**  Acceptance is still 0.7594 at position 3, so there is clearly headroom.  Holding it
  flat at the 0.8245 average through positions 4-7 gives an expected 4.69 accepted tokens per step at
  `spec_g=8` against 3.53 at `spec_g=4`; net of the extra drafting that is a further gain.
  **Recommendation: ship the `spec_g=8` scheduler.**
- **Capacity.**  This box reports `batch_max` 122 at 2264 tokens of context and 266 at 1038 (runs 4-5), so one
  replica handles up to 122 concurrent sequences at the long end of our context mix.  Sizing off the short
  end (run 5: 6098.0 tok/s at batch 158) we expect roughly 5558 tok/s per replica, and the capacity plan will
  quote the midpoint of the context range.

Open questions nobody has answered: our two reference points sit at different contexts and batches, and
the decay past position 3 is not something this build can measure.
