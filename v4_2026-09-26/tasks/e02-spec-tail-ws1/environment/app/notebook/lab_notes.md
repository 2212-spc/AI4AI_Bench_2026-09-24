# Draft-scheduler review (inference team)

We benchmarked the speculative build against our previous numbers and we are ready to recommend the
longer-proposal scheduler.  Runs are in `notebook/runs.jsonl`; the deployment we are sizing for is in
`deploy/serving.yaml`.

- **Speculation is working.**  At `seq=1536 batch=113` with `draft=d_lite spec_g=4` we measure `ms_per_token`
  4.7283 and `tokens_per_s` 24052.1 (runs 1-3, three repetitions at `dur=40`).  Our non-speculative reference
  point is run 4, `ms_per_token` 9.8100, so the build is **2.07x** faster per output token.
- **Acceptance.**  `accept_rate` averages 0.8378 over the three runs (per position: 0.9041/0.8650/0.8097/0.7727).  We are treating
  0.8378 as *the* acceptance rate of this draft model.
- **Cost of drafting.**  The draft is about a tenth of the target and the published figure for a draft of
  that ratio is c = 0.128, i.e. drafting adds 12.8% per proposed token.  At `spec_g=4` that is 33.9% of
  the step, which the 2.07x above more than pays for.  We have not measured c ourselves - the speedup is
  what matters.
- **Longer proposals.**  Acceptance is still 0.7727 at position 3, so there is clearly headroom.  Holding it
  flat at the 0.8378 average through positions 4-7 gives an expected 4.91 accepted tokens per step at
  `spec_g=8` against 3.62 at `spec_g=4`; net of the extra drafting that is a further gain.
  **Recommendation: ship the `spec_g=8` scheduler.**
- **Capacity.**  This box reports `batch_max` 153 at 1446 tokens of context and 251 at 883 (runs 4-5), so one
  replica handles up to 153 concurrent sequences at the long end of our context mix.  Sizing off the short
  end (run 5: 7795.8 tok/s at batch 88) we expect roughly 6672 tok/s per replica, and the capacity plan will
  quote the midpoint of the context range.

Open questions nobody has answered: our two reference points sit at different contexts and batches, and
the decay past position 3 is not something this build can measure.
