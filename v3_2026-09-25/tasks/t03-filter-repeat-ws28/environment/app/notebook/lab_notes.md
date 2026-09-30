# Lab notes (data team)

- Runs 1-5: scaling runs on fresh data (q = 0, whole corpus, D = 20 N), seed 0.
- Runs 6-11: quality-filter ablation.  N = 5e7, D = 8e9 tokens on our standard ablation subset
  (sub = 1.4e+09 raw unique tokens, the subset we use for every data ablation), filter levels q = 0, 0.3,
  0.6, seeds 0 and 1.  Mean loss change vs q = 0: -0.0257 nats at q = 0.3, -0.0110 nats at q = 0.6.
  Seed-to-seed spread is about 0.01 nats, so filtering buys us nothing measurable.  Decision: drop the
  classifier, keep every token.
- Production plan: N = 1e9, D = 2e11 tokens on the next corpus snapshot (2e10-1e11 raw unique tokens once
  dedup finishes), q = 0.  That is 2-10 epochs; repetition up to ~4 epochs is nearly free in the
  literature and we accept the small cost beyond that.
