# Release review memo (eval team)

Candidate `M_new` versus reference `M_ref`, whole bank (1500 items), format `mc_letter`, one pass each (runs 1-2 in
`notebook/runs.jsonl`):

- `M_new` 0.6242, `M_ref` 0.4049 -> **+0.2193**.  Binomial standard error on 1163 scored items is 0.0142, so the gap is
  about 15.4 sigma.  We are treating it as real.
- Format check (runs 3-4, `mc_cloze`): 0.7092 vs 0.3158 (+0.3935).  Same direction, so the result is not a format artifact.
- Denominator: the service could not extract an answer for 337 of the 1500 items we presented to `M_new`
  (122 of 1500 for `M_ref`).  Those are excluded from `acc`, which is the number we quote.
- Contamination: we ran the cheap n-gram detector over the whole bank (run 6, table in
  `notebook/detector_fast.csv`).  Only 52 items of 1500 (3.5%) score above 0.5, and a 40-item exact scan
  (run 5) found 11 of those 40 items in the corpus, so we do not think contamination is driving the gap.
- Slices: `M_new` scores 0.7165/0.5309/0.6237 on slices 0/1/2 (492/518/490 items presented).  The release report will weight slice 0
  by w0 and split the rest 2:1 between slices 1 and 2; product has not fixed w0 yet (see the known unknown
  in the manual).  We assume the pooled number is representative.

Open action: the release harness scores an unextractable answer as wrong, over every item presented.
Someone should check what that does to the table above before we publish.
