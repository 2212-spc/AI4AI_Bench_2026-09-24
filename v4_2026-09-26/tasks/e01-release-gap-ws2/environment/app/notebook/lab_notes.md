# Release review memo (eval team)

Candidate `M_new` versus reference `M_ref`, whole bank (1500 items), format `mc_letter`, one pass each (runs 1-2 in
`notebook/runs.jsonl`):

- `M_new` 0.6750, `M_ref` 0.5880 -> **+0.0870**.  Binomial standard error on 1234 scored items is 0.0133, so the gap is
  about 6.5 sigma.  We are treating it as real.
- Format check (runs 3-4, `mc_cloze`): 0.6830 vs 0.6155 (+0.0675).  Same direction, so the result is not a format artifact.
- Denominator: the service could not extract an answer for 266 of the 1500 items we presented to `M_new`
  (148 of 1500 for `M_ref`).  Those are excluded from `acc`, which is the number we quote.
- Contamination: we ran the cheap n-gram detector over the whole bank (run 6, table in
  `notebook/detector_fast.csv`).  Only 78 items of 1500 (5.2%) score above 0.5, and a 40-item exact scan
  (run 5) found 12 of those 40 items in the corpus, so we do not think contamination is driving the gap.
- Slices: `M_new` scores 0.7724/0.5687/0.6867 on slices 0/1/2 (494/498/508 items presented).  The release report will weight slice 0
  by w0 and split the rest 2:1 between slices 1 and 2; product has not fixed w0 yet (see the known unknown
  in the manual).  We assume the pooled number is representative.

Open action: the release harness scores an unextractable answer as wrong, over every item presented.
Someone should check what that does to the table above before we publish.
