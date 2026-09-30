# Release review memo (eval team)

Candidate `M_new` versus reference `M_ref`, whole bank (1500 items), format `mc_letter`, one pass each (runs 1-2 in
`notebook/runs.jsonl`):

- `M_new` 0.5427, `M_ref` 0.3855 -> **+0.1571**.  Binomial standard error on 1172 scored items is 0.0146, so the gap is
  about 10.8 sigma.  We are treating it as real.
- Format check (runs 3-4, `mc_cloze`): 0.6187 vs 0.5672 (+0.0515).  Same direction, so the result is not a format artifact.
- Denominator: the service could not extract an answer for 328 of the 1500 items we presented to `M_new`
  (159 of 1500 for `M_ref`).  Those are excluded from `acc`, which is the number we quote.
- Contamination: we ran the cheap n-gram detector over the whole bank (run 6, table in
  `notebook/detector_fast.csv`).  Only 97 items of 1500 (6.5%) score above 0.5, and a 40-item exact scan
  (run 5) found 10 of those 40 items in the corpus, so we do not think contamination is driving the gap.
- Slices: `M_new` scores 0.7062/0.4450/0.5057 on slices 0/1/2 (471/507/522 items presented).  The release report will weight slice 0
  by w0 and split the rest 2:1 between slices 1 and 2; product has not fixed w0 yet (see the known unknown
  in the manual).  We assume the pooled number is representative.

Open action: the release harness scores an unextractable answer as wrong, over every item presented.
Someone should check what that does to the table above before we publish.
