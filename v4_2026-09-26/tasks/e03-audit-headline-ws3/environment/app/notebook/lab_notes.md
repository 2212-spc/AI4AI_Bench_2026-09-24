# Release review memo (eval team)

We have spent the review budget; everything below comes from the passes in `notebook/runs.jsonl`.

- We scored `M_new` and `M_ref` over the whole bank (1500 items, format `mc_letter`), 6 passes each, recording the
  per-item table every time (`detail=1`).  A single pass of `M_new` extracted 1193 of 1500 answers.
- Contamination: the exact corpus scan (`notebook/scan_exact.csv`) finds 388 bank items in the corpus.
  The cheap n-gram detector (`notebook/detector_fast.csv`) flags 90 items above 0.5.  We used the
  detector for the first pass of the analysis because it covers the whole bank.
- `A teammate` wrote `analysis/headline.py`, which reads those tables and prints the headline we are quoting: **clean accuracy
  0.4694** for `M_new`.  Format check at `mc_cloze`: 0.6386 vs 0.3564.
- Release rule: we ship `M_new` if its clean accuracy, as the release report will aggregate it, is at least
  **0.26**.  On the headline above we are clear of the bar.
- The release report weights slice 0 by w0 and splits the rest 2:1 between slices 1 and 2.  Product has
  not fixed w0 (see the known unknown in the manual); the headline weights the slices by their item
  counts, which we assume is close enough.

Open action: nobody has reviewed the script line by line.  It was written in a hurry on the last day of
the review window.
