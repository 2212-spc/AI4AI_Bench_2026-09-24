# `pass_rate`

Every run reports exactly one number, `pass_rate`, on the suite named in its plan row.  What the harness
writes depends on how the suite is scored, which `suites.json` records along with `random_guess`, the
score an uninformed answer earns on that suite:

  * `scoring: "raw"` - the fraction of the suite's items the run passed, reported as is, except that the
    harness never emits a value below `random_guess`: a run that scores under the baseline is written out
    at the baseline.
  * `scoring: "normalized"` - the raw fraction rescaled so that `random_guess` maps to 0 and a perfect
    run maps to 1, clipped below at 0.

Either way the number lies in [0, 1] and is rounded to four decimals before anything else sees it, so
every `pass_rate` in this archive - recorded, screened out, or lost - is a multiple of 0.0001.
