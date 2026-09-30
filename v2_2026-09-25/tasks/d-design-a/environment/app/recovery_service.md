# Cold-storage recovery requests

The evaluation harness writes a per-run trace bundle to cold storage before it writes anything to the
archive, and the bundles are still there.  A **recovery request** names one launched run and returns its
exact `pass_rate`, the same number the archive should have kept.  Recoveries are slow and are charged to
the team's quarterly budget, which is why the planning questions in `queries.json` ask how much a given
request would be worth *before* it is issued.

Two properties of the service matter for those questions:

  * a request is issued blind.  You commit to a set of runs and pay for it; you do not get to look at the
    first value before choosing the second.
  * a recovery returns the value that run actually had.  It is not an estimate and it cannot come back
    with a number that contradicts anything already in this archive - but which number it will be is not
    something the archive determines, except where the archive already pins it.
