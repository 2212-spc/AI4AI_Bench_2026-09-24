You are the on-call analyst for a model-evaluation fleet.  A six-week sweep finished; the
archive in `/app` is what survived of it.

Every planned run appears in `plan.csv` with its four configuration knobs (`suite`, `curriculum`,
`retrieval`, `decoder`).  `launch_log.csv` says which of them actually started.  Each launched run
produced a single number, its `pass_rate` on the suite it was assigned - and the archive no longer holds
all of those numbers.  `archive_notes.md` inventories the files and `metric_card.md` says what the number
means.  Neither of them explains the gaps, and nobody is left who can: the `reason` column of
`retention_log.csv` holds internal archiver codes with no published meaning, and no write-up of the
sweep survives.  Every file in `/app` is exactly what it claims to be and no file is wrong, but what each
surviving record still implies about a missing value is entirely for you to establish from the records
themselves.

Your job is *not* to guess the missing numbers.  It is to report exactly what the surviving records still
pin down about them - no more and no less - and to state the mechanism you concluded was at work.

## The quantity

Fix a configuration `c` (one value for each of the four knobs).  `L(c)` is the set of run ids that appear
in `launch_log.csv` *and* whose `plan.csv` row has exactly those four knob values; count each run id once,
however many rows mention it.  `N(c) = |L(c)|`.

`m(c)` is the plain average of the true `pass_rate` of the runs in `L(c)`, over all `N(c)` of them,
including the ones whose value the archive no longer holds.  Every launched run has a true `pass_rate`;
the archive's failure to keep it does not remove the run from the average.  `m(c)` is a property of these
particular runs, not of some larger population - there is nothing to extrapolate to and nothing to
resample.

## What to report

Call an assignment of a value to every launched run whose `pass_rate` the archive does not hold
*consistent* if it contradicts nothing in `/app`.  For each item in `/app/queries.json`, report the
**sharp** interval: the smallest closed interval containing the quantity's value under every consistent
assignment.  An interval that is correct but wider than necessary is wrong, and so is a single number
reported as a degenerate interval when the quantity is not pinned down.

  * `cells`: the interval for `m(c)`.
  * `contrasts`: the interval for `m(to) - m(from)`.
  * `decisions`: is `m(to) - m(from)` greater than `margin`?  Answer `"yes"` if that holds under every
    consistent assignment, `"no"` if it holds under none of them, and `"cannot_tell"` otherwise.
  * `supports`: the interval for one named run's own true `pass_rate`.
  * `mechanism`: two claims about how the archive came to be missing what it is missing.  `screen_code`
    is the code in the `reason` column of `retention_log.csv` under which a row was removed *because of
    the value it held*; the other code, whatever it is, removed rows for a reason that has nothing to do
    with their values.  `grid` is the positive number that every true `pass_rate` in this sweep is an
    exact multiple of - recorded, removed, or never recorded.  Both are determined by the files; neither
    is a matter of taste.

Write `/app/answers.json`:

```json
{
  "cells":     {"b01": {"lo": 0.4131, "hi": 0.4131}},
  "contrasts": {"k01": {"lo": -0.0182, "hi": 0.0413}},
  "decisions": {"d01": "cannot_tell"},
  "supports":  {"s01": {"lo": 0.05, "hi": 0.2699}},
  "mechanism": {"screen_code": "RSN-00", "grid": 0.01}
}
```

Every id in `queries.json` must appear.  Interval endpoints are graded to an absolute tolerance of 5e-4,
so give at least four decimals; the answer is exact arithmetic, not an estimate.  `grid` is graded
exactly.  Scoring is all-or-nothing over the 34 items.  Do not edit anything under `/app` except
`answers.json`.


You have 5400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
