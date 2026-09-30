You are the on-call analyst for a model-evaluation fleet.  A six-week sweep and three
short follow-up campaigns have finished; the archive in `/app` is what survived of them.

Every planned run appears in `plan.csv` with its four configuration knobs (`suite`, `curriculum`,
`retrieval`, `decoder`).  `launch_log.csv` says which of them actually started.  Each launched run
produced a single number, its `pass_rate` on the suite it was assigned - but the archive lost some of
those numbers, in several different ways.  `incidents.md` is the on-call write-up of what happened,
`metric_card.md` says what the number means, `recovery_service.md` describes the cold-storage recovery
service, and `recovery_log.csv` is the record of every recovery request the on-call team issued during
the incident response.  Read all of them, and read the logs: how much each surviving record still
constrains a missing value is for you to work out.  Nothing in `/app` is a decoy: every file is exactly
what it claims to be, and no file is wrong.

## The quantity

Fix a configuration `c` (one value for each of the four knobs).  `L(c)` is the set of run ids that appear
in `launch_log.csv` *and* whose `plan.csv` row has exactly those four knob values; count each run id once,
however many rows mention it.  `N(c) = |L(c)|`, and `m(c)` is the plain average of the true `pass_rate` of
all `N(c)` of them, including the ones whose value the archive lost.

Call an assignment of a value to every launched run whose `pass_rate` the archive did not keep
*consistent* if it contradicts nothing in `/app`.  For a contrast `m(to) - m(from)`, its **sharp
interval** is the smallest closed interval containing its value under every consistent assignment.

## Recoveries

A recovery request names a set `S` of launched runs and is answered out of cold storage.  You must choose
`S` and pay for it before you see anything that comes back.

**This instruction does not tell you what a recovery request comes back with, and it is not the same for
every run.**  `recovery_service.md` says what kind of thing the service is; `recovery_log.csv` is a sample
of what it actually did, on runs that are not the ones you are asked about.  Working out, from those two
and from the rest of the archive, what a request for a given run would leave you knowing is the main part
of this task.  Whatever it is, it is true, it never contradicts `/app`, and it is a statement about the
runs in `S` only.

Call an **outcome** of a request for `S` one possible thing the service could come back with for every run
in `S` - one that contradicts nothing in `/app`.  Given an outcome, the sharp interval of a contrast is
recomputed with the outcome added to everything the archive already says, and it can only get narrower or
stay the same.  Which outcome you get is not yours to choose, so the only thing you can promise your
budget committee before paying is the **guaranteed width**:

    guaranteed_width(contrast, S)
        = the largest width the contrast's sharp interval can have, taken over every outcome that a
          request for S could have.

With `S` empty this is just the width of the sharp interval today.  Recovering a run is **worth taking**
for a contrast when adding it to an empty `S` makes the guaranteed width strictly smaller.

## What to report

For each item in `/app/queries.json`:

  * `contrasts`: the sharp interval for `m(to) - m(from)` today, as `lo` and `hi`.
  * `recovery`: for the single run `run_id`, the **residual width** a request naming just that run would
    leave: the width of the range that run's own `pass_rate` could still take afterwards, in the worst
    case over the outcomes of that request, counting only that run's own record and the answer itself and
    ignoring every other constraint in the archive.  Report `0` if such a request pins the value exactly,
    and the full width of the range that run's own record alone leaves it in if such a request adds
    nothing at all.
  * `values`: is recovering the single run `run_id` worth taking for that contrast?  `"yes"` or `"no"`.
  * `widths`: `guaranteed_width` for that contrast after recovering the runs listed in `recover`
    (possibly the empty list).
  * `plans`: the smallest number `k` of runs from `candidates` whose recovery guarantees a width of at
    most `target`, together with one set of that size that achieves it.  Every run you name must come
    from that item's `candidates` list.  If several sets of size `k` work, any one of them is accepted.

Write `/app/answers.json`:

```json
{
  "contrasts": {"k01": {"lo": -0.0182, "hi": 0.0413}},
  "recovery":  {"r01": 0.0},
  "values":    {"v01": "yes"},
  "widths":    {"w01": 0.135294},
  "plans":     {"p01": {"k": 2, "runs": ["r01234", "r02345"]}}
}
```

Every id in `queries.json` must appear.  Numbers are graded to an absolute tolerance of 0.0005, so give
at least six decimals; the answers are exact arithmetic, not estimates.  Scoring is all-or-nothing over
the 31 items.  Do not edit anything under `/app` except `answers.json`.


You have 5400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
