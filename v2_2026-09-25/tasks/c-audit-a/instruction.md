# What can this log actually answer?

`/app/log.csv` is the launch-and-outcome log of 6000 completed training runs of one recipe on a shared
cluster. `/app/columns.md` documents every column, including *when* each one was measured.

Platform engineering wants 12 numbers before the next deployment. Each is in `/app/queries.json`:

```json
{"id": "q04", "knob": "grad_accum", "from": 1, "to": 2, "baseline": {...all 10 settings...}}
```

Read it as this counterfactual contrast:

> if every job in the fleet were launched with the `baseline` settings but `knob = from`, and then instead
> with the `baseline` settings but `knob = to`, by how much would the mean `val_loss` change?

The mean is over the cluster's own hardware mix and over run-to-run noise; `delta = mean(to) - mean(from)`,
so a negative delta means `to` is better.

**Some of these 12 contrasts cannot be recovered from this log at all.** Part of the job is deciding
which, and saying why. Do not guess a number for a contrast the data cannot support, and do not refuse a
contrast the data can support - both cost you the task.

## Deliverable

`/app/answers.json`, one entry per query id:

```json
{
  "q01": {"verdict": "identified", "delta": -0.0123},
  "q05": {"verdict": "underdetermined", "reason": "aliased_with_other_knob"}
}
```

* `verdict`: `"identified"` if the log determines the contrast, `"underdetermined"` if it does not.
* `delta`: required for `identified`. Correct if within 15% of the true value, or within 0.0025
  absolute, whichever is the larger margin.
* `reason`: required for `underdetermined`, exactly one of
  * `aliased_with_other_knob` - this knob never moved on its own; its effect is inseparable from another
    column's, at any sample size.
  * `level_never_observed` - a value the query asks about never appears in the log.
  * `combination_never_observed` - every value appears somewhere, but the particular combination the
    query needs is absent, so the contrast rests on an assumption the log cannot check.

Also write `/app/audit.md`: for each query, the estimator or the obstruction you found, in a few lines.

## Rules

* Offline task: `/app/log.csv`, `/app/columns.md`, `/app/queries.json` are all the evidence there is.
  There is no simulator and no way to launch a new run.
* Python 3 with numpy is available. Any method is allowed; only the numbers are graded.
* Grading is all-or-nothing over the 12 queries: every verdict, every required delta and every required
  reason has to be right.


You have 5400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
