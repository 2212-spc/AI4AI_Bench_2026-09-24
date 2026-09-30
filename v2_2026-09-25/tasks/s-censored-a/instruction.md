# What the sweep would have said if you could see all of it

`/app` holds a finished hyper-parameter sweep: `manifest.csv` (every run that was launched),
`results.csv` (every run that was recorded), `sweep.json` (the grid and the launcher constants) and
`launcher.md` (how the launcher behaved). `results.csv` is missing 30947 of the 96000 launched runs.
`launcher.md` explains the two mechanisms that remove rows; only one of them looks at the outcome.

`queries.json` asks for 13 contrasts. Each is "hold this baseline cell fixed, move one knob from A to B,
what happens to the mean final validation loss" - a difference of two cell means, in loss units, at the
stated baseline. `report_cells.json` asks, for 6 named cells, what fraction of the runs launched there
were removed *by the divergence guard specifically*.

Some of the contrasts cannot be answered from this sweep at all. Saying so is a correct answer and is
worth exactly as much as a number; inventing a number for one of them is wrong, and so is refusing to
answer one that the data do determine.

## Deliverable

`/app/answers.json`:

```json
{
  "queries": {
    "q01": {"verdict": "identified", "delta": -0.0421},
    "q07": {"verdict": "underdetermined", "reason": "no_surviving_run"}
  },
  "censoring": {"r1": 0.0, "r2": 0.31}
}
```

Every query id in `queries.json` and every cell id in `report_cells.json` must appear. `delta` is
`mean(B) - mean(A)` in loss units. `reason` must be exactly one of:

* `combination_never_launched` - the cell exists in principle but the sweep never launched it
* `level_never_launched` - the level itself never appears in the sweep
* `no_surviving_run` - every launched run in one of the two cells was removed from the table

## How it is graded

* All-or-nothing over 19 items: 13 verdicts (with the reason code where you abstain), 9
  numbers, 6 fractions.
* Each query in `queries.json` carries its own `tol`, and each report cell its own; an answer counts if it
  is within that tolerance of the truth. The tolerances are not arbitrary - each is five standard
  deviations of the sampling error of a correct estimator on this much data, so a correct procedure
  clears them with room to spare. If your own procedure's error is anywhere near its tolerance, that is
  evidence the procedure is wrong, not that the tolerance is tight. You can check yours: you have enough
  information to simulate the whole pipeline and re-estimate.
* A fraction must lie in [0, 1].

Python 3 with numpy is available and there is no network.


You have 5400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
