# `sweeprun` v3.1 - operator notes

## What the sweep was

A replicated factorial over 5 knobs. Every cell that policy allowed was launched 2000 times with
independent seeds. Assignment is by design, not by operator choice: there is no selection on anything.

Two policy restrictions applied to the grid itself:

* `precision=fp16` was withdrawn before this sweep started and was never launched.
* `precision=fp8` requires the large-node partition, so it was only ever launched at `batch=1024`.

## manifest.csv

One row per *launched* run: `run_id` plus the knob settings. 96000 rows. This file is written at submit
time and is never edited afterwards.

## results.csv

One row per run that *finished and was recorded*: 65053 rows. A launched run is missing from this file for
exactly one of two reasons, and the log does not record which.

**1. Divergence guard.** `--abort-above-val-loss 2.955` is set fleet-wide. A run whose final validation
loss would exceed 2.955 is aborted and its row is discarded; nothing is written. The guard reads the
outcome, so it fires more often exactly where the loss is high.

**2. Preemption.** The cluster preempts jobs by priority class. Priority is assigned from the requested
node count, which on this cluster is a function of `batch` alone:

| batch | nodes | priority class |
|------:|------:|:---------------|
| 256 | 2 | guaranteed |
| 512 | 4 | standard |
| 1024 | 8 | spot / preemptible |

The scheduler looks at the priority class and at nothing else - it cannot see the loss, and it decides
before the run produces one. Preemption rates per class are not published anywhere and drift week to week.

## val_loss

Final validation loss, one number per recorded run. Run-to-run spread within a cell is dominated by seed
noise and is well described by a normal law with a common scale across cells.
