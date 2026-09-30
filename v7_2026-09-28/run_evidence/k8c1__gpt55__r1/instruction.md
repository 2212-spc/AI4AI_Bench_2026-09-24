# Postmortem: the fine-tune that looked fine offline

Release `ft-2026.09` shipped last week and online quality on the **watched slice** dropped. The offline
eval suite did not catch it - that is why it shipped.

What the release report says:

| | before (`ft-2026.08`) | after (`ft-2026.09`) |
|---|---|---|
| offline, watched slice | 0.8986 | 0.8871 |
| offline, everything else | ~0.79 | 0.7862 |
| online, watched slice | 0.8883 | **0.7194** |

Four knobs describe a fine-tune in this project. This release changed **three** of them, each for its own
reason, and none of those reasons has gone away. Here is the recipe, what shipped, and how far back each
one may be moved:

    knob       recipe   shipped   may be restored only as far as
    capacity   1.0      0.539     0.691     student capacity headroom on the watched slice
    mixture    1.0      0.548     0.697     the watched slice's share of the fine-tuning mixture
    lr_scale   1.0      1.389     1.261     fine-tuning learning rate vs the recipe default
    leak       0.0      ?         -        fraction of the eval slice that overlaps the training set

**One of those three changes caused the online regression. The release report cannot tell you which** -
all three shipped together, and the offline number came out clean. `leak` is not a knob anyone sets: it is
a property of the eval slice, it is *measured*, this quarter's overlap audit is still open, and the `0.0`
above is the recipe's assumption rather than a measurement of what shipped.

`capacity` and `mixture` cannot go above their recipe values - there is no more serving budget and no more
licensed data.

## The lab

Run `/app/bin/lab spec` first. Budget: **20 re-runs**.

* `lab rerun` re-runs the fine-tune with knobs you choose and gives you the **offline** numbers (1 re-run,
  more if it has to retrain);
* `lab canary` serves a config to live traffic and gives you the **online** number - this is the number that
  regressed, and it costs **4 re-runs** a go;
* `lab decontaminate` re-scores a config on an eval slice with the train-overlapping items removed (1 re-run).

Knobs you leave out of a config keep their **release** value, not their recipe value.

## Deliverable: `/app/report.json`

    {"cause": "<one of ['capacity', 'mixture', 'lr_scale']>",
     "fix":   {"<knob>": <value>, ...}}

## How it is graded (mechanically, against the simulator; only the file counts)

1. **R0** - `report.json` parses, names a cause from ['capacity', 'mixture', 'lr_scale'] and gives a fix.
2. **R1** - running your fix recovers **>= 90%** of what is reachable online under the release
   constraint, measured on 3 hidden re-runs.
3. **R2** - your declared cause survives the grader's counterfactual: resetting *only* that knob must
   recover the regression, and holding *only* that knob at its release value must reproduce it.
4. **R3** - your fix actually moves the knob you blamed.

All four must hold. Note what R1 and R3 do together: naming the right knob is not a fix, and a fix that
recovers quality by accident does not tell you the cause.

Write `/app/report.json` and stop.
