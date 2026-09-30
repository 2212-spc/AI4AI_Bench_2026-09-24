# Release regression: find the change that actually cost you quality

Last week's runtime release bundled five changes, `H-1`, `H-2`, `H-3`, `H-4`, `H-5`.  They landed together because they share code: some
of them touch modules used by **both** the training pipeline and the evaluation harness.

The reported eval score dropped from **56.79** (checkpoint `previous`) to **50.29** (checkpoint `release`).
Leadership wants one change reverted - the one that will buy back the most real quality - and they want a
number for how much it buys.

You have **120 credits**.

* `lab train revert=<ids>` - retrain with those changes rolled back **in the training pipeline**.
  Returns a checkpoint id.  **8 credits.**  Training is deterministic given the reversion set: the same
  reversion always produces the same weights, so repeating a training run tells you nothing new.
* `lab score ckpt=<id> harness=<ids> n=<1..8> - run a checkpoint through the eval harness with those
  changes rolled back **in the harness**.  **n credits.**  `ckpt` may be `release`, `previous`, or any id
  `train` returned.  The two arguments are independent: you choose the weights and you choose the harness.

Each eval unit carries **+-0.06 points** of independent measurement noise (mean zero).  There is no other
randomness anywhere in this task.

## What the score is made of

The number the harness reports is

    reported = quality(weights) + offset(harness configuration)

`quality` is a property of the weights alone; `offset` is a property of the harness configuration alone.
Both are additive in the five changes.  Neither term is observable on its own - the harness only ever
reports their sum.

## Deliverable: `/app/report.json`

```json
{"culprit": "H-1",
 "effect": {"lo": 0.00, "hi": 0.00},
 "harness_only": ["H-2"]}
```

* `culprit` - the single change whose reversion raises **quality** the most.
* `effect` - an interval for the gain in **quality** from reverting exactly the change you named, in score
  points.  Width may not exceed **0.45 points**.
* `harness_only` - every change whose |quality effect| is at most **0.30 points** while its
  |offset effect| is at least **0.90 points**: changes that move the reported number without moving the
  model.  The list may be empty.  No change in this release sits near either threshold.
