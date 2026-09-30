# Release regression: which change should we roll back?

Last week's runtime release bundled five changes, `H-1`, `H-2`, `H-3`, `H-4`, `H-5`, shipped from one manifest.  Since then the in-house
eval dashboard has dropped from **58.81** (checkpoint `previous`) to **52.33** (checkpoint `release`).

Leadership will roll back exactly **one** change.  They want the one whose rollback buys back the most model
quality **as customers see it - on the external suite** - and a number for how much it buys.

You have **120 credits**.

* `lab train revert=<ids>` - retrain with those changes rolled back in the training pipeline.  Returns a
  checkpoint id.  **8 credits.**  Deterministic given the reversion set.
* `lab score ckpt=<id> n=<1..8> [harness=<ids>]` - the in-house eval harness.  **n credits**,
  **+-0.081 points** of noise per unit.  `harness` lists changes rolled back in the harness; if you omit it,
  the harness is built from the checkpoint's own manifest, i.e. with the same changes rolled back as in its
  training (`release`: none, `previous`: all five).  That is how the dashboard numbers above were produced.
* `lab ext ckpt=<id> n=<1..8>` - the external suite, run by a third party.  **n x 4 credits**,
  **+-0.30 points** of noise per unit.

`ckpt` may be `release`, `previous`, or any id `train` returned.  Both evals report accuracy in percent on
held-out problems from the same task distribution (different item sets).  Noise is independent and mean
zero; there is no other randomness in this task.

## Deliverable: `/app/report.json`

```json
{"culprit": "H-1",
 "effect": {"lo": 0.00, "hi": 0.00}}
```

* `culprit` - the single change whose rollback raises the **external-suite** score the most.
* `effect` - an interval for that gain, in external-suite points: the external score of a checkpoint retrained
  with exactly the change you named rolled back, minus that of `release`.  Width may not exceed **0.45 points**.
