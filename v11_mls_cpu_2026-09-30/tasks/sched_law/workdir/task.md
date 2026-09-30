# Task: predict the final loss of *unseen* learning-rate schedules

You are on the training-infrastructure team. Before launching expensive runs we want to know how
the **final training loss depends on the learning-rate schedule**, so we can pick schedules
without running them. For a given model/dataset ("world") you get a few cheap *training curves*
and must predict the **final loss** of other schedules on the same world.

## What you edit
`solution.py` — implement

```python
def fit_predict(train: dict, queries: dict) -> dict:
    # train[name]   = {"lrs": np.ndarray (T,), per-step learning rate,
    #                  "steps": np.ndarray, step indices (1-based) where loss was measured,
    #                  "loss": np.ndarray, measured loss at those steps (has small eval noise)}
    # queries[name] = np.ndarray of per-step learning rates (length may differ from T)
    # return {name: predicted loss after the last step of that schedule}
```

Only `solution.py` is graded. You may add helper files next to it.
Must run in < 10 min per world on 4 CPU cores, numpy/scipy/sklearn only, no network.

## What you have
- `data/dev_worlds.pkl` — 3 dev worlds (list of dicts with `train`, `queries`, `truth`, `scale`).
  In every world the training curves are: constant lr at peak, peak/3, peak/10, and cosine-to-zero
  from peak, all with the same length T.
- `baselines/` — three published schedule-aware loss laws, reimplemented with the same interface:
  lr-area power law, the annealing-momentum law (Tissue et al. 2024) and the multi-power law
  (Luo et al. 2025). `solution.py` starts as a copy of the lr-area baseline.
- `eval_dev.py` — scores a solution on the dev worlds.

## How it is graded
Metric per world = mean over query schedules of |predicted − true final loss| / scale, where
scale = final loss(constant peak lr) − final loss(cosine), i.e. how big schedule effects are
in that world. **Pass = metric ≤ 0.10 on every world.**
Graded worlds = the 3 dev worlds + **4 hidden worlds** (other models/datasets, other peak lr,
other run lengths T). Hidden worlds are also queried on further schedule families used in
practice (e.g. warmup, restarts, re-warming after decay, runs longer than the training curves),
not only the ones in the dev queries. The true dev answers are in the pickle; the hidden ones
are never available to you.

Write a short `NOTES.md` explaining what structure you found and why you expect it to transfer.
