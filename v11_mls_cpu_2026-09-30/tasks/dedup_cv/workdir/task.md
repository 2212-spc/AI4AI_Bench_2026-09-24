# Task: model selection that holds up in deployment

You maintain an automated model-selection step. For each dataset it receives a feature matrix `X`
(float, n x d) and a real-valued target `y`, and must choose ONE configuration from the fixed menu in
`candidates.py` (kNN, random forests, gradient boosting at several regularisation levels). The chosen
configuration is then refit on all of `X, y` and deployed to predict the target for **future records
collected by the same process**. Its deployment error (MSE against the true target) is what counts.

The records come from a real-world logging pipeline; do not assume anything about how they were
collected beyond what you can verify from the data. No record identifiers are provided.

## What to write
Edit `solution.py` so it defines

```python
def select(X: np.ndarray, y: np.ndarray) -> str   # returns a candidate name from candidates.py
```

`solution.py` may import `candidates.py`, numpy, scipy, scikit-learn. CPU only; at most ~15 minutes
per dataset on 4 cores. Do not edit `candidates.py` (the grader uses its own copy).

## How it is graded
Metric per dataset = deployment MSE of your pick / deployment MSE of the best candidate on the menu.
You pass only if the ratio is <= **1.20 on every dataset**: the 2 visible dev datasets in
`data/dev_settings.pkl` (which include each candidate's deployment MSE so you can check yourself with
`python3 eval_dev.py`) and **4 hidden datasets** from the same kind of pipeline, with different sizes,
dimensionality, noise levels and target shapes. Hard-coding answers for the dev sets will not help.

Starting point: `solution.py` is plain shuffled 5-fold CV. `baselines/` contains other standard
selection rules. Also write `NOTES.md` (what you found about the data, what you changed, why it
should hold on the hidden datasets).
