# Task: make a clinical risk model survive deployment to new hospitals

`data/train.csv` is a pooled EHR extract from five hospitals (`site` = S1..S5): age, temperature,
weight, height, glucose, systolic / diastolic blood pressure, heart rate, and a continuous `outcome`
(a severity score). The modelling step is FIXED: `model.py` (gradient boosting; `site` used as a
categorical feature if present; the grader uses its own copy). You control only the data preparation.

The model will be deployed in batches of patients. Deployment batches come from the training
hospitals **and from hospitals that were never in the training data**. The column schema is the
same everywhere; nothing else about how each hospital exports its data is documented.

## What to write
`preprocess.py` must define

```python
def fit_preprocess(train_df: pd.DataFrame) -> object          # train_df has no 'outcome' column
def transform(state, df: pd.DataFrame) -> pd.DataFrame         # same rows, same order; any columns
```

The grader calls `fit_preprocess` on the raw training extract, trains `model.py` on
`transform(state, train)`, and predicts each deployment batch with `transform(state, batch)` (a batch
always contains the `site` column; batches have a few thousand rows and may mix hospitals). numpy /
pandas / scipy / scikit-learn only, CPU.

## How it is graded
Metric per batch = test MSE / the MSE the same model achieves when every hospital's data is exported
correctly. Pass needs <= **1.10 on every batch**: 2 visible dev batches (`data/dev_mix.csv`,
`data/dev_S4.csv`, check with `python3 eval_dev.py`) and 4 hidden batches, which include new hospitals.
A preparation that helps on average but breaks one hospital fails.

`baselines/` has standard choices (identity, drop site, per-site standardisation). Write `NOTES.md`:
what you found, what each step does, and why it is safe for hospitals you have not seen.
