# Task: 12-month default forecasts from an immature loan book

A lender opens new credit accounts every month. You get the account history at a snapshot date:

| column | meaning |
|---|---|
| `account_id` | id |
| `open_month` | month the account was opened (0 = oldest cohort) |
| `x1..x6` | applicant features recorded at opening |
| `default_month` | calendar month of default, NaN if it has not defaulted by the snapshot |
| `close_month` | calendar month the customer closed the account (without defaulting), NaN if still open or defaulted |

`snapshot_month` = the first month NOT observed (everything happened in months `< snapshot_month`).

**Goal.** For a batch of new applicants (features only), predict the probability that each defaults
within 12 months of opening, *assuming the account stays open* (closures are a customer choice, not a
credit outcome). Economic conditions change over time and affect everyone's default risk; the planning
assumption is that **conditions over the next 12 months will be like those of the last 3 months before
the snapshot**. Applicant mix also drifts over time.

## What to write
`solution.py` must define

```python
def fit_predict(history: pd.DataFrame, snapshot_month: int, new: pd.DataFrame) -> np.ndarray
```

returning one probability per row of `new`. numpy / pandas / scipy / scikit-learn / statsmodels only,
CPU, at most ~15 min per portfolio.

## How it is graded
Credit teams use these forecasts by risk band. Metric per portfolio: sort the new applicants into 10
deciles of their *true* risk; for each decile compute |mean predicted - mean true| / mean true; average
over deciles. Pass needs **<= 0.25 on every portfolio**: the 2 visible dev portfolios in
`data/dev_settings.pkl` (true probabilities included, check with `python3 eval_dev.py`) and 4 hidden
portfolios with different histories (lengths, volumes, closure rates, drift, and economic paths).

Starting point: `solution.py` is the standard 12-month performance-window model; `baselines/` has other
standard approaches. Write `NOTES.md` explaining your approach and why it is unbiased.
