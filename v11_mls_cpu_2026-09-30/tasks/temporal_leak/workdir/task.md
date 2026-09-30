# Task: make the store-demand features honest

Every night at the end of day d-1 we forecast tomorrow's (day d) unit sales for every store, for replenishment.
The forecasting model is fixed (`model.py`, gradient-boosted trees, do not edit). What you own is
`features.py`:

```python
def build(tables: dict[str, pd.DataFrame], rows: pd.DataFrame) -> pd.DataFrame
```

`rows` has columns `store, day` (the store-days to forecast); return one feature row per input row, same
order, numeric columns only (NaN allowed). The same function is used for training and for live forecasting.

## The data warehouse (`tables`)
Every table records when each row was written:

| table | columns | notes |
|---|---|---|
| `sales_log` | store, day, units, recorded_at | a store-day can be written several times (later rows supersede earlier ones) |
| `promos` | store, day, discount, planned_at | discount running at the store on that day; `planned_at` = day it was decided |
| `weather` | store, day, kind, temp, recorded_at | `kind` is `forecast` or `actual` |
| `stockouts` | store, day, recorded_at | the store ran out of stock that day |

## How it is trained and graded
For each setting (a chain of stores):
1. **Training.** You get the warehouse dump taken at the end of day `n_hist-1` and a table of training
   targets `train_rows (store, day, units)` = the *final* unit counts of past days. The runner calls
   `X = build(dump, train_rows[["store","day"]])` and fits the fixed model on `(X, units)`.
2. **Live.** Then 60 live days are replayed one at a time: for day d the runner passes the warehouse
   **exactly as it was at the end of day d-1** (only rows written by then) and the rows `(store, d)`; the
   model predicts and the prediction is compared with the *final* units of day d.

Metrics over the 60 live days: `wape = sum|pred-y| / sum y`, divided by the wape of an oracle that knows the
true demand state up to d-1 (`wape_ratio`), and `bias = sum(pred)/sum(y) - 1`.
**Pass needs `wape_ratio <= 1.55` and `|bias| <= 0.03` on every setting**: the 2 visible dev settings in
`data/dev_settings.pkl` (full warehouse incl. the live period, check with `python3 eval_dev.py`) and 4 hidden
settings (other chains: different numbers of stores, history lengths, reporting behaviour, promo habits,
weather-forecast quality and demand volatility).

numpy / pandas / scipy / scikit-learn only, CPU; the whole run (training + 60 days) must finish within
15 minutes per setting.

## Starting point
`features.py` is the current production feature code: it backtests very well offline and disappoints in
production. `baselines/` has two earlier attempts to fix it. Write `NOTES.md` explaining what was wrong
and why your features mean the same thing at training time and at forecast time.
