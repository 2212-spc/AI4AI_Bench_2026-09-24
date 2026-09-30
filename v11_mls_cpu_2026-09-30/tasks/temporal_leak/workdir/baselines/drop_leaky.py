"""Drop every input that could possibly be from the future; keep only past sales lags and calendar."""
import numpy as np, pandas as pd


def build(tables, rows):
    s = tables["sales_log"].sort_values("recorded_at").groupby(["store", "day"], as_index=False).last()
    full = rows[["store"]].drop_duplicates().merge(pd.DataFrame(dict(day=np.arange(rows.day.min() - 30, rows.day.max() + 1))), how="cross")
    s = full.merge(s[["store", "day", "units"]], on=["store", "day"], how="left").sort_values(["store", "day"])
    g = s.groupby("store")["units"]
    for k in [1, 2, 7, 14]: s[f"lag{k}"] = g.shift(k)
    s["mean7"] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    s["mean28"] = g.transform(lambda x: x.shift(1).rolling(28, min_periods=1).mean())
    X = rows.merge(s.drop(columns=["units"]), on=["store", "day"], how="left")
    X["dow"] = X.day % 7
    return X.drop(columns=["day"])
