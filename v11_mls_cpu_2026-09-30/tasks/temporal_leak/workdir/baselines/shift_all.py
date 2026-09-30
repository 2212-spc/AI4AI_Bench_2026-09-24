"""'Fix' the leakage by shifting every day-level input back by one day."""
import numpy as np, pandas as pd


def build(tables, rows):
    s = tables["sales_log"].sort_values("recorded_at").groupby(["store", "day"], as_index=False).last()
    s = s.sort_values(["store", "day"])
    g = s.groupby("store")["units"]
    s["roll7"] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    s["lag1"] = g.shift(1); s["lag7"] = g.shift(7)
    X = rows.merge(s[["store", "day", "roll7", "lag1", "lag7"]], on=["store", "day"], how="left")
    prev = rows.assign(day=rows.day - 1)
    p = tables["promos"].groupby(["store", "day"], as_index=False).discount.max()
    X["discount_prev"] = prev.merge(p, on=["store", "day"], how="left").discount.fillna(0.0).to_numpy()
    w = tables["weather"]; w = w[w.kind == "actual"][["store", "day", "temp"]]
    X["temp_prev"] = prev.merge(w, on=["store", "day"], how="left").temp.to_numpy()
    so = tables["stockouts"].assign(stockout=1.0)[["store", "day", "stockout"]]
    X["stockout_prev"] = prev.merge(so, on=["store", "day"], how="left").stockout.fillna(0.0).to_numpy()
    X["dow"] = X.day % 7
    return X.drop(columns=["day"])
