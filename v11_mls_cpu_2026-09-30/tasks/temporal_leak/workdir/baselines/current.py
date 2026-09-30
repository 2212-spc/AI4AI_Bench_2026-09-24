"""Current feature code (built offline from the warehouse dump; offline backtests look great)."""
import numpy as np, pandas as pd


def build(tables, rows):
    s = tables["sales_log"].sort_values("recorded_at").groupby(["store", "day"], as_index=False).last()
    s = s.sort_values(["store", "day"])
    s["roll7"] = s.groupby("store")["units"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    s["lag1"] = s.groupby("store")["units"].shift(1); s["lag7"] = s.groupby("store")["units"].shift(7)
    X = rows.merge(s[["store", "day", "roll7", "lag1", "lag7"]], on=["store", "day"], how="left")
    p = tables["promos"].groupby(["store", "day"], as_index=False).discount.max()
    X = X.merge(p, on=["store", "day"], how="left").fillna({"discount": 0.0})
    w = tables["weather"]; w = w[w.kind == "actual"][["store", "day", "temp"]]
    X = X.merge(w, on=["store", "day"], how="left")
    so = tables["stockouts"].assign(stockout=1.0)[["store", "day", "stockout"]]
    X = X.merge(so, on=["store", "day"], how="left").fillna({"stockout": 0.0})
    X["dow"] = X.day % 7
    return X.drop(columns=["day"])
