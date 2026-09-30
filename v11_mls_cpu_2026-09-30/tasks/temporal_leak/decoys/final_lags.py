"""Decoy (final_lags): reference pipeline, but past sales read as the latest stored value of each day.
Original doc: rebuild every training row exactly as the warehouse looked at the end of the day before
(point-in-time), so train and live features mean the same thing, then nowcast the not-yet-final counts.

  * sales: for target day d, the value of day d-k is the latest version with recorded_at <= d-1.
    Young versions are incomplete (or over-complete when returns are netted later); the revision curve
    c[age] = version/final is estimated from days that are final in the dump and divided out.
  * level: exponentially weighted mean of nowcast sales divided by the store's weekday profile, skipping
    promo and stockout days; base = level * weekday factor.
  * promos: only promos with planned_at <= d-1 (flash promos are unknown the night before).
  * weather: the forecast issued at d-1 (actuals do not exist yet).
"""
import numpy as np, pandas as pd

K = 35


def _versions(sales_log):
    v = sales_log.sort_values(["store", "day", "recorded_at"]).copy()
    v["age"] = v.groupby(["store", "day"]).cumcount()            # revision number, 0 = first report
    return v


def _revision_curve(v):
    """c[store, rev] = sum(version rev) / sum(final), from days whose revisions are surely complete"""
    fin = v.groupby(["store", "day"], as_index=False).agg(fin=("units", "last"), last=("recorded_at", "max"))
    span = int((v.recorded_at - v.day).max())
    fin = fin[fin["last"] <= v.recorded_at.max() - span - 1]
    vv = v.merge(fin[["store", "day", "fin"]], on=["store", "day"])
    num = vv.groupby(["store", "age"]).units.sum().unstack(); den = vv.groupby(["store", "age"]).fin.sum().unstack()
    c = (num / den).ffill(axis=1).fillna(1.0)
    return c


def _pit_matrix(v, rows):
    """M[i, k-1] = value of day d_i-k as visible at the end of day d_i-1, and its age"""
    q = pd.concat([pd.DataFrame(dict(store=rows.store.to_numpy(), tday=(rows.day - k).to_numpy(),
                                     cutoff=(rows.day - 1).to_numpy(), k=k, _i=np.arange(len(rows)))) for k in range(1, K + 1)],
                  ignore_index=True).sort_values("cutoff")
    m = pd.merge_asof(q, v.rename(columns={"day": "tday"}).sort_values("recorded_at"), left_on="cutoff",
                      right_on="recorded_at", by=["store", "tday"], direction="backward")
    U = m.pivot(index="_i", columns="k", values="units").reindex(index=np.arange(len(rows)), columns=range(1, K + 1))
    A = m.pivot(index="_i", columns="k", values="age").reindex(index=np.arange(len(rows)), columns=range(1, K + 1))
    return U.to_numpy(float), A.to_numpy(float)


def build(tables, rows):
    rows = rows.reset_index(drop=True)
    v = _versions(tables["sales_log"])
    c = _revision_curve(v)
    U, A = _pit_matrix(v.assign(recorded_at=v.day), rows)   # 'latest value of each day' view
    C = c.reindex(rows.store.to_numpy()).fillna(1.0).to_numpy()
    Ai = np.clip(np.nan_to_num(A, nan=C.shape[1] - 1), 0, C.shape[1] - 1).astype(int)
    N = U                                   # nowcast of the final count
    days = rows.day.to_numpy()[:, None] - np.arange(1, K + 1)[None, :]
    # events known at d-1 about past days: promos that ran, stockouts
    p = tables["promos"]; so = tables["stockouts"]
    ev = set(zip(p.store, p.day)) | set(zip(so.store, so.day))
    st = rows.store.to_numpy()
    bad = np.array([[(s, d) in ev for d in dd] for s, dd in zip(st, days)])
    # weekday profile per store from final, event-free history in the dump
    fin = v.groupby(["store", "day"], as_index=False).units.last()
    fin = fin[~pd.Series(list(zip(fin.store, fin.day))).isin(ev).to_numpy()]
    fin["dow"] = fin.day % 7
    prof = fin.groupby(["store", "dow"]).units.mean().unstack()
    prof = prof.div(prof.mean(1), axis=0)
    W = prof.reindex(st).to_numpy()                                        # rows x 7
    wd = np.take_along_axis(W, (days % 7).astype(int), 1)
    # temperature elasticity: log sales vs actual temperature within store-weeks (known past data only)
    wa = tables["weather"]; wa = wa[wa.kind == "actual"][["store", "day", "temp"]]
    h = fin.merge(wa, on=["store", "day"])
    h = h[h.units > 0]
    h["y"] = np.log(h.units) - np.log(prof.reindex(h.store).to_numpy()[np.arange(len(h)), h.dow.to_numpy()])
    h["wk"] = h.day // 7
    g = h.groupby(["store", "wk"])
    yc = h.y - g.y.transform("mean"); tc = h.temp - g.temp.transform("mean")
    beta = float((yc * tc).sum() / (tc * tc).sum()) if len(h) > 50 else 0.0
    ta = rows[["store"]].assign(_i=np.arange(len(rows))).merge(pd.DataFrame(dict(k=np.arange(1, K + 1))), how="cross")
    ta["day"] = rows.day.to_numpy()[ta._i] - ta.k
    ta = ta.merge(wa, on=["store", "day"], how="left")
    T = ta.pivot(index="_i", columns="k", values="temp").reindex(index=np.arange(len(rows)), columns=range(1, K + 1)).to_numpy()
    D = np.where(bad | np.isnan(N), np.nan, N / wd / np.exp(beta * (np.nan_to_num(T, nan=15.0) - 15)))
    wts = 0.85 ** np.arange(K)[None, :] * ~np.isnan(D)
    lvl = np.nansum(np.nan_to_num(D) * wts, 1) / wts.sum(1)
    wts2 = 0.95 ** np.arange(K)[None, :] * ~np.isnan(D)
    lvl_slow = np.nansum(np.nan_to_num(D) * wts2, 1) / wts2.sum(1)
    dow_t = (rows.day % 7).to_numpy()
    X = pd.DataFrame({"dow": dow_t})
    X["wfac"] = W[np.arange(len(rows)), dow_t]
    X["base"] = lvl * X.wfac; X["base_slow"] = lvl_slow * X.wfac
    X["lvl1"] = D[:, 0]; X["lvl2"] = D[:, 1]; X["trend"] = lvl / lvl_slow
    q = rows.merge(p, on=["store", "day"], how="left")
    q = q[(q.planned_at.isna()) | (q.planned_at <= q.day - 1)]
    disc = rows.merge(q.groupby(["store", "day"], as_index=False).discount.max(), on=["store", "day"], how="left")
    X["discount"] = disc.discount.fillna(0.0).to_numpy()
    X["base_x_promo"] = X.base * (1 + 2 * X.discount)
    w = tables["weather"]; w = w[w.kind == "forecast"][["store", "day", "temp", "recorded_at"]]
    wf = rows.merge(w, on=["store", "day"], how="left")
    X["temp_fcst"] = np.where(wf.recorded_at <= wf.day - 1, wf.temp, np.nan)
    X["base_temp"] = X.base_x_promo * np.exp(beta * (X.temp_fcst.fillna(15.0) - 15))
    return X
