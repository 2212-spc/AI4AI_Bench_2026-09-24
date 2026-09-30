"""Reference: discrete-time hazard model (pooled logistic ~ proportional hazards for rare monthly events).

Every account contributes one row per month it was at risk (open, not yet defaulted / closed, before the
snapshot), labelled 1 in the month it defaulted.  Closures and the snapshot then act as ordinary
censoring, and recent cohorts contribute their early months instead of being dropped or mislabelled.
logit h = f(x) + a(tenure) + b(calendar month): the calendar term separates 'economy' from 'tenure' and
from the drifting applicant mix.  12-month default under current conditions = 1 - prod_k (1 - h(x, k, c))
with the calendar effect of the last observed quarter (3 months)."""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
FEATS = [f"x{i}" for i in range(1, 7)]
KMAX = 24


def person_months(h, T):
    end = h[["default_month", "close_month"]].min(axis=1).fillna(T - 1).clip(upper=T - 1).astype(int)
    n = (end - h.open_month + 1).values
    idx = np.repeat(np.arange(len(h)), n)
    k = np.concatenate([np.arange(1, m + 1) for m in n])
    c = h.open_month.values[idx] + k - 1
    ev = (h.default_month.values[idx] == c).astype(int)
    return h[FEATS].values[idx], k, c, ev


def fit_predict(history, snapshot_month, new):
    T = snapshot_month
    X, k, c, y = person_months(history, T)
    spl = SplineTransformer(n_knots=6, degree=3, extrapolation="linear").fit(X)

    def design(X, k, c):
        inter = np.column_stack([X[:, i] * X[:, j] for i, j in combinations(range(X.shape[1]), 2)])
        K = 10 * np.eye(KMAX)[np.minimum(k, KMAX) - 1]          # x10: dummies barely penalised
        C = 10 * np.eye(T // 3 + 1)[(T - 1 - c) // 3]            # calendar quarters counted back from snapshot
        return np.hstack([spl.transform(X), inter, K, C])

    m = LogisticRegression(C=1.0, max_iter=3000).fit(design(X, k, c), y)
    Xn = new[FEATS].values; n = len(Xn); S = np.ones(n)
    for kk in range(1, 13):
        hz = m.predict_proba(design(Xn, np.full(n, kk), np.full(n, T - 1)))[:, 1]   # last quarter = current conditions
        S *= 1 - hz
    return 1 - S
