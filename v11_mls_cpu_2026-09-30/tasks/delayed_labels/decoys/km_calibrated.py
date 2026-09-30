"""Decoy: performance-window model, then rescale so the mean prediction matches a Kaplan-Meier estimate of
12-month default over ALL accounts (handles censoring and closures in aggregate, ignores calendar time)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
FEATS = [f"x{i}" for i in range(1, 7)]


def fit_predict(history, snapshot_month, new):
    h = history[history.open_month <= snapshot_month - 12]
    y = ((h.default_month - h.open_month) < 12).astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0).fit(h[FEATS], y)
    p = m.predict_proba(new[FEATS])[:, 1]
    end = history[["default_month", "close_month"]].min(axis=1).fillna(snapshot_month - 1)
    dur = (end - history.open_month + 1).values; ev = history.default_month.notna().values
    S = 1.0
    for k in range(1, 13):
        at = (dur >= k).sum(); d = ((dur == k) & ev).sum()
        S *= 1 - d / max(at, 1)
    return np.clip(p * (1 - S) / p.mean(), 0, 1)
