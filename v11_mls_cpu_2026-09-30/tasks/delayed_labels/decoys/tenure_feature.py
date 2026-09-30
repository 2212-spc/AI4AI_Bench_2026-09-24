"""Decoy: keep all accounts, label = defaulted by snapshot, add months-on-book as a feature and predict
with months-on-book = 12 (a popular shortcut for immature labels)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
FEATS = [f"x{i}" for i in range(1, 7)]


def fit_predict(history, snapshot_month, new):
    end = history[["default_month", "close_month"]].min(axis=1).fillna(snapshot_month - 1)
    mob = end - history.open_month + 1
    X = history[FEATS].assign(mob=mob)
    y = history.default_month.notna().astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0).fit(X, y)
    return m.predict_proba(new[FEATS].assign(mob=12))[:, 1]
