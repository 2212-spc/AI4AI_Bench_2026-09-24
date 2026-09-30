"""Baseline: label = 'has defaulted by the snapshot', gradient boosting on the features."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
FEATS = [f"x{i}" for i in range(1, 7)]


def fit_predict(history, snapshot_month, new):
    y = history.default_month.notna().astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0)
    return m.fit(history[FEATS], y).predict_proba(new[FEATS])[:, 1]
