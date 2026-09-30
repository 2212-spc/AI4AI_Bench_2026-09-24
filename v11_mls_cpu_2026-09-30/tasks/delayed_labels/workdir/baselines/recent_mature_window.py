"""Baseline: performance window restricted to the most recent 12 mature cohorts (recency for drift)."""
from sklearn.ensemble import HistGradientBoostingClassifier
FEATS = [f"x{i}" for i in range(1, 7)]


def fit_predict(history, snapshot_month, new):
    h = history[(history.open_month <= snapshot_month - 12) & (history.open_month > snapshot_month - 24)]
    y = ((h.default_month - h.open_month) < 12).astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0)
    return m.fit(h[FEATS], y).predict_proba(new[FEATS])[:, 1]
