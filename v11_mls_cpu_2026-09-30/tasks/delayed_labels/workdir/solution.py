"""Baseline: the standard credit 'performance window'. Keep only accounts opened >= 12 months before the
snapshot, label = defaulted within 12 months of opening, fit gradient boosting."""
from sklearn.ensemble import HistGradientBoostingClassifier
FEATS = [f"x{i}" for i in range(1, 7)]


def fit_predict(history, snapshot_month, new):
    h = history[history.open_month <= snapshot_month - 12]
    y = ((h.default_month - h.open_month) < 12).astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0)
    return m.fit(h[FEATS], y).predict_proba(new[FEATS])[:, 1]
