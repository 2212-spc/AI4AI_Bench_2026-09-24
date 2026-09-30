"""Decoy: performance window, but drop accounts that closed before 12 months without defaulting (their
label is 'unknown'), which looks like the careful thing to do."""
from sklearn.ensemble import HistGradientBoostingClassifier
FEATS = [f"x{i}" for i in range(1, 7)]


def fit_predict(history, snapshot_month, new):
    h = history[history.open_month <= snapshot_month - 12]
    d = (h.default_month - h.open_month) < 12
    closed_early = (h.close_month - h.open_month) < 12
    h, d = h[d | ~closed_early], d[d | ~closed_early]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0)
    return m.fit(h[FEATS], d.astype(int)).predict_proba(new[FEATS])[:, 1]
