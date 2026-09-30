"""Decoy (generic ML): regress loss on hand-made schedule features over all measured points
(lr area, current lr, recent lr averages at several windows, lr drop, step) with gradient boosting + ridge blend."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV


def _feats(lrs, idx):
    S1 = np.cumsum(lrs); peak = lrs.max()
    F = []
    for t in idx:
        f = [np.log1p(S1[t]), S1[t] ** -0.5, lrs[t], np.log1p(t)]
        for w in (10, 50, 200, 800):
            a = max(0, t - w + 1); f.append(lrs[a:t + 1].mean())
        f.append(lrs[: t + 1].max() - lrs[t])
        F.append(f)
    return np.array(F)


def fit_predict(train, queries):
    X = np.vstack([_feats(c["lrs"], c["steps"] - 1) for c in train.values()])
    y = np.concatenate([c["loss"] for c in train.values()])
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05).fit(X, y)
    r = RidgeCV(alphas=np.logspace(-6, 2, 20)).fit(X, y)
    out = {}
    for k, lrs in queries.items():
        x = _feats(lrs, [len(lrs) - 1])
        out[k] = float(0.5 * g.predict(x)[0] + 0.5 * r.predict(x)[0])
    return out
