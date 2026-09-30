"""Baseline: shuffled 10-fold CV with the one-standard-error rule (pick the most regularised config whose
CV error is within 1 SE of the best).  Regularisation order: larger k / larger leaf / smaller lr = simpler."""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from candidates import candidates


def simplicity(name):
    if name.startswith("knn_k"):
        return int(name[5:]) / 128
    if name.startswith("rf_leaf"):
        return int(name[7:]) / 250
    return int(name.split("leaf")[1]) / 300


def select(X, y):
    rows = []
    for n, mk in candidates():
        s = -cross_val_score(mk(), X, y, cv=KFold(10, shuffle=True, random_state=0), scoring="neg_mean_squared_error")
        rows.append((n, s.mean(), s.std(ddof=1) / np.sqrt(len(s))))
    b = min(rows, key=lambda r: r[1])
    ok = [r for r in rows if r[1] <= b[1] + b[2]]
    return max(ok, key=lambda r: simplicity(r[0]))[0]
