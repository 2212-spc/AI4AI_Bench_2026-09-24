"""Baseline: standard shuffled 5-fold CV, pick the lowest mean MSE."""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from candidates import candidates


def select(X, y):
    best, name = np.inf, None
    for n, mk in candidates():
        s = -cross_val_score(mk(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error").mean()
        if s < best:
            best, name = s, n
    return name
