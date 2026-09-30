"""Baseline: 3x repeated shuffled 5-fold CV (lower-variance estimate), pick the lowest mean MSE."""
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score
from candidates import candidates


def select(X, y):
    sc = {n: -cross_val_score(mk(), X, y, cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=0),
                              scoring="neg_mean_squared_error").mean() for n, mk in candidates()}
    return min(sc, key=sc.get)
