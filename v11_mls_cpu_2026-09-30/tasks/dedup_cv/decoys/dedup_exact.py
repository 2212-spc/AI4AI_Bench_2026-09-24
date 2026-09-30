"""Decoy: drop exact duplicate rows, then random CV (catches textbook duplicates, not near-duplicates)."""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from candidates import candidates


def select(X, y):
    _, idx = np.unique(np.round(X, 6), axis=0, return_index=True)
    X, y = X[idx], y[idx]
    sc = {n: -cross_val_score(mk(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error").mean() for n, mk in candidates()}
    return min(sc, key=sc.get)
