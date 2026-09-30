"""Decoy: a single random 30% holdout (a 'cleaner' validation that is still row-level)."""
import numpy as np
from sklearn.model_selection import train_test_split
from candidates import candidates


def select(X, y):
    a, b, ya, yb = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = {n: np.mean((mk().fit(a, ya).predict(b) - yb) ** 2) for n, mk in candidates()}
    return min(sc, key=sc.get)
