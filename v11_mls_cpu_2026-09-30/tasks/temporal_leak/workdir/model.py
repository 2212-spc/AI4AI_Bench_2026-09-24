"""Fixed forecasting model (do not edit): gradient-boosted trees on whatever features features.build returns."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def fit(X, y):
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      min_samples_leaf=40, random_state=0)
    return m.fit(X.to_numpy(float), y)


def predict(m, X):
    return m.predict(X.to_numpy(float))
