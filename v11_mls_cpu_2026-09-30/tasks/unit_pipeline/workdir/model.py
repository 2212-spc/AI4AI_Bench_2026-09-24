"""Fixed model used by the grader (do not edit; the grader uses its own copy).
Every non-numeric column is dropped except 'site', which is used as a categorical feature if present."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def _matrix(df, sites):
    X = df.drop(columns=[c for c in df.columns if c != "site" and not np.issubdtype(df[c].dtype, np.number)])
    if "site" in X:
        X = X.assign(site=pd.Categorical(X["site"], categories=sites).codes.astype(float))
        X.loc[X.site < 0, "site"] = np.nan
    return X


def fit(df, y):
    sites = sorted(df["site"].unique()) if "site" in df else []
    X = _matrix(df, sites)
    cat = [c == "site" for c in X.columns]
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, min_samples_leaf=20,
                                      categorical_features=cat if any(cat) else None, random_state=0)
    m.fit(X, y)
    return (m, sites, list(X.columns))


def predict(model, df):
    m, sites, cols = model
    return m.predict(_matrix(df, sites)[cols])
