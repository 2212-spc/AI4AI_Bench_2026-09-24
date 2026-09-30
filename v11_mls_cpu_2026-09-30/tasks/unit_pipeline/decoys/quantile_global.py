"""Decoy: drop site, then a global rank/quantile transform of every column (robust to scale, but
mixed units in one column stay mixed)."""
from sklearn.preprocessing import QuantileTransformer
NUM = ["age", "temp", "weight", "height", "glucose", "sys_bp", "dia_bp", "heart_rate"]


def fit_preprocess(train_df):
    return QuantileTransformer(n_quantiles=200, output_distribution="normal").fit(train_df[NUM])


def transform(state, df):
    d = df.drop(columns=["site"]).copy()
    d[NUM] = state.transform(df[NUM])
    return d
