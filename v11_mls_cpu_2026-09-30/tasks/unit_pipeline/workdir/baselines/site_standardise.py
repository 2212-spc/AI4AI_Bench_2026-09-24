"""Baseline: per-site z-scoring of every numeric column (standard fix for 'site batch effects'),
computed on each incoming batch, site id dropped."""
NUM = ["age", "temp", "weight", "height", "glucose", "sys_bp", "dia_bp", "heart_rate"]


def fit_preprocess(train_df):
    return {}


def transform(state, df):
    d = df.copy()
    for c in NUM:
        g = d.groupby("site")[c]
        d[c] = (d[c] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return d.drop(columns=["site"])
