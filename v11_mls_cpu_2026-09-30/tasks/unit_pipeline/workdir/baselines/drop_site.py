"""Baseline: drop the site id (so unseen sites are not out-of-vocabulary) and keep the raw measurements."""


def fit_preprocess(train_df):
    return {}


def transform(state, df):
    return df.drop(columns=["site"])
