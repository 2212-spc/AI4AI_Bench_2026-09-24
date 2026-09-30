"""Decoy: per-site multiplicative alignment - rescale each site's column so its median matches the
training-pool median.  Fixes any pure unit factor (lb/kg, m/cm, mmol/mg) but also erases real
population differences, and cannot handle offsets (F/C) or swaps."""
NUM = ["age", "temp", "weight", "height", "glucose", "sys_bp", "dia_bp", "heart_rate"]


def fit_preprocess(train_df):
    return {c: float(train_df[c].median()) for c in NUM}


def transform(state, df):
    d = df.copy()
    for c in NUM:
        med = d.groupby("site")[c].transform("median")
        d[c] = d[c] / med.where(med.abs() > 1e-9, 1.0) * state[c]
    return d.drop(columns=["site"])
