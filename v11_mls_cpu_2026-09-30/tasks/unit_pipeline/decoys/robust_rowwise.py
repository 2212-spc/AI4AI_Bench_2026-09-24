"""Decoy: row-wise plausibility repair only (F->C, mmol->mg, swap where sys<dia, m->cm, HR 0->NaN),
but weight left alone (lb vs kg is not separable per row) and site kept."""
import numpy as np


def fit_preprocess(train_df):
    return {}


def transform(state, d):
    d = d.copy()
    f = d.temp > 50; d.loc[f, "temp"] = (d.loc[f, "temp"] - 32) / 1.8
    g = d.glucose < 35; d.loc[g, "glucose"] *= 18.016
    sw = d.sys_bp < d.dia_bp; d.loc[sw, ["sys_bp", "dia_bp"]] = d.loc[sw, ["dia_bp", "sys_bp"]].values
    h = d.height < 3; d.loc[h, "height"] *= 100
    d.loc[d.heart_rate <= 0, "heart_rate"] = np.nan
    return d
