"""Reference: the columns are physiology recorded in site-specific UNITS / layouts, not different
populations.  Every repair is justified by physical plausibility (a value that is impossible in the
standard unit but ordinary in a known alternative unit), decided per site-batch where the unit is a
site-level property, so genuine population differences between sites are left untouched.  'site' is
dropped so deployment on unseen sites uses the same (harmonised) physiology the model learned."""
import numpy as np
import pandas as pd


def fit_preprocess(train_df):
    return {}


def _harmonise_batch(d):
    d = d.copy()
    # temperature: Fahrenheit readings are > 50 (body temperature in C is never > 45)
    f = d.temp > 50
    d.loc[f, "temp"] = (d.loc[f, "temp"] - 32) / 1.8
    # glucose: mmol/L values are < 35, mg/dL essentially never
    g = d.glucose < 35
    d.loc[g, "glucose"] = d.loc[g, "glucose"] * 18.016
    # blood pressure: systolic must exceed diastolic; a site that swapped the columns has sys<dia on ~all rows
    if (d.sys_bp < d.dia_bp).mean() > 0.5:
        d[["sys_bp", "dia_bp"]] = d[["dia_bp", "sys_bp"]].values
    else:
        sw = d.sys_bp < d.dia_bp
        d.loc[sw, ["sys_bp", "dia_bp"]] = d.loc[sw, ["dia_bp", "sys_bp"]].values
    # height in metres
    h = d.height < 3
    d.loc[h, "height"] = d.loc[h, "height"] * 100
    # weight: pounds is a site-level unit; implied median BMI > 45 is implausible for a hospital population
    bmi = d.weight / (d.height / 100) ** 2
    if np.median(bmi) > 45:
        d["weight"] = d["weight"] / 2.20462
    # heart rate 0 = missing
    d.loc[d.heart_rate <= 0, "heart_rate"] = np.nan
    # heart rate counted per 30 s: a site whose median resting HR is < 50 bpm is implausible
    if np.nanmedian(d.heart_rate) < 50:
        d["heart_rate"] = d["heart_rate"] * 2
    return d


def transform(state, df):
    out = pd.concat([_harmonise_batch(g) for _, g in df.groupby("site", sort=False)]).loc[df.index]
    return out.drop(columns=["site"])
