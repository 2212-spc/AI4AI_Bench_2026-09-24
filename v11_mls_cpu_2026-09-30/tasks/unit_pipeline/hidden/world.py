"""Hidden world for unit_pipeline.

Patients from several hospitals ("sites"). The outcome depends only on physiology in standard units.
Each site's EHR export has its own quirks (the agent is not told which):
  S1  clean
  S2  glucose exported in mmol/L instead of mg/dL
  S3  systolic/diastolic columns swapped; heart rate counted per 30 s
  S4  temperature in Fahrenheit, weight in pounds
  S5  height in metres; heart-rate missing coded as 0
Sites also genuinely differ in patient mix (age, BMI, glucose), so per-site standardisation destroys
real signal. Deployment includes sites never seen in training that reuse the same vendor quirks."""
import numpy as np
import pandas as pd

COLS = ["site", "age", "temp", "weight", "height", "glucose", "sys_bp", "dia_bp", "heart_rate"]

POP = {  # genuine population differences: (age mean, bmi mean, glucose mean, fever rate)
    "S1": (52, 26.0, 105, 0.15), "S2": (61, 28.5, 125, 0.20), "S3": (45, 25.0, 98, 0.10),
    "S4": (66, 29.5, 132, 0.30), "S5": (38, 24.0, 95, 0.12), "S6": (58, 30.5, 140, 0.25),
    "S7": (70, 27.0, 118, 0.35), "S8": (49, 26.5, 110, 0.18),
}
QUIRKS = {"S1": [], "S2": ["glu_mmol"], "S3": ["bp_swap", "hr_half"], "S4": ["temp_F", "wt_lb"],
          "S5": ["ht_m", "hr_zero"], "S6": ["temp_F", "wt_lb", "glu_mmol"], "S7": ["bp_swap", "ht_m", "hr_half"],
          "S8": []}


def physiology(rng, site, n):
    a, bmi_m, glu_m, fev = POP[site]
    age = np.clip(rng.normal(a, 14, n), 18, 95)
    height = rng.normal(170, 9, n)                       # cm
    bmi = np.clip(rng.normal(bmi_m, 4.5, n), 15, 55)
    weight = bmi * (height / 100) ** 2                   # kg
    fever = rng.random(n) < fev
    temp = np.where(fever, rng.normal(38.6, 0.6, n), rng.normal(36.8, 0.3, n))
    glucose = np.clip(rng.lognormal(np.log(glu_m), 0.25, n), 50, 450)   # mg/dL
    dia = rng.normal(78 + 0.15 * (age - 50), 9, n)
    sys_ = dia + rng.normal(45 + 0.35 * (age - 50), 10, n).clip(20, None)
    hr = rng.normal(75 + 12 * fever, 11, n)
    return pd.DataFrame(dict(site=site, age=age, temp=temp, weight=weight, height=height, glucose=glucose,
                             sys_bp=sys_, dia_bp=dia, heart_rate=hr))


def outcome(df, rng):
    bmi = df.weight / (df.height / 100) ** 2
    pp = df.sys_bp - df.dia_bp
    y = (0.035 * (df.age - 50) + 0.10 * (bmi - 26) + 1.1 * np.maximum(df.temp - 37.6, 0)
         + 0.012 * (df.glucose - 110) + 0.9 * np.tanh((pp - 50) / 15) + 0.025 * (df.heart_rate - 75)
         + 0.6 * ((bmi > 32) & (df.glucose > 150)))
    return y.values + rng.normal(0, 0.45, len(df))


def export(df, rng):
    """Apply each site's EHR quirks to clean physiology."""
    out = df.copy()
    for site, q in QUIRKS.items():
        m = out.site == site
        if "glu_mmol" in q: out.loc[m, "glucose"] = out.loc[m, "glucose"] / 18.016
        if "bp_swap" in q: out.loc[m, ["sys_bp", "dia_bp"]] = out.loc[m, ["dia_bp", "sys_bp"]].values
        if "temp_F" in q: out.loc[m, "temp"] = out.loc[m, "temp"] * 1.8 + 32
        if "wt_lb" in q: out.loc[m, "weight"] = out.loc[m, "weight"] * 2.20462
        if "ht_m" in q: out.loc[m, "height"] = out.loc[m, "height"] / 100
        if "hr_half" in q: out.loc[m, "heart_rate"] = out.loc[m, "heart_rate"] / 2   # beats per 30 s
        if "hr_zero" in q:
            z = m & (rng.random(len(out)) < 0.12)
            out.loc[z, "heart_rate"] = 0.0
    for c in ["temp"]:
        out[c] = out[c].round(1)
    for c in ["weight", "height", "glucose", "sys_bp", "dia_bp", "heart_rate", "age"]:
        out[c] = out[c].round(1 if c not in ("height",) else 2)
    return out


def make(seed, mix, n, clean=False):
    rng = np.random.default_rng(seed)
    sites = rng.choice(list(mix), size=n, p=np.array(list(mix.values())) / sum(mix.values()))
    parts = [physiology(rng, s, int((sites == s).sum())) for s in mix]
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = outcome(df, rng)
    return (df.round(2) if clean else export(df, rng)), y


TRAIN_MIX = {"S1": 0.40, "S2": 0.25, "S3": 0.18, "S4": 0.12, "S5": 0.05}
# name, seed, deployment mix, hidden
SETTINGS = [
    ("dev_mix", 31, TRAIN_MIX, False),
    ("dev_S4", 32, {"S4": 0.7, "S1": 0.3}, False),
    ("hid_S5", 301, {"S5": 0.6, "S2": 0.4}, True),
    ("hid_new6", 302, {"S6": 0.8, "S1": 0.2}, True),     # unseen site, S4+S2 style quirks
    ("hid_new7", 303, {"S7": 0.7, "S3": 0.3}, True),     # unseen site, swap + metres
    ("hid_new8", 304, {"S8": 1.0}, True),                # unseen clean site (fixes must not hurt)
]
N_TRAIN, N_TEST = 6000, 3000


def build_train(clean=False):
    return make(7, TRAIN_MIX, N_TRAIN, clean)


def build_setting(row, clean=False):
    name, seed, mix, hidden = row
    X, y = make(seed, mix, N_TEST, clean)
    return dict(name=name, X=X, y=y, hidden=hidden)
