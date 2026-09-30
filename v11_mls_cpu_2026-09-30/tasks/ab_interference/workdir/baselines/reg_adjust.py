"""Baseline: OLS  outcome ~ treated + pre_outcome + market fixed effects; returns the treated coefficient
(within-market comparison, adjusted for the pre-period covariate)."""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    d = units.copy()
    # within-market demeaning == market fixed effects (Frisch-Waugh-Lovell)
    for c in ["outcome", "treated", "pre_outcome"]:
        d[c + "_w"] = d[c] - d.groupby("market")[c].transform("mean")
    X = d[["treated_w", "pre_outcome_w"]].to_numpy(float)
    b = np.linalg.lstsq(X, d["outcome_w"].to_numpy(float), rcond=None)[0]
    return float(b[0])
