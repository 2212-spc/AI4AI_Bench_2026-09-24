"""Baseline: CUPED (Deng et al. 2013) - difference in means on the pre-period-adjusted outcome
y - theta * (x - mean x), theta = cov(y, x) / var(x), x = pre_outcome."""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    y, x = units["outcome"].to_numpy(float), units["pre_outcome"].to_numpy(float)
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    ya = y - theta * (x - x.mean())
    t = units["treated"].to_numpy() == 1
    return float(ya[t].mean() - ya[~t].mean())
