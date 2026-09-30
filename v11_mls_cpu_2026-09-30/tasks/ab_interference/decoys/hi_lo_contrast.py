"""Decoy: 'compare near-launch vs near-holdout markets'.  CUPED-adjusted mean of treated sellers in
high-saturation markets (p >= 0.7) minus control sellers in low-saturation markets (p <= 0.3):
approximates the all-treated vs none-treated contrast without modelling p."""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    y, x = units["outcome"].to_numpy(float), units["pre_outcome"].to_numpy(float)
    ya = y - np.cov(y, x)[0, 1] / np.var(x, ddof=1) * (x - x.mean())
    p, t = units["p_market"].to_numpy(), units["treated"].to_numpy() == 1
    return float(ya[(p >= 0.7) & t].mean() - ya[(p <= 0.3) & ~t].mean())
