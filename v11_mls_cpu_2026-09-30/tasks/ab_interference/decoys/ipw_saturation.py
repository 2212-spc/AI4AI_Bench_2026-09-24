"""Decoy: inverse-propensity weighting by the design saturation.  A seller's treatment propensity is
p_market, so weight treated by 1/p and controls by 1/(1-p) (Hajek), with CUPED adjustment."""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    y, x = units["outcome"].to_numpy(float), units["pre_outcome"].to_numpy(float)
    ya = y - np.cov(y, x)[0, 1] / np.var(x, ddof=1) * (x - x.mean())
    p, t = units["p_market"].to_numpy(float), units["treated"].to_numpy() == 1
    w1, w0 = 1 / p[t], 1 / (1 - p[~t])
    return float(np.sum(w1 * ya[t]) / w1.sum() - np.sum(w0 * ya[~t]) / w0.sum())
