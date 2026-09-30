"""Decoy: market-level quadratic regression (pre-normalised q = ybar/xbar on [1, p, p^2]) with every
market weighted equally - 'each market is one randomised unit'.  GTE = baseline level * (q(1)-q(0))."""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    g = units.groupby("market")
    d = pd.DataFrame(dict(n=g.size(), p=g["p_market"].first(), y=g["outcome"].mean(), x=g["pre_outcome"].mean()))
    c = np.polyfit(d.p.to_numpy(float), (d.y / d.x).to_numpy(), 2)
    rel = np.polyval(c, 1.0) - np.polyval(c, 0.0)
    return float(rel * units["pre_outcome"].mean())
