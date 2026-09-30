"""Decoy: market-level quadratic regression of the raw market mean outcome on p (weights = market
size), no pre-period adjustment.  GTE = fitted mean at p=1 minus at p=0."""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    g = units.groupby("market")
    d = pd.DataFrame(dict(n=g.size(), p=g["p_market"].first(), y=g["outcome"].mean()))
    c = np.polyfit(d.p.to_numpy(float), d.y.to_numpy(), 2, w=np.sqrt(d.n.to_numpy(float)))
    return float(np.polyval(c, 1.0) - np.polyval(c, 0.0))
