"""Decoy: market-level regression that is linear in p.  Pre-normalised market mean q = ybar/xbar,
WLS on [1, p] (x log-size interaction), weights n*xbar, GTE = weighted slope.  Right idea, but a
straight line cannot represent curvature of the market response, so the answer depends on where
the design put its markets."""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    g = units.groupby("market")
    d = pd.DataFrame(dict(n=g.size(), p=g["p_market"].first(), y=g["outcome"].mean(), x=g["pre_outcome"].mean()))
    q = (d.y / d.x).to_numpy(); w = (d.n * d.x).to_numpy(float)
    z = np.log(d.n.to_numpy(float)); z = z - np.sum(w * z) / w.sum()
    p = d.p.to_numpy(float)
    X = np.column_stack([np.ones_like(p), p, z, p * z]); sw = np.sqrt(w)
    b = np.linalg.lstsq(X * sw[:, None], q * sw, rcond=None)[0]
    eff = b[1] + b[3] * z
    return float(np.sum(d.n * d.x * eff) / d.n.sum())
