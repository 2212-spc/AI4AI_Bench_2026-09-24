"""Reference: exploit the market-level saturation design.

Units in a market interact (they share buyers), so a within-market treated-vs-control contrast
measures "treated next to controls", not "everyone treated vs nobody treated".  But the treated
fraction p_m was randomised across markets, so the market-level mean outcome as a function of p,
mu_m(p), is identified, and GTE = per-seller average of mu_m(1) - mu_m(0).

1. Collapse to markets: n_m, p_m, ybar_m (post mean), xbar_m (pre-period mean).
2. Outcomes scale with market demand, so model the ratio q_m = ybar_m / xbar_m (pre-period
   normalisation removes the large between-market level differences) as a smooth low-degree
   (quadratic) function of p; p only covers [0.1, 0.9], so a quadratic is the least-flexible form
   that captures curvature of the market response without wild extrapolation.
3. Let the curve vary with an observed market characteristic (log market size, centred) -
   Lin (2013)-style interacted adjustment - because effects may be heterogeneous across markets.
4. Weighted least squares with weights n_m * xbar_m: each market counts in proportion to its
   share of total seller outcome, which is exactly how it enters the per-seller GTE.
5. GTE = sum_m n_m xbar_m [q_m(1) - q_m(0)] / sum_m n_m.
No term assumes interference exists: with none, q(p) is just linear in p and the fit returns the
direct effect.
"""
import numpy as np
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    g = units.groupby("market")
    d = pd.DataFrame(dict(n=g.size(), p=g["p_market"].first(), y=g["outcome"].mean(), x=g["pre_outcome"].mean()))
    q = (d.y / d.x).to_numpy()
    w = (d.n * d.x).to_numpy(float)
    z = np.log(d.n.to_numpy(float)); z = z - np.sum(w * z) / w.sum()

    def design(p):
        P = np.vander(np.broadcast_to(p, z.shape).astype(float), 3, increasing=True)
        return np.hstack([P, P * z[:, None]])

    sw = np.sqrt(w)
    b = np.linalg.lstsq(design(d.p.to_numpy()) * sw[:, None], q * sw, rcond=None)[0]
    eff = (design(1.0) - design(0.0)) @ b                     # per-market relative launch effect
    return float(np.sum(d.n * d.x * eff) / d.n.sum())
