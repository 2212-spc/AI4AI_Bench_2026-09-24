"""Baseline: difference in means, treated sellers vs control sellers (pooled over all markets)."""
import pandas as pd


def estimate(units: pd.DataFrame) -> float:
    t = units["treated"] == 1
    return float(units.loc[t, "outcome"].mean() - units.loc[~t, "outcome"].mean())
