"""Decoy (generic hedge): average the three published laws' predictions ('ensembling is robust')."""
import importlib.util, os
import numpy as np
_B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")


def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(_B, n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def fit_predict(train, queries):
    preds = [_load(n).fit_predict(train, queries) for n in ("lr_area", "momentum_law")]
    return {k: float(np.median([p[k] for p in preds])) for k in queries}
