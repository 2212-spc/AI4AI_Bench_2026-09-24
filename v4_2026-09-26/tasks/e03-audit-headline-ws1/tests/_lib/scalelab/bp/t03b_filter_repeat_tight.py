"""T03b filter-repeat, tight budget: the T03 world, notebook and questions with 12 lab runs instead of 40
(a controlled budget ablation of T03).  Separating the quality and repetition constants now has to reuse
the notebook: its masked ablation is one equation in (mu, nu, Rs); a short fresh-data filter sweep and a
two-point repetition sweep supply the others.  Same draw, notebook, items and rivals as T03.
"""
from .t03_filter_repeat import *                   # noqa: F401,F403
from . import t03_filter_repeat as _base
from ..common import run_rows, c1_to_r

ID = "t03b-filter-repeat-tight"
WORLD_ID = _base.ID                                # same world, notebook and salt as the base
TITLE = _base.TITLE + " (tight budget)"
MAX_RUNS, TOTAL_FLOPS = 12, 3e19


def spec(p):
    s = _base.spec(p)
    s["caps"] = dict(s["caps"], total_flops=TOTAL_FLOPS, max_runs=MAX_RUNS)
    return s


def oracle_design(rows_nb):
    reqs = []; s = 100
    for q in (0.0, 0.6, 0.85):                                # fresh-data filter sweep
        for k in range(2):
            reqs.append({"N": 5e7, "D": 5e9, "q": q, "seed": s}); s += 1
    for ep in (4.0, 25.0):                                    # repetition sweep (q = 0, small subsample)
        for k in range(2):
            reqs.append({"N": 5e7, "D": 5e9, "q": 0.0, "sub": 5e9 / ep, "seed": s}); s += 1
    for N, D in ((2e7, 4e10), (3e8, 3e9)):                    # off-diagonal C1 points
        reqs.append({"N": N, "D": D, "q": 0.0, "seed": s}); s += 1
    return reqs


def oracle(sess, rows_nb, ctx, rng, drop=None):
    own = run_rows(sess, oracle_design(rows_nb))
    keys = [k for k in _base.FIT_KEYS if k not in (drop or {})]
    ph = _base.fit_rows(rows_nb + own, rng, fixed=drop, keys=keys)
    ph = _base.fit_rows(rows_nb + own, rng, fixed=drop, keys=keys, init_from=c1_to_r(ph), n_starts=4)
    return _base.answers_from(ph, ctx), ph
