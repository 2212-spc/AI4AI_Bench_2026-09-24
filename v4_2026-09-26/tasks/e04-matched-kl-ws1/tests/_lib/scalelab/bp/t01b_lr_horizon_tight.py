"""T01b lr-horizon, tight budget: the T01 world, notebook and questions, but the lab allows 15 runs instead
of 40 (a controlled budget ablation of T01).  A full N x D x lr factorial is no longer affordable: the
notebook's diagonal sweeps have to carry the N-dependence, and the agent's own runs have to be spent
off the diagonal where the horizon effect is visible.  Everything else is identical to T01 (same draw,
same notebook construction, same items, same rivals), so the score difference between T01 and T01b on
matched world seeds is the effect of the budget alone.
"""
import math
from .t01_lr_horizon import *                      # noqa: F401,F403  (same world, notebook, items, rivals)
from . import t01_lr_horizon as _base
from .. import world as W
from ..common import run_rows, c1_to_r

ID = "t01b-lr-horizon-tight"
WORLD_ID = _base.ID                                # same world, notebook and salt as the base
TITLE = _base.TITLE + " (tight budget)"
MAX_RUNS, TOTAL_FLOPS = 15, 7e19


def spec(p):
    s = _base.spec(p)
    s["caps"] = dict(s["caps"], total_flops=TOTAL_FLOPS, max_runs=MAX_RUNS)
    return s


def oracle_design(rows_nb):
    s, i = _base._team_fit(rows_nb)
    reqs = []; seed = 100
    for N, D in _base.ORACLE_POINTS:
        c = math.exp(i) * N ** (-s) * (D / (20 * N)) ** (-0.25)
        for k in (-1, 0, 1):
            reqs.append({"N": N, "D": D, "lr": min(5e-2, c * 2.0 ** k), "seed": seed}); seed += 1
    return reqs


def oracle(sess, rows_nb, ctx, rng, drop=None):
    """9 coarse off-diagonal runs, fit, then 6 runs at +-0.5 ln around each fitted optimum (15 runs)."""
    own = run_rows(sess, oracle_design(rows_nb))
    ph = _base.fit_rows(rows_nb + own, rng, fixed=drop, n_starts=6)
    seed = 200; reqs = []
    for N, D in _base.ORACLE_POINTS:
        e = float(W.eta_star(W.full(ph), N, D))
        for u in (-0.5, 0.5):
            reqs.append({"N": N, "D": D, "lr": float(min(5e-2, e * math.exp(u))), "seed": seed}); seed += 1
    own += run_rows(sess, reqs)
    ph = _base.fit_rows(rows_nb + own, rng, fixed=drop, init_from=c1_to_r(ph))
    return _base.answers_from(ph, ctx), ph
