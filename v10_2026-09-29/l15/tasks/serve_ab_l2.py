"""SERVE-AB level L2 = L1 + two heterogeneity operators on the SAME task text (minimal pair):
  H7a  hour-block heterogeneity: B's slowdown kappa and quality lift differ between business hours and the
       rest of the day (pooling the A/B across hours mis-states both the peak cost and the peak benefit);
  H7b  weekly day types: weekend load is 16-28% lower and day-to-day noise is larger (sd 5%), so the value of
       a fixed hourly schedule is a mixture over day types that sit on opposite sides of the saturation kink
       (Jensen: plugging in the average day is biased exactly where the decision is made).
Everything else, including every principle of L1, is inherited."""
import numpy as np
from .serve_ab import *            # noqa: F401,F403
from . import serve_ab as _b

LEVEL = "L2"


def sample_params(seed):
    return _b._sample(seed, LEVEL)


def pool_gate(seeds):
    return _b._pool([sample_params(s) for s in seeds])


class World(_b.World):
    NAME = "serve_ab_l2"


def strat_one_day_type(sess, art_dir, rng):
    """P5 ABLATION: the full L1 reference (drill, Erlang-A, autoscaler) but the load model is one average day
    (all 14 history days pooled, day-to-day sd from their spread) and kappa / lift pooled over hours."""
    ab = _b._guard_ab(sess)
    ph = _b._fit_history(sess)
    hist = _b._hist(sess)
    days = sorted(set(r["day"] for r in hist))
    tot = [sum(r["requests"] for r in hist if r["day"] == d) for d in days]
    ph["w_we"] = 1.0
    ph["sd_day"] = float(np.std(np.array(tot) / np.mean(tot), ddof=1))
    ph["lam"] = [float(np.mean([r["requests"] for r in hist if r["hour"] == h])) / 3600.0 for h in range(24)]
    a_night = np.median([ph["lam"][h] * ph["sA"] for h in range(_b.DRILL_LO, _b.DRILL_HI + 1)])
    k = max(1, int(round(a_night / 1.3)))
    dr = _b._test(sess, fraction=0.0, hours="%d-%d" % (_b.DRILL_LO, _b.DRILL_HI), days=2, max_replicas=k)
    kh, dh = _b._fit_arms(ab, ph)
    lam = np.array(ph["lam"])
    kp = float(np.sum(lam * np.array(kh)) / lam.sum()); dp = float(np.sum(lam * np.array(dh)) / lam.sum())
    ph.update({"kap_h": [kp] * 24, "dq_h": [dp] * 24, "theta": _b._theta_from(dr)})
    xs, D = _b.optimum(ph)
    _b._write(art_dir, xs, _b._iv(_b.daily_delta(ph, np.ones(24)), _b.W1_CAP), _b._iv(D, _b.W2_CAP))


STRATEGIES = dict(_b.STRATEGIES)
STRATEGIES["one_day_type"] = (strat_one_day_type, "fail")
PRINCIPLES = dict(_b.PRINCIPLES)
PRINCIPLES["P5_mixture_not_average_day"] = ("one_day_type", ["R2_full_launch"])
