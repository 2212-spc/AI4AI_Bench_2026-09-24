"""Mechanism cards for the goal-3 AV compiler.

A card is a 1-D AI4AI response family with sampled world constants.  It knows nothing about items; the
assumption-violation (AV) templates in templates.py turn (card, world, design) into an item.

Every card declares
  prior        {key: (lo, hi)}  uniform prior of each world constant (used for sampling AND for the driver
               analysis, which perturbs each key by 10% of its prior range)
  stat_keys    keys that are *statistical* (noise level, population spread) rather than *mechanism* constants;
               an item whose shortcut bias is driven by a stat key is card-agnostic (see README section 4)
  y(x, th, ctx)   noiseless response at instrument value x; ctx=0 is the notebook's context, ctx=1 production
  sig(x, th)      per-run seed noise sd
  unit_y(x, th, q, q2)   response of a population unit with latent offsets q (level) and q2 (slope)
  censor_dir   +1: units above a threshold are reported only as "> c"; -1: below, "< c"
  floor_kind   'probe' (a separate instrument measures the floor), 'inv' (floor is the intercept of a linear
               relation in a transformed coordinate), or None (no physical floor on this axis -> M6 is N/A)
"""
import math

import numpy as np

_lg = np.vectorize(math.lgamma)


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


class Card:
    key = ""
    title = ""
    axis = ""
    lo, hi = 0.0, 1.0
    better = "min"
    censor_dir = +1
    censor_story = ""
    ctx_story = ""
    floor_kind = None
    floor_story = ""
    nominal_floor = None
    prior = {}
    stat_keys = ("s0", "tau", "tau2", "tau_r")
    recipe_rel = False          # M3: recipe effects additive (losses, pass rates) or multiplicative (step counts)

    def sample(self, rng):
        return {k: _u(rng, lo, hi) for k, (lo, hi) in self.prior.items()}

    def y(self, x, th, ctx=0):
        raise NotImplementedError

    def sig(self, x, th):
        raise NotImplementedError

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    def unit_y(self, x, th, q, q2):
        return self.y(x, th) + th["tau"] * q + th["tau2"] * q2 * (x - self.mid)

    def floor(self, th):
        return None


class ComputeScaling(Card):
    key = "K1_compute"
    title = "loss vs log10 compute (saturating power law)"
    axis = "x = log10(C / C0)"
    lo, hi = 0.0, 3.0
    ctx_story = ("the notebook's sweep used fresh data; production is data-constrained and starts repeating "
                 "tokens past x_rep, adding a softplus penalty of D nats per decade")
    censor_story = "eval domains whose loss exceeds the watchdog clip c are killed and logged only as '> c'"
    floor_kind = "probe"
    floor_story = "the synthetic data generator's entropy can be probed directly (noisy probe run)"
    nominal_floor = 1.69            # the published Chinchilla E, the constant an agent may recite
    prior = dict(E=(1.70, 2.10), A=(1.0, 2.0), al=(0.25, 0.45), D=(0.03, 0.08), xr=(1.0, 2.0),
                 s0=(0.006, 0.012), tau=(0.06, 0.25), tau2=(0.0, 0.05), tau_r=(0.002, 0.02))

    def y(self, x, th, ctx=0):
        x = np.asarray(x, float)
        v = th["E"] + th["A"] * 10 ** (-th["al"] * x)
        if ctx:
            w = 0.25
            v = v + th["D"] * w * np.log1p(np.exp((x - th["xr"]) / w))
        return v

    def sig(self, x, th):
        return th["s0"] * 10 ** (-0.15 * np.asarray(x, float))

    def floor(self, th):
        return th["E"]


class LRBasin(Card):
    key = "K2_lrbasin"
    title = "final loss vs log2 learning rate (asymmetric basin)"
    axis = "x = log2(lr / lr_ref)"
    lo, hi = -3.0, 3.0
    ctx_story = ("the notebook swept LR at batch b0; production uses 4*b0, which moves the optimum by "
                 "beta*log2(4) (beta=0.5 is the sqrt rule, 1.0 the linear rule; the world's beta is unknown)")
    censor_story = "runs whose loss exceeds c are killed by the divergence watchdog and logged only as '> c'"
    floor_kind = None
    prior = dict(ys=(2.4, 2.8), a=(0.01, 0.03), b=(0.005, 0.02), xo=(-1.0, 0.5), beta=(0.25, 1.0),
                 dy=(0.0, 0.02), s0=(0.004, 0.010), tau=(0.05, 0.20), tau2=(0.0, 0.03), tau_r=(0.002, 0.02))

    def y(self, x, th, ctx=0):
        x = np.asarray(x, float)
        xo = th["xo"] + (2.0 * th["beta"] if ctx else 0.0)
        d = x - xo
        return th["ys"] - (th["dy"] if ctx else 0.0) + th["a"] * d ** 2 + th["b"] * np.maximum(0.0, d) ** 3

    def sig(self, x, th):
        return th["s0"] * (1.0 + 0.5 * np.maximum(0.0, np.asarray(x, float) - th["xo"]))


class DataRepetition(Card):
    key = "K3_repeat"
    title = "loss vs log2 epochs over a fixed unique-token pool (Muennighoff-style decay of repeated data)"
    axis = "x = log2(epochs)"
    lo, hi = 0.0, 4.0
    ctx_story = ("the notebook's pool is web text with repetition half-life R*; production data is code, whose "
                 "R*_1 is a different, unknown constant")
    censor_story = "domains whose loss exceeds the clip c are killed and logged only as '> c'"
    floor_kind = "probe"
    floor_story = "the generator entropy of the synthetic pool can be probed directly (noisy probe run)"
    nominal_floor = 1.69
    prior = dict(E=(1.8, 2.2), Bc=(0.4, 0.8), be=(0.25, 0.40), Rs=(8.0, 30.0), Rs1=(2.0, 8.0),
                 s0=(0.004, 0.010), tau=(0.05, 0.20), tau2=(0.0, 0.03), tau_r=(0.002, 0.02))

    def y(self, x, th, ctx=0):
        r = 2.0 ** np.asarray(x, float)
        R = th["Rs1"] if ctx else th["Rs"]
        deff = 1.0 + R * (1.0 - np.exp(-(r - 1.0) / R))
        return th["E"] + th["Bc"] * deff ** (-th["be"])

    def sig(self, x, th):
        return th["s0"] * np.ones_like(np.asarray(x, float))

    def floor(self, th):
        return th["E"]


class PassAtK(Card):
    key = "K4_passk"
    title = "pass@k vs log2 k under heterogeneous per-problem success rates (Beta mixture)"
    axis = "x = log2(k)"
    lo, hi = 0.0, 7.0
    better = "max"
    ctx_story = ("the notebook measured a benchmark with concentration s (same mean pass@1); production is a new "
                 "benchmark with the same pass@1 but concentration s1 - the pass@k curve's bend is set by s")
    censor_dir = -1
    censor_story = "problem categories with zero solves in the sample budget are logged only as '< c'"
    floor_kind = None
    prior = dict(pbar=(0.08, 0.35), s=(0.4, 3.0), s1=(0.4, 3.0), tau=(0.3, 1.2), tau2=(0.0, 0.15),
                 tau_r=(0.005, 0.03))
    stat_keys = ("tau", "tau2", "tau_r")
    P_RUN = 400        # problems per eval run

    @staticmethod
    def _pk(k, pbar, s):
        a, b = pbar * s, (1.0 - pbar) * s
        return 1.0 - np.exp(_lg(b + k) + _lg(a + b) - _lg(b) - _lg(a + b + k))

    def y(self, x, th, ctx=0):
        k = 2.0 ** np.asarray(x, float)
        return self._pk(k, th["pbar"], th["s1"] if ctx else th["s"])

    def sig(self, x, th):
        v = self.y(x, th)
        return np.sqrt(v * (1 - v) / self.P_RUN)

    def unit_y(self, x, th, q, q2):
        lg = math.log(th["pbar"] / (1 - th["pbar"])) + th["tau"] * q + th["tau2"] * q2 * (x - self.mid)
        pb = 1.0 / (1.0 + np.exp(-lg))
        return self._pk(2.0 ** x, pb, th["s"])


class CriticalBatch(Card):
    key = "K5_critbatch"
    title = "steps-to-target vs log2 batch (S = S_min (1 + B_crit / B))"
    axis = "x = log2(B / B0)"
    lo, hi = 0.0, 7.0
    ctx_story = ("the notebook measured steps to an early loss target; production asks for a lower target, where "
                 "B_crit is larger by an unknown factor 2**gam")
    censor_story = "runs that do not reach the target inside the step cap are logged only as '> cap'"
    floor_kind = "inv"
    floor_story = "S_min is the intercept of S against 1/B, which is exactly linear on this card"
    prior = dict(Smin=(2.0, 6.0), bc=(1.5, 4.0), gam=(0.5, 2.0), srel=(0.015, 0.04), tau=(0.10, 0.30),
                 tau2=(0.0, 0.05), tau_r=(0.005, 0.03))
    stat_keys = ("srel", "tau", "tau2", "tau_r")
    recipe_rel = True

    def y(self, x, th, ctx=0):
        bc = th["bc"] + (th["gam"] if ctx else 0.0)
        return th["Smin"] * (1.0 + 2.0 ** (bc - np.asarray(x, float)))

    def sig(self, x, th):
        return th["srel"] * self.y(x, th)

    def unit_y(self, x, th, q, q2):
        return self.y(x, th) * np.exp(th["tau"] * q + th["tau2"] * q2 * (x - self.mid))

    def floor(self, th):
        return th["Smin"]


CARDS = [ComputeScaling(), LRBasin(), DataRepetition(), PassAtK(), CriticalBatch()]


def Y(card, x, th, ctx=0):
    return float(np.asarray(card.y(np.array([float(x)]), th, ctx))[0])


def S(card, x, th):
    return float(np.asarray(card.sig(np.array([float(x)]), th))[0])
