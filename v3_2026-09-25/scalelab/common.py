"""Shared helpers for blueprints: world-parameter draws, textbook priors, notebook writing, oracle fit
boxes, and the Blueprint protocol.

A blueprint module defines:
  ID, TITLE, CARDS, OBSTACLES, CLAIM (one line: the ability this task tests)
  draw(rng) -> params                         world parameters (hidden)
  spec(params) -> lab spec                    knobs, fixed values, caps, metrics
  known_unknowns(params) -> list of dicts     {name, param, range, text}: documented, zero-footprint
  notebook(params, salt, rng) -> (rows, notes_md)
  items(params) -> list of items             keys computed from the noise-free world
  oracle(session, rows, rng, H) -> answers   library-aware reference solver (H = hint level used)
  rivals(params, rows, rng) -> {name: answers or callable(session)}
"""
import math
import numpy as np
from .cards import CARDS
from . import world as W
from . import fit as F


def loguni(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def draw_card(rng, card, overrides=None, log_keys=("A", "Bc", "eta0", "bc0", "h0", "w0", "Q", "tau0", "km", "Rs")):
    out = {}
    for k, (tb, rg) in CARDS[card]["params"].items():
        if rg is None:
            continue
        lo, hi = (overrides or {}).get(k, rg)
        out[k] = loguni(rng, lo, hi) if (k in log_keys and lo > 0) else float(rng.uniform(lo, hi))
    if card == "C1":             # plausibility coupling (see cards.py): reducible loss at the reference scale
        out["A"] = float(rng.uniform(0.2, 0.6)) * 1e9 ** out["alpha"]
        out["Bc"] = float(rng.uniform(0.2, 0.7)) * 2e10 ** out["beta"]
    return out


def c1_from_r(q):
    """Fit-space reparametrisation of C1: rN = A*1e9^-alpha, rD = Bc*2e10^-beta (reducible loss at the
    reference scale).  Decorrelates the amplitude from the exponent, which the raw (A, alpha) form does not."""
    q = dict(q)
    if "rN" in q:
        q["A"] = q.pop("rN") * 1e9 ** q["alpha"]
    if "rD" in q:
        q["Bc"] = q.pop("rD") * 2e10 ** q["beta"]
    return q


def c1_to_r(q):
    q = dict(q)
    if "A" in q:
        q["rN"] = q.pop("A") * 1e9 ** (-q["alpha"])
    if "Bc" in q:
        q["rD"] = q.pop("Bc") * 2e10 ** (-q["beta"])
    return q


R_BOX = {"rN": (0.08, 1.0), "rD": (0.08, 1.2)}


def textbook(cards):
    out = {}
    for c in cards:
        for k, (tb, rg) in CARDS[c]["params"].items():
            if tb is not None:
                out[k] = tb
    return out


def box_for(keys, widen=1.0, overrides=None):
    """Fit box from card world ranges (optionally widened).  Positive scale params are log-bounded."""
    spec = {}
    for c in CARDS.values():
        for k, (tb, rg) in c["params"].items():
            if k in keys and rg is not None:
                lo, hi = (overrides or {}).get(k, rg)
                lg = lo > 0 and k in ("A", "Bc", "eta0", "bc0", "h0", "w0", "Q", "tau0", "km", "Rs", "sigma0", "k_lo", "k_hi", "k_tau", "mu", "ca")
                if lg:
                    spec[k] = (lo / widen, hi * widen, True)
                else:
                    span = (hi - lo) * (widen - 1) / 2
                    spec[k] = (lo - span, hi + span, False)
    for k in keys:
        if k not in spec and overrides and k in overrides:
            lo, hi = overrides[k]; spec[k] = (lo, hi, lo > 0)
    return F.Box({k: spec[k] for k in keys})


def run_rows(sess, reqs):
    """Execute a list of requests through the session; returns result rows (skips LabError)."""
    from .lab import LabError
    out = []
    for r in reqs:
        try:
            out.append(sess.run(r))
        except LabError:
            pass
    return out


def rows_xy(rows, keys):
    """Flatten ok rows into (config arrays dict, y array)."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("loss") is not None]
    cfg = {k: np.array([float(r["config"][k]) for r in ok]) for k in keys}
    y = np.array([r["loss"] for r in ok])
    return cfg, y, ok


def fmt(x, sig=4):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    return ("%." + str(sig) + "g") % x


def rows_to_csv(rows, cols):
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(fmt(r.get(c)) if not isinstance(r.get(c), str) else r.get(c) for c in cols))
    return "\n".join(lines) + "\n"
