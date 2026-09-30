"""R-CRIT: r_retune, hardened against the protocol claude-fable-5-1 used to pass it.

Same situation, same text, same ops, same deliverable as `r_retune` (re-tune lr and wd after a forced change of
batch and/or token budget).  What changes is ONE hidden mechanism, chosen from the frontier trajectories:

  r_retune round 1: gpt-6 0/6 (all far from threshold: recited rules, one-sided checks); claude-fable-5-1 passed
  2 of 3 graded runs.  Reading the passes: Fable measured the effective decay near the target length (the P3-P6
  traps did not bind it), and took the LEARNING RATE from short runs - anchor-calibrated proxy DiD, or a
  power-law trend of the lr optimum over run length fitted on runs <= 1/4 of the target and extrapolated.  In
  r_retune both of those are exact, because the lr optimum there is linear in (log batch, log tokens).  So the
  one premise every Fable run shared - "the lr optimum at full length can be read off short runs" - was a
  premise the world happened to satisfy.  r_crit is the world in which it does not hold, for a documented
  physical reason.

THE MECHANISM: A CRITICAL BATCH THAT GROWS WITH TRAINING.  How far a larger batch moves the optimal lr depends
on the batch relative to the gradient noise scale (McCandlish et al. 2018: eps_opt(B) = eps_max / (1 +
B_noise / B), i.e. lr ~ B below the critical batch, saturating above it), and the critical batch grows as
training proceeds (the noise scale grows as the loss falls; Zhang et al. 2024 find the critical batch scales
with the data size).  In log2 units (B = log2 batch/B0, T = log2 tokens/D0):

    mu(B,T) = u0 - alpha T + rho [ sm(B, Bc(T)) - sm(0, c0) ]          lr optimum
    Bc(T)   = c0 + kappa T                                             log2 critical batch, grows with tokens
    sm(x,y) = -s log2(2^(-x/s) + 2^(-y/s))                             smooth min; s = 1 is exactly McCandlish

At the target length the new batch sits below the critical batch (Bc(T_new) = B_new + cn, cn >= 0), so the full
batch gain rho*B_new applies.  At 1/16 of the target length Bc is kappa*4 ~ 2-3.6 octaves lower: the new batch is
far above critical there and the lr gain is saturated.  Consequences (all exact, all checked per instance):

  P7  THE ANCHOR-CALIBRATED PROXY DIFFERENCE IS BIASED.  lr shift measured at matched proxy fraction (the
      r_retune oracle) = saturated gain at the proxy, not the gain at full length; >= K_TRAP * EPS off.
  P8  THE HORIZON TREND IS CURVED.  The lr optimum is not log-linear in run length once Bc crosses the batch;
      a trend fitted on runs <= 1/4 of the target (new setup, with the anchor where it is on the same batch)
      and extrapolated, and the "slope from the current setup, level from short new-setup runs" variant, are
      both >= K_TRAP * EPS off.
  and every r_retune principle still holds: P3 wd must move, P4 tau exponent is measured, P5 tuning at the
  proxy is biased, P6 lr and lr*wd are the knobs, recitations are >= K_DECOY * EPS off.

The agent's own check - one full-length run of its recipe, or a comparison against the carried-over recipe -
passes on the P7/P8 recipes: both beat the carried-over recipe, and one run cannot show the distance to the
optimum.  A full-length lr sweep (3 runs) at the new setup shows it immediately; the budget allows it.

THE REFERENCE PROTOCOL (oracle; public facts only): (1) an anchor-calibrated DiD at 1/8 of the target length
(CENTRE_DEPTH = 3; 1 seed per point; each of its two 3-point lr sweeps WALKS one spacing toward the better edge,
up to MAX_WALK extra points, until the minimum is bracketed), used only to CENTRE a later sweep - the instance
condition requires the walk to need <= MAX_WALK - 1 extra points and the whole protocol to fit BUDGET; (2) the
r_retune effective-decay sweep at one horizon where wd visibly matters, tau exponent read off the anchor;
(3) a 3-point full-length lr sweep at the new setup holding lr*wd at the measured optimum, started from (1):
runs at (1) -/+ half an octave, then one more a full octave beyond the better of the two (a bracket of spacing
1 octave that covers a centring error of up to ~1.4 octaves); the vertex is the lr, and the fitted parabola's value at the vertex is the loss (the
shipped recipe lies on the swept line), interval from its own Lagrange weights.

DISCOVERABILITY (D2): the proxy DiD at 1/4 of the target differs from the one at 1/16 by >= 4 sd of their
difference - an agent that checks its lr correction at a second horizon sees that it is not horizon-invariant.
Nothing in the text says so.  D1 (wd curvature visible at the oracle's wd horizon) and T1 (wd nearly invisible
at the cheapest proxy) are kept from r_retune.

Ground truth: E[loss] = F(B,T) + A (u - mu)^2 + C(T) (w - nu)^2 exactly as in r_retune, with the mu above, so
the optimum, every decoy's gap and the oracle's error model are closed form.
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for
from . import r_retune as R
from .r_retune import (FULL_COST, MAX_SEEDS, WIDTH_CAP, K_DECOY, PROXY_DEPTH, TOKENS_MIN_FRAC,
                       RECITED_RHO, RECITED_WD, _sig3, nu, curv_w, floor_loss, _run_cost,
                       _vertex_var, _batches, _parse_run, _cost_run, _vertex, _write, _load, _save, _sweep_w)

BUDGET = 384.0
EPS_LOSS = 0.0006          # R1 (r_retune: 0.0008) - see the recitation-coverage note at VARIANTS
K_TRAP = 4.0               # P7/P8 recipes must be >= K_TRAP * EPS_LOSS off (wrong answers land far away)
SHARP = 0.6                # smooth-min sharpness of the critical-batch knee (octaves)
H_FULL = 1.0               # half-width (octaves) of the oracle's full-length lr sweep
H_PROXY = 1.25
ALO, AHI = 0.05, 0.70
RLO, RHI = 0.20, 1.00
RECITED_ALPHA = (0.0, 1.0 / 3, 0.5)     # lr ~ tokens^-a


# Forced changes with a lever arm of >= 3 octaves on the lr (a 2-octave batch change bounds the lr shift to
# [0, 2] octaves with the sqrt and linear rules at 1 and 2: every gap between recitations is then narrow enough
# for a hedged guess "between sqrt and linear" to cover most instances - measured, see DESIGN notes).
VARIANTS = {"batch8x": (3, 0), "batch16x": (4, 0), "tokens8x": (0, 3), "tokens16x": (0, 4),
            "batch2x_tokens8x": (1, 3)}
_VORDER = ["batch8x", "tokens8x", "batch16x", "tokens16x", "batch2x_tokens8x"]


def _base(g):
    variant = _VORDER[int(g.integers(len(_VORDER)))]
    Bn, Tn = VARIANTS[variant]
    B0 = int(g.choice([128, 256] if Bn >= 4 else ([256, 512] if Bn >= 2 else [256, 512, 1024])))
    D0 = int(g.choice([32, 48, 64] if Bn >= 3 else ([12, 16, 24, 32] if Tn >= 4 else [12, 16, 24, 32, 48])))
    lr0 = _sig3(float(np.exp(g.uniform(math.log(1.5e-4), math.log(1.2e-3)))))
    wd0 = _sig3(float(np.exp(g.uniform(math.log(0.02), math.log(0.2)))))
    u0, v0 = math.log2(lr0), math.log2(wd0)
    return {"variant": variant, "B_new": Bn, "T_new": Tn, "B0": B0, "D0": D0, "lr0": lr0, "wd0": wd0,
            "u0": u0, "v0": v0, "w0": u0 + v0,
            "E": round(float(g.uniform(2.25, 2.75)), 4), "K": round(float(g.uniform(0.30, 0.55)), 4),
            "a": round(float(g.uniform(0.20, 0.35)), 4), "b": round(float(g.uniform(0.004, 0.012)), 5)}


# ------------------------------------------------------------------ hidden mechanics
def sm(x, y, s=SHARP):
    return -s * math.log2(2.0 ** (-x / s) + 2.0 ** (-y / s))


def bcrit(p, T):
    return p["c0"] + p["kappa"] * T


def gain(p, B, T):
    """rho * log2(B / (B + B_noise(T))) in log2 units (s = 1: exactly McCandlish et al. 2018)."""
    return p["rho"] * sm(0.0, B - bcrit(p, T))


def mu(p, B, T):
    return p["u0"] - p["alpha"] * T + gain(p, B, T) - gain(p, 0.0, 0.0)


def exp_loss(p, B, T, u, v):
    w = u + v
    return floor_loss(p, B, T) + p["A"] * (u - mu(p, B, T)) ** 2 + curv_w(p, T) * (w - nu(p, B, T)) ** 2


def penalty(p, u, v):
    Bn, Tn = p["B_new"], p["T_new"]
    return p["A"] * (u - mu(p, Bn, Tn)) ** 2 + p["C1"] * (u + v - nu(p, Bn, Tn)) ** 2


def optimum(p):
    Bn, Tn = p["B_new"], p["T_new"]
    u, w = mu(p, Bn, Tn), nu(p, Bn, Tn)
    return u, w - u


# ------------------------------------------------------------------ shortcut lr estimates (exact, noise-free)
def lr_did(p, k=PROXY_DEPTH):
    """Anchor-calibrated proxy difference at matched fraction 2^-k of each setup (the r_retune oracle)."""
    return p["u0"] + mu(p, p["B_new"], p["T_new"] - k) - mu(p, 0, -k)


def _trend_pts(p):
    Bn, Tn = p["B_new"], p["T_new"]
    Tmin = math.log2(TOKENS_MIN_FRAC)
    Ts = [T for T in (Tn - 5, Tn - 4, Tn - 3, Tn - 2) if T >= Tmin - 1e-9]
    return Ts


def _linfit_at(xs, ys, x):
    a, b = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    return float(a * x + b)


def lr_trend(p):
    """Log-linear trend of the new setup's lr optimum over runs <= 1/4 of the target, extrapolated to the
    target; the anchor (lr0 optimal at the current length) joins the fit when it is on the same batch."""
    Bn, Tn = p["B_new"], p["T_new"]
    xs = _trend_pts(p)
    ys = [mu(p, Bn, T) for T in xs]
    if Bn == 0:
        xs, ys = xs + [0.0], ys + [p["u0"]]
    return _linfit_at(xs, ys, Tn)


def lr_mixed(p):
    """Horizon slope from the current setup (-6..-4 plus the anchor), level from the new setup at 1/8, 1/4."""
    Bn, Tn = p["B_new"], p["T_new"]
    xs = [-6.0, -5.0, -4.0, 0.0]
    ys = [mu(p, 0, T) for T in xs]
    a = float(np.polyfit(xs, ys, 1)[0])
    lv = [mu(p, Bn, T) - a * T for T in (Tn - 3, Tn - 2)]
    return float(np.mean(lv) + a * Tn)


def lr_near(p):
    """Two-point extrapolation from 1/4 and 1/2 of the target: a legitimate near-target route (info only)."""
    Bn, Tn = p["B_new"], p["T_new"]
    y1, y2 = mu(p, Bn, Tn - 2), mu(p, Bn, Tn - 1)
    return 2 * y2 - y1


# ------------------------------------------------------------------ public reference design (no hidden info)
CENTRE_DEPTH = 3
MAX_WALK = 4              # extra points a centring sweep may add while walking toward a bracketed minimum


def did_sweeps(p, k, seeds):
    """The two short-horizon lr sweeps of an anchor-calibrated DiD at depth k (1/2^k of each setup).  Starting
    centres use only public facts: short runs like a higher lr, a bigger batch likes a higher lr."""
    Bn, Tn = p["B_new"], p["T_new"]
    cu_old = p["u0"] + 0.4 * k
    cu_new = cu_old + 0.8 * Bn - 0.4 * Tn
    return [dict(B=0, T=-k, c=cu_old, h=H_PROXY, seeds=seeds), dict(B=Bn, T=Tn - k, c=cu_new, h=H_PROXY, seeds=seeds)]


def walk_steps(dev, h):
    """Extra points a 3-point walking sweep started `dev` octaves from the vertex needs (noise-free)."""
    return max(0, int(math.ceil((abs(dev) - 0.5 * h) / h - 1e-12)))


def oracle_design(p):
    """The effective-decay sweep of r_retune; an anchor-calibrated DiD at 1/4 of each setup (one seed) used only
    to CENTRE the full-length lr bracket; the bracket itself."""
    d = R.oracle_design(p)
    Bn, Tn = p["B_new"], p["T_new"]
    d["lr"] = did_sweeps(p, CENTRE_DEPTH, 1)
    d["lr_full"] = dict(B=Bn, T=Tn, h=H_FULL, seeds=1)
    d["confirm_seeds"] = 0
    return d


def _lagr(x):
    """Lagrange weights of the 3-point parabola through -1, 0, +1 evaluated at x (units of h)."""
    return (x * (x - 1) / 2.0, 1.0 - x * x, x * (x + 1) / 2.0)


# ------------------------------------------------------------------ truth and instance conditions
def _recipes(p):
    Bn, Tn = p["B_new"], p["T_new"]
    u0, v0, w0 = p["u0"], p["v0"], p["w0"]
    us, vs = optimum(p)
    ws = us + vs
    out = {}
    lrs = [("rho%g" % r, u0 + r * Bn) for r in RECITED_RHO] + [("measured", us)]
    for lname, u in lrs:
        for wname, cc in RECITED_WD.items():
            if cc is None:
                out["lr_%s__%s" % (lname, wname)] = (u, v0)
            else:
                w = w0 + cc[0] * Bn - cc[1] * Tn
                out["lr_%s__%s" % (lname, wname)] = (u, w - u)
    # recited lr rules with the effective decay exactly right (the wd traps already avoided): batch exponent
    # 0 / 1/2 / 1 and token-horizon exponent 0 / 1/3 / 1/2
    for r in RECITED_RHO:
        for a in RECITED_ALPHA:
            u = u0 + r * Bn - a * Tn
            out["lr_rho%g_tok%.2g__w_star" % (r, a)] = (u, ws - u)
    Tp = Tn - PROXY_DEPTH
    up = mu(p, Bn, Tp)
    out["proxy_optimum"] = (up, nu(p, Bn, Tp) - up)
    for nm, u in (("did4", lr_did(p, 4)), ("trend", lr_trend(p)), ("mixed", lr_mixed(p))):
        out["lr_%s__w_star" % nm] = (u, ws - u)          # the P7/P8 recipes: lr shortcut, lr*wd exactly right
    return out


TRAP_KEYS = ("lr_did4__w_star", "lr_trend__w_star", "lr_mixed__w_star")


def _ofat_exact(p):
    us, vs = optimum(p)
    ws = us + vs
    A, C1, v0 = p["A"], p["C1"], p["v0"]
    return (A * us + C1 * (ws - v0)) / (A + C1), v0


def _cross_centre(p):
    return p["u0"] + 0.5 * p["B_new"] - 0.3 * p["T_new"], p["v0"]


def _cross_exact(p):
    """Full-length 5-point cross in (lr, wd) around the sqrt-rule lr and production wd, per-axis vertices."""
    us, vs = optimum(p)
    ws = us + vs
    cu, cv = _cross_centre(p)
    A, C1 = p["A"], p["C1"]
    u1 = (A * us + C1 * (ws - cv)) / (A + C1)
    u1 = min(cu + 2 * 0.7, max(cu - 2 * 0.7, u1))
    v1 = min(cv + 2 * 1.0, max(cv - 2 * 1.0, ws - cu))
    return u1, v1


def _sh_grid(p):
    return R._sh_grid(p)


def _proxy_grid(p):
    return R._proxy_grid(p)


def truth(p):
    us, vs = optimum(p)
    Bn, Tn = p["B_new"], p["T_new"]
    rec = {k: round(penalty(p, *uv), 6) for k, uv in _recipes(p).items()}
    uo, vo = _ofat_exact(p)
    un = lr_near(p)
    return {"variant": p["variant"], "u_star": round(us, 5), "v_star": round(vs, 5), "w_star": round(us + vs, 5),
            "lr_star": _sig3(2.0 ** us), "wd_star": _sig3(2.0 ** vs),
            "best_loss": round(floor_loss(p, Bn, Tn), 6),
            "decoy_penalty": rec, "ofat_penalty": round(penalty(p, uo, vo), 6),
            "carry_over_penalty": round(penalty(p, p["u0"], p["v0"]), 6),
            "near2_lr_penalty": round(p["A"] * (un - us) ** 2, 6),
            "did2_lr_penalty": round(p["A"] * (lr_did(p, 2) - us) ** 2, 6),
            "lr_offsets_log2": {"did4": round(lr_did(p, 4) - us, 3), "did2": round(lr_did(p, 2) - us, 3),
                                "trend": round(lr_trend(p) - us, 3), "mixed": round(lr_mixed(p) - us, 3),
                                "near2": round(un - us, 3)}}


def _conditions(p):
    Bn, Tn = p["B_new"], p["T_new"]
    Sn = Tn - Bn
    sig = p["sig"]
    info = {}
    rec = {k: penalty(p, *uv) for k, uv in _recipes(p).items()}
    # C1: recitations / proxy optimum >= K_DECOY eps;  P7/P8 traps >= K_TRAP eps
    bad = [k for k, v in rec.items() if v < (K_TRAP if k in TRAP_KEYS else K_DECOY) * EPS_LOSS]
    c_rec = not [k for k in bad if k not in TRAP_KEYS]
    c_trap = not [k for k in bad if k in TRAP_KEYS]
    # C2: searches
    thr = K_DECOY * EPS_LOSS
    c_ofat = penalty(p, *_ofat_exact(p)) >= thr
    c_cross = penalty(p, *_cross_exact(p)) >= 1.5 * EPS_LOSS
    c_sh = min(penalty(p, *uv) for uv in _sh_grid(p)) >= 1.5 * EPS_LOSS
    pg = _proxy_grid(p)
    lp = [exp_loss(p, Bn, Tn - 3, *uv) for uv in pg]
    near = [uv for uv, l in zip(pg, lp) if l <= min(lp) + 2.5 * sig]
    c_pg = min(penalty(p, *uv) for uv in near) >= 1.5 * EPS_LOSS
    # C3: the reference design is solvable under its own error model
    d = oracle_design(p)
    steps = [walk_steps(mu(p, s["B"], s["T"]) - s["c"], s["h"]) for s in d["lr"]]
    ok_prox = max(steps) <= MAX_WALK - 1                     # one spare step for noise
    fl = d["lr_full"]
    c_full = lr_did(p, CENTRE_DEPTH)                         # the full bracket starts from the 1/4-length DiD
    m_full = c_full + (fl["h"] / 2.0 if mu(p, Bn, Tn) > c_full else -fl["h"] / 2.0)
    off_u = mu(p, Bn, Tn) - m_full                           # offset from the adaptive bracket's centre
    vu = _vertex_var(p["A"], sig, fl["h"], fl["seeds"], off_u)
    sw = d["wd"]
    cm = curv_w(p, sw["T"])
    off_w = nu(p, sw["B"], sw["T"]) - sw["c"]
    vnu = _vertex_var(cm, sig, sw["h"], sw["seeds"], off_w) * (Sn / sw["S"]) ** 2
    e_pen = p["A"] * vu + p["C1"] * vnu
    x = off_u / fl["h"]
    lw = _lagr(x)
    half = 2.8 * sig * math.sqrt(sum(l * l for l in lw) / fl["seeds"])
    c_orc = e_pen <= EPS_LOSS / 5.0 and abs(off_w) <= 1.2 * sw["h"] and ok_prox and abs(x) <= 0.8
    # C4: R2 feasible from the sweep's own interpolation at 2.8 sigma
    c_r2 = 2 * half <= WIDTH_CAP
    # D1: wd curvature visible at the oracle's wd horizon
    sd2 = sig * math.sqrt(6.0 / sw["seeds"])
    z_d1 = 2 * cm * sw["h"] ** 2 / sd2
    # D2: the proxy DiD is visibly horizon-dependent (1/16 vs 1/4 of each setup)
    sd_v = math.sqrt(_vertex_var(p["A"], sig, H_PROXY, 2, 0.5))
    z_d2 = abs(lr_did(p, 2) - lr_did(p, 4)) / (2.0 * sd_v)          # 4 vertices in the difference
    # T1: wd nearly invisible at the cheapest proxy
    z_t1 = curv_w(p, Tn - PROXY_DEPTH) / sig
    # C5: budget
    cost = sum((3 + n + 1) * _run_cost(p, s["T"], s["seeds"]) for s, n in zip(d["lr"], steps)) + \
        3 * _run_cost(p, sw["T"], sw["seeds"]) + \
        3 * _run_cost(p, Tn, fl["seeds"])
    c_bud = cost <= BUDGET
    conds = [c_rec, c_trap, c_ofat, c_cross, c_sh, c_pg, c_orc, c_r2, z_d1 >= 4.0, z_d2 >= 4.0, z_t1 <= 2.0, c_bud]
    info.update({"rec_below_thr": bad,
                 "trap_penalty_eps": {k: round(rec[k] / EPS_LOSS, 2) for k in TRAP_KEYS},
                 "oracle_expected_penalty": round(e_pen, 7), "oracle_centre_offset_h": round(x, 3),
                 "oracle_half_width": round(half, 5), "disc_z_wd_curvature": round(z_d1, 2),
                 "disc_z_did_horizon": round(z_d2, 1), "trap_z_proxy_wd": round(z_t1, 2),
                 "oracle_cost": round(cost, 2), "bc_new_minus_bn": round(bcrit(p, Tn) - Bn, 3),
                 "wd_sweep_at": {"B": sw["B"], "T": sw["T"], "seeds": sw["seeds"]}})
    return conds, info


COND_NAMES = ["C1_recitations", "C1b_P7P8_traps", "C2_ofat", "C2_cross", "C2_sh", "C2_proxy_grid", "C3_oracle",
              "C4_r2", "D1_wd_visible", "D2_did_horizon", "T1_proxy_wd", "C5_budget"]


# ------------------------------------------------------------------ instance construction
def _hidden(g, base):
    Bn, Tn = base["B_new"], base["T_new"]
    h = {"rho": round(float(g.uniform(RLO, RHI)), 4), "alpha": round(float(g.uniform(ALO, AHI)), 4),
         "kappa": round(float(g.uniform(0.50, 0.90)), 4), "cn": round(float(g.uniform(0.0, 1.5)), 4),
         "beta": round(float(g.uniform(0.15, 0.95)), 4), "gamma": round(float(g.uniform(0.90, 1.50)), 4),
         "A": round(float(g.uniform(0.030, 0.050)), 5), "C1": round(float(g.uniform(0.050, 0.090)), 5),
         "sig": round(float(g.uniform(0.0012, 0.0018)), 5)}
    h["c0"] = round(Bn + h["cn"] - h["kappa"] * Tn, 6)
    return h


def sample_params(seed):
    """Construct, don't screen: public part drawn once, hidden constants redrawn until the conditions hold."""
    g = np.random.default_rng(93000 + seed)
    base = _base(g)
    p = None
    for _ in range(400):
        p = dict(base)
        p.update(_hidden(g, base))
        if all(_conditions(p)[0]):
            break
    return p


def instance_gate(p):
    c, info = _conditions(p)
    t = truth(p)
    info = dict(info)
    info.update(t)
    info["conds"] = [bool(x) for x in c]
    return bool(all(c)), info


def pool_gate(seeds):
    ps = [s for s in seeds if instance_gate(sample_params(s))[0]]
    n = len(ps)
    if n < 4:
        return True, {"n": n}
    vs = {}
    for s in ps:
        v = sample_params(s)["variant"]
        vs[v] = vs.get(v, 0) + 1
    mc = max(vs.values())
    return bool(mc <= math.ceil(n / 4.0) and len(vs) >= 4), {"n": n, "modal_variant": mc, "variants": vs,
                                                            "cap": math.ceil(n / 4.0)}


# ------------------------------------------------------------------ ops
def _run_run(w, a, ctx):
    p = w.p
    B, T, u, v, seeds, bt, tok, lr, wd = _parse_run(w, a)
    m = exp_loss(p, B, T, u, v)
    g = rng_for(w.salt, "run", ctx["i"])
    vals = [m + float(g.normal(0, p["sig"])) for _ in range(seeds)]
    return {"batch": bt, "tokens_B": round(tok, 6), "lr": lr, "wd": wd, "seeds": seeds,
            "steps": int(round(tok * 1e9 / (bt * 4096))),
            "final_val_loss": round(float(np.mean(vals)), 5), "per_seed": [round(x, 5) for x in vals]}


class World(_W):
    NAME = "r_crit"
    ARTIFACTS = ["recipe.json"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "credits"
    OPS = {"run": (_cost_run, _run_run, "train with the given batch, token count, lr and wd; returns final val loss")}

    def public_spec(self):
        return R.World.public_spec(self)

    def grade(self, art_dir, ledger=None):
        p = self.p
        t = truth(p)
        items = {}
        try:
            rep = json.load(open(os.path.join(art_dir, "recipe.json")))
        except Exception as ex:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {"R0_recipe": {"ok": False,
                                            "detail": "recipe.json missing or unreadable: %s" % type(ex).__name__}}}
        why, ok0 = [], True
        try:
            lr, wd = float(rep["lr"]), float(rep["wd"])
            if not (math.isfinite(lr) and math.isfinite(wd) and lr > 0 and wd > 0):
                raise ValueError
        except Exception:
            ok0 = False; why.append("lr and wd must be positive finite numbers")
        iv = rep.get("loss") if isinstance(rep, dict) else None
        try:
            lo, hi = float(iv["lo"]), float(iv["hi"])
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError
        except Exception:
            ok0 = False; why.append("loss must be an object with finite numeric lo <= hi")
        items["R0_recipe"] = (ok0, "; ".join(why) or "recipe.json is well-formed")
        if not ok0:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()}}
        u, v = math.log2(lr), math.log2(wd)
        pen = penalty(p, u, v)
        us, vs = optimum(p)
        pen_lr = p["A"] * (u - us) ** 2
        pen_wd = p["C1"] * (u + v - us - vs) ** 2
        items["R1_recipe"] = (
            pen <= EPS_LOSS + 1e-12,
            "expected final val loss of your recipe at the new setup exceeds the best achievable by %.5f "
            "(allowed %.4f): lr part %.5f, lr*wd part %.5f.  Best recipe lr=%.3g wd=%.3g; yours lr=%.3g wd=%.3g "
            "(log2 offsets: lr %+.2f, lr*wd %+.2f)." % (pen, EPS_LOSS, pen_lr, pen_wd, 2 ** us, 2 ** vs, lr, wd,
                                                        u - us, u + v - us - vs))
        tgt = floor_loss(p, p["B_new"], p["T_new"]) + pen
        cov, wide = lo <= tgt <= hi, (hi - lo) > WIDTH_CAP + 1e-9
        items["R2_loss"] = (bool(cov and not wide),
                            "expected final val loss of YOUR recipe at the new setup is %.5f; you reported "
                            "[%.5f, %.5f] (%s, width %.5f, cap %.4f)." % (tgt, lo, hi, "covers" if cov else "misses",
                                                                        hi - lo, WIDTH_CAP))
        rec = _recipes(p)
        near = {k: round(math.hypot(u - uv[0], v - uv[1]), 3) for k, uv in rec.items()}
        closest = min(near, key=near.get)
        diag = {"variant": p["variant"], "penalty": round(pen, 6), "pen_lr": round(pen_lr, 6),
                "pen_lrwd": round(pen_wd, 6), "du_log2": round(u - us, 3), "dw_log2": round(u + v - us - vs, 3),
                "wd_changed": abs(v - p["v0"]) > 0.05, "closest_decoy": closest, "closest_decoy_dist": near[closest],
                "lr_offset_vs_shortcuts": {k: round(u - (us + d_), 3) for k, d_ in t["lr_offsets_log2"].items()},
                "interval_centre_minus_truth": round((lo + hi) / 2 - tgt, 6)}
        if ledger is not None:
            runs = [r for r in ledger if r["op"] == "run"]
            Tn = p["T_new"]
            fracs = []
            for r in runs:
                try:
                    fracs.append(math.log2(float(r["args"]["tokens"]) / p["D0"]) - Tn)
                except Exception:
                    pass
            by_w, by_u = {}, {}
            for r in runs:
                a = r["args"]
                try:
                    key = (int(a["batch"]), round(float(a["tokens"]), 6))
                    by_w.setdefault(key + (round(float(a["lr"]), 12),), set()).add(round(float(a["wd"]), 12))
                    by_u.setdefault(key, set()).add(round(float(a["lr"]), 12))
                except Exception:
                    pass
            wd_depths = [math.log2(k[1] / p["D0"]) - Tn for k, s in by_w.items() if len(s) >= 2]
            bn = p["B0"] * 2 ** p["B_new"]
            lr_depths = [math.log2(k[1] / p["D0"]) - Tn for k, s in by_u.items() if len(s) >= 2 and k[0] == bn]
            diag.update({"n_runs": len(runs), "n_full_length": sum(1 for f in fracs if f > -1e-6),
                         "shortest_octaves_below": round(min(fracs), 2) if fracs else None,
                         "wd_varied_at_fixed_lr_max_octave": round(max(wd_depths), 2) if wd_depths else None,
                         "lr_varied_new_batch_max_octave": round(max(lr_depths), 2) if lr_depths else None,
                         "spent": round(sum(r["cost"] for r in ledger), 2)})
        n_ok = sum(1 for v_ in items.values() if v_[0])
        return {"pass": bool(all(v_[0] for v_ in items.values())), "score": round(n_ok / 3.0, 4),
                "items": {k: {"ok": bool(v_[0]), "detail": v_[1]} for k, v_ in items.items()},
                "diag": diag, "ref": t}


# ------------------------------------------------------------------ strategies
def _run(sess, B, T, u, v, seeds=1):
    p = sess.w.p
    return sess.call("run", {"batch": p["B0"] * 2 ** B, "tokens": p["D0"] * 2.0 ** T, "lr": 2.0 ** u,
                             "wd": 2.0 ** v, "seeds": int(seeds)})["final_val_loss"]


def _sweep_u(sess, B, T, c, h, w, seeds):
    ys = [_run(sess, B, T, c + d, w - (c + d), seeds) for d in (-h, 0.0, h)]
    return _vertex(c, h, ys)


def _walk_u(sess, B, T, c, h, w, seeds):
    """3-point lr sweep (lr*wd fixed) that walks toward the better edge until the minimum is bracketed."""
    f = lambda u: _run(sess, B, T, u, w - u, seeds)
    ys = {c - h: f(c - h), c: f(c), c + h: f(c + h)}
    for _ in range(MAX_WALK):
        xs = sorted(ys)
        i = min(range(len(xs)), key=lambda j: ys[xs[j]])
        if 0 < i < len(xs) - 1:
            break
        nx = xs[0] - h if i == 0 else xs[-1] + h
        ys[nx] = f(nx)
    xs = sorted(ys)
    i = min(max(1, min(range(len(xs)), key=lambda j: ys[xs[j]])), len(xs) - 2)
    return _vertex(xs[i], h, [ys[xs[i - 1]], ys[xs[i]], ys[xs[i + 1]]])


def _confirm(sess, art_dir, u, v, seeds=1):
    p = sess.w.p
    uu, vv = math.log2(float("%.6g" % 2.0 ** u)), math.log2(float("%.6g" % 2.0 ** v))
    y = _run(sess, p["B_new"], p["T_new"], uu, vv, seeds)
    half = 2.8 * p["sig"] / math.sqrt(seeds)
    _write(art_dir, uu, vv, y - half, y + half)


def _lr_did_meas(sess, k=PROXY_DEPTH):
    """Anchor-calibrated DiD: k = PROXY_DEPTH is the r_retune reference (2 seeds at 1/16), k = CENTRE_DEPTH the
    r_crit oracle's centring step (1 seed at 1/4)."""
    p = sess.w.p
    sws = did_sweeps(p, k, 2 if k == PROXY_DEPTH else 1)
    w_fix = p["w0"] + 0.5 * k
    m = [_walk_u(sess, s["B"], s["T"], s["c"], s["h"], w_fix, s["seeds"]) for s in sws]
    return p["u0"] + (m[1] - m[0])


def _wd_measured(sess, u_any):
    p = sess.w.p
    sw = R.oracle_design(p)["wd"]
    Sn = p["T_new"] - p["B_new"]
    nm = _sweep_w(sess, sw["B"], sw["T"], u_any, sw["c"], sw["h"], sw["seeds"])
    return p["w0"] - (p["w0"] - nm) / sw["S"] * Sn


def _bracket_centre(c, h, first_pair_right_better):
    return c + h / 2.0 if first_pair_right_better else c - h / 2.0


def _lr_full(sess, c, w, h=H_FULL):
    """Adaptive 3-point full-length lr sweep at fixed lr*wd: runs at c-h/2 and c+h/2, then extend one step h on
    the better side.  Returns (vertex, fitted loss at the vertex, 2.8-sigma half-width from the Lagrange
    weights).  Covers a centring error of up to ~1.4 h with the vertex inside the bracket."""
    p = sess.w.p
    Bn, Tn = p["B_new"], p["T_new"]
    f = lambda u: _run(sess, Bn, Tn, u, w - u, 1)
    ya, yb = f(c - h / 2.0), f(c + h / 2.0)
    m = _bracket_centre(c, h, yb < ya)
    if yb < ya:
        ys = [ya, yb, f(c + 1.5 * h)]
    else:
        ys = [f(c - 1.5 * h), ya, yb]
    uh = _vertex(m, h, ys)
    lw = _lagr((uh - m) / h)
    yv = float(sum(l * y for l, y in zip(lw, ys)))
    half = min(WIDTH_CAP / 2.0 - 1e-6, 2.8 * p["sig"] * math.sqrt(sum(l * l for l in lw)))
    return uh, yv, half


def strat_oracle(sess, art_dir, rng):
    """Proxy DiD only to centre; effective decay measured where it matters; lr measured AT full length on the
    swept line through the shipped lr*wd, so the sweep's own parabola gives the loss of the shipped recipe."""
    ud = _lr_did_meas(sess, CENTRE_DEPTH)
    ws = _wd_measured(sess, ud)
    uh, yv, half = _lr_full(sess, ud, ws)
    uu = math.log2(float("%.6g" % 2.0 ** uh))
    vv = math.log2(float("%.6g" % 2.0 ** (ws - uh)))
    _write(art_dir, uu, vv, yv - half, yv + half)


def strat_carry_over(sess, art_dir, rng):
    """P1 ABLATION: ship the production recipe unchanged."""
    p = sess.w.p
    _confirm(sess, art_dir, p["u0"], p["v0"])


def strat_recite_rules(sess, art_dir, rng):
    """P2 ABLATION: sqrt batch rule for lr, constant-timescale (exponent 1) rule for lr*wd, no measurement."""
    p = sess.w.p
    u = p["u0"] + 0.5 * p["B_new"]
    w = p["w0"] - (p["T_new"] - p["B_new"])
    _confirm(sess, art_dir, u, w - u)


def strat_lr_only(sess, art_dir, rng):
    """P3 ABLATION: the lr measured correctly at full length, production wd kept."""
    p = sess.w.p
    ud = _lr_did_meas(sess, CENTRE_DEPTH)
    uh, _, _ = _lr_full(sess, ud, p["w0"])
    _confirm(sess, art_dir, uh, p["v0"])


def strat_lr_tau1(sess, art_dir, rng):
    """P4 ABLATION: the lr measured correctly at full length, lr*wd from the textbook exponent-1 rule."""
    p = sess.w.p
    ud = _lr_did_meas(sess, CENTRE_DEPTH)
    w = p["w0"] - (p["T_new"] - p["B_new"])
    uh, yv, half = _lr_full(sess, ud, w)
    _write(art_dir, uh, w - uh, yv - half, yv + half)


def strat_proxy_direct(sess, art_dir, rng):
    """P5 ABLATION: tune both knobs properly at the cheap proxy length and ship."""
    p = sess.w.p
    Bn, Tp = p["B_new"], p["T_new"] - PROXY_DEPTH
    cu = p["u0"] + 0.5 * Bn - 0.3 * p["T_new"] + 0.3 * PROXY_DEPTH
    w_fix = p["w0"] + 0.5 * PROXY_DEPTH
    up = _sweep_u(sess, Bn, Tp, cu, 1.0, w_fix, 2)
    wp = _sweep_w(sess, Bn, Tp, up, p["w0"] + 0.5 * PROXY_DEPTH, 1.5, 2)
    _confirm(sess, art_dir, up, wp - up)


def strat_full_ofat(sess, art_dir, rng):
    """P6 ABLATION / SEARCH: full-length lr line search at the production wd; ship the best measured point."""
    p = sess.w.p
    cu = p["u0"] + 0.5 * p["B_new"] - 0.3 * p["T_new"]
    best = None
    for du in (-0.5, 0.0, 0.5):
        y = _run(sess, p["B_new"], p["T_new"], cu + du, p["v0"], 1)
        if best is None or y < best[0]:
            best = (y, cu + du)
    half = 2.8 * p["sig"]
    _write(art_dir, best[1], p["v0"], best[0] - half, best[0] + half)


def strat_proxy_did(sess, art_dir, rng):
    """P7 ABLATION = the r_retune reference protocol: anchor-calibrated proxy DiD for lr, effective decay
    measured where it matters, one full-length confirmation."""
    ud = _lr_did_meas(sess)
    ws = _wd_measured(sess, ud)
    _confirm(sess, art_dir, ud, ws - ud)


def strat_short_trend(sess, art_dir, rng):
    """P8 ABLATION: lr optimum measured on the new setup at 1/32 .. 1/4 of the target (anchor included when on
    the same batch), log-linear trend extrapolated; effective decay measured where it matters; confirm."""
    p = sess.w.p
    Bn, Tn = p["B_new"], p["T_new"]
    xs = _trend_pts(p)
    w_fix = p["w0"]
    ys = []
    for T in xs:
        c = p["u0"] + 0.5 * Bn - 0.3 * T
        ys.append(_sweep_u(sess, Bn, T, c, H_PROXY, w_fix, 1))
    if Bn == 0:
        xs, ys = list(xs) + [0.0], ys + [p["u0"]]
    ut = _linfit_at(xs, ys, Tn)
    ws = _wd_measured(sess, ut)
    _confirm(sess, art_dir, ut, ws - ut)


def strat_successive_halving(sess, art_dir, rng):
    """SEARCH: 3x3 grid around the scaled production recipe at 1/16 length, top 3 at 1/4, top 1 at full."""
    p = sess.w.p
    Bn, Tn = p["B_new"], p["T_new"]
    r1 = sorted(((_run(sess, Bn, Tn - 4, u, v), (u, v)) for u, v in _sh_grid(p)))[:3]
    r2 = sorted(((_run(sess, Bn, Tn - 2, u, v), (u, v)) for _, (u, v) in r1))[:1]
    u, v = r2[0][1]
    y = _run(sess, Bn, Tn, u, v)
    _write(art_dir, u, v, y - 2.8 * p["sig"], y + 2.8 * p["sig"])


def strat_proxy_grid(sess, art_dir, rng):
    """SEARCH: 4x4 grid at 1/8 length on the new setup, ship the best, confirm at full length."""
    p = sess.w.p
    Bn, Tn = p["B_new"], p["T_new"]
    best = min(((_run(sess, Bn, Tn - 3, u, v), (u, v)) for u, v in _proxy_grid(p)))
    _confirm(sess, art_dir, *best[1])


def strat_full_cross(sess, art_dir, rng):
    """SEARCH: all five affordable full-length runs as a cross in (lr, wd) around the sqrt-rule lr and the
    production wd; per-axis parabola vertices; loss interval from the best measured point."""
    p = sess.w.p
    Bn, Tn = p["B_new"], p["T_new"]
    cu, cv = _cross_centre(p)
    y0 = _run(sess, Bn, Tn, cu, cv)
    yu = [_run(sess, Bn, Tn, cu - 0.7, cv), y0, _run(sess, Bn, Tn, cu + 0.7, cv)]
    yv = [_run(sess, Bn, Tn, cu, cv - 1.0), y0, _run(sess, Bn, Tn, cu, cv + 1.0)]
    u1, v1 = _vertex(cu, 0.7, yu), _vertex(cv, 1.0, yv)
    yb = min(yu + yv)
    _write(art_dir, u1, v1, yb - 2.8 * p["sig"], yb + 2.8 * p["sig"])


def strat_wide(sess, art_dir, rng):
    p = sess.w.p
    us, vs = optimum(p)
    L = floor_loss(p, p["B_new"], p["T_new"])
    _write(art_dir, us, vs, L - 0.02, L + 0.02)


def strat_nop(sess, art_dir, rng):
    pass


def strat_bad_json(sess, art_dir, rng):
    open(os.path.join(art_dir, "recipe.json"), "w").write("{nope")


STRATEGIES = {"oracle": (strat_oracle, "pass"),
              "carry_over": (strat_carry_over, "fail"),
              "recite_rules": (strat_recite_rules, "fail"),
              "lr_only": (strat_lr_only, "fail"),
              "lr_tau1": (strat_lr_tau1, "fail"),
              "proxy_direct": (strat_proxy_direct, "fail"),
              "full_ofat": (strat_full_ofat, "fail"),
              "proxy_did": (strat_proxy_did, "fail"),
              "short_trend": (strat_short_trend, "fail"),
              "successive_halving": (strat_successive_halving, "fail"),
              "proxy_grid": (strat_proxy_grid, "fail"),
              "full_cross": (strat_full_cross, "fail"),
              "wide": (strat_wide, "fail"),
              "nop": (strat_nop, "fail"),
              "bad_json": (strat_bad_json, "fail")}
NOISY_FAIL = ()
SEARCH = ["successive_halving", "proxy_grid", "full_ofat", "full_cross"]
PRINCIPLES = {"P1_retune_after_forced_change": ("carry_over", ["R1_recipe"]),
              "P2_measure_dont_recite": ("recite_rules", ["R1_recipe"]),
              "P3_wd_must_move": ("lr_only", ["R1_recipe"]),
              "P4_tau_exponent_is_measured": ("lr_tau1", ["R1_recipe"]),
              "P5_proxy_length_is_biased": ("proxy_direct", ["R1_recipe"]),
              "P6_lr_and_lrwd_are_the_knobs": ("full_ofat", ["R1_recipe"]),
              "P7_batch_gain_depends_on_horizon": ("proxy_did", ["R1_recipe"]),
              "P8_lr_horizon_trend_is_curved": ("short_trend", ["R1_recipe"])}


# ------------------------------------------------------------------ grader falsification
def _set(p, art_dir, u, v, dlo, dhi):
    tgt = floor_loss(p, p["B_new"], p["T_new"]) + penalty(p, u, v)
    rep = _load(art_dir)
    rep.update({"lr": 2.0 ** u, "wd": 2.0 ** v, "loss": {"lo": tgt + dlo, "hi": tgt + dhi}})
    _save(art_dir, rep)


def _mut_lr(p, t, art_dir, leg):
    us, vs = optimum(p)
    if leg == "small":
        _set(p, art_dir, us + 0.03, vs - 0.03, -0.4 * WIDTH_CAP, 0.4 * WIDTH_CAP)
        return None
    du = 2.0 * math.sqrt(EPS_LOSS / p["A"])
    _set(p, art_dir, us + du, vs - du, -0.4 * WIDTH_CAP, 0.4 * WIDTH_CAP)
    return "R1_recipe"


def _mut_wd(p, t, art_dir, leg):
    us, vs = optimum(p)
    if leg == "small":
        rep = _load(art_dir)
        rep["notes"] = "extra key; nothing else in the file is read"
        _save(art_dir, rep)
        return None
    dv = 2.0 * math.sqrt(EPS_LOSS / p["C1"])
    _set(p, art_dir, us, vs + dv, -0.4 * WIDTH_CAP, 0.4 * WIDTH_CAP)
    return "R1_recipe"


def _mut_loss_cov(p, t, art_dir, leg):
    us, vs = optimum(p)
    d = 0.1 * WIDTH_CAP if leg == "small" else 1.5 * WIDTH_CAP
    _set(p, art_dir, us, vs, d - 0.45 * WIDTH_CAP, d + 0.45 * WIDTH_CAP)
    return None if leg == "small" else "R2_loss"


def _mut_loss_width(p, t, art_dir, leg):
    us, vs = optimum(p)
    wdt = 1.0 * WIDTH_CAP if leg == "small" else 1.2 * WIDTH_CAP
    _set(p, art_dir, us, vs, -wdt / 2, wdt / 2)
    return None if leg == "small" else "R2_loss"


def _mut_trap(p, t, art_dir, leg):
    """The P7 recipe itself (proxy-DiD lr, exact lr*wd) with a correct interval must fail R1; the optimum with
    a correct interval must pass."""
    us, vs = optimum(p)
    if leg == "small":
        _set(p, art_dir, us, vs, -0.4 * WIDTH_CAP, 0.4 * WIDTH_CAP)
        return None
    u = lr_did(p, PROXY_DEPTH)
    _set(p, art_dir, u, us + vs - u, -0.4 * WIDTH_CAP, 0.4 * WIDTH_CAP)
    return "R1_recipe"


MUTATE = [("lr_value", _mut_lr), ("wd_value", _mut_wd), ("loss_coverage", _mut_loss_cov),
          ("loss_width", _mut_loss_width), ("p7_recipe", _mut_trap)]


# ------------------------------------------------------------------ task text: identical to r_retune
_STORY = {
    "batch8x": R._MIG.replace("{k}", "eight") + "  The token budget is unchanged at **{dn:g}B tokens**.",
    "batch16x": R._MIG.replace("{k}", "sixteen") + "  The token budget is unchanged at **{dn:g}B tokens**.",
    "tokens8x": R._DATA + "  The global batch stays at **{bn} sequences**.",
    "tokens16x": R._DATA + "  The global batch stays at **{bn} sequences**.",
    "batch2x_tokens8x": "Two things change at once.  " + R._MIG.replace("{k}", "two") + "  And: " +
                        R._DATA[0].lower() + R._DATA[1:]}
assert set(_STORY) == set(VARIANTS)


def instruction(p, t):
    """r_retune's task text with r_crit's own stories, budget and tolerance (same operational facts)."""
    bn = p["B0"] * 2 ** p["B_new"]
    dn = p["D0"] * 2 ** p["T_new"]
    q = dict(p, variant="batch8x")                   # any key R knows; the story line is swapped below
    txt = R.instruction(q, t)
    old_story = R._STORY["batch8x"].format(bn=bn, b0=p["B0"], dn=dn, d0=p["D0"])
    new_story = _STORY[p["variant"]].format(bn=bn, b0=p["B0"], dn=dn, d0=p["D0"])
    assert txt.count(old_story) == 1
    txt = txt.replace(old_story, new_story)
    for a, b in (("You have **%g credits**." % R.BUDGET, "You have **%g credits**." % BUDGET),
                 ("within **%.4f** of the best" % R.EPS_LOSS, "within **%.4f** of the best" % EPS_LOSS)):
        assert txt.count(a) == 1, a
        txt = txt.replace(a, b)
    return txt


def docs(p):
    return R.docs(p)


def hints(p):
    return {1: R.hints(p)[1], 2: """
Hint 2: how far a larger batch moves the lr optimum depends on how far into training you are; an lr correction
calibrated on short runs need not hold at full length.  Measure lr x wd where it visibly matters, and check the
lr at the length it will be used.
"""}
