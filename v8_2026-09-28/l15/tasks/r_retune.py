"""R-RETUNE: re-tune (lr, wd) after a forced change to the pretraining setup.

The situation every pretraining team meets: the production recipe (lr0, wd0) was tuned at full scale and is
the optimum for the CURRENT setup.  Something outside the team's control changes the setup - the global batch
has to double because of a cluster migration, or the token budget doubles - and the next run has to ship with
a recipe tuned for the NEW setup, on a budget that buys ~3 full-length runs.

Difficulty mechanism (this family's *signature*, in the v8 sense: standard experimental hygiene itself yields
a complete, self-consistent, WRONG recipe that the agent's own full-horizon check then confirms):

  P3  WD MUST MOVE.  The obvious retune moves the learning rate and keeps the production weight decay.  Under
      decoupled AdamW the per-step decay is lr*wd, so moving lr alone already moves the effective decay, and
      the forced change moves the optimal effective decay again (it depends on the run length in STEPS).
  P4  THE TEXTBOOK EXPONENT IS NOT THIS WORLD'S.  The standard rule for how the optimal lr*wd scales with the
      number of steps (constant timescale as a fraction of training, exponent 1) is recited, not measured; in
      this world the exponent `beta` is hidden and < 1 (as later pretraining work reports).  Same for the
      batch-lr exponent `rho` (sqrt/linear rules are recitations) and the horizon shift `alpha` of the lr optimum.
  P5  THE CHEAP PROXY IS BIASED, AND WEAKEST EXACTLY ON WD.  Short proxy runs (cosine schedule over their own,
      shorter length) prefer a higher lr and a higher lr*wd, and at proxy length the loss responds to wd far
      less than at full length (the wd curvature grows with run length).  "Tune on short runs, confirm at full
      length" ships the proxy optimum; "calibrate the proxy against the anchor" removes the bias, but done at
      the cheapest proxy length the wd reading is too noisy (NOISY_FAIL `proxy_did_both`).
  P6  OFAT IN THE WRONG COORDINATES.  A full-length lr line search at the production wd is a controlled
      experiment - and its optimum is biased, because lr and wd are not the independent knobs (lr and lr*wd are).

The confirmation that a careful agent runs - one full-length run of its recipe - PASSES on every one of these
wrong recipes: each beats the carried-over recipe, and a single run cannot show how far a recipe is from the
optimum.  Only the rubric's item R1 (expected-loss gap to the true optimum) sees it.

RECITATION RESISTANCE.  Every forced change moves log2(steps) by >= 2 octaves (|S| >= 2), the hidden
exponents are drawn wide (beta in [0.15, 0.95], rho in [0.1, 0.9]), and the tolerance radius in lr*wd is ~0.1
octave, so no fixed exponent covers more than a small slice of the prior; `_conditions` additionally rejects
any instance on which a recitable rule (lr: none/sqrt/linear in batch; lr*wd: keep wd, keep lr*wd, exponent
1/2, exponent 1, and the batch-1/tokens-1/2 form) lands within K_DECOY * EPS of the optimum, with the lr part
recited or measured.

Ground truth is exact and closed form.  In log2 units (u = log2 lr, v = log2 wd, w = u + v = log2 lr*wd),
with B = log2(batch/B0), T = log2(tokens/D0), S = T - B (log2 steps relative to the current setup):

    E[loss] = F(B, T) + A (u - mu(B,T))^2 + C(T) (w - nu(B,T))^2
    mu(B,T) = u0 + rho B - alpha T                 lr optimum
    nu(B,T) = w0 - beta S                          optimal effective decay: tau-law with hidden exponent
    C(T)    = C1 2^(gamma (T - T_new))             wd sensitivity grows with run length
    F(B,T)  = E + K 2^(-a T) + b B

At (B,T) = (0,0) the optimum is (u0, w0) exactly, i.e. the stated production recipe.  Seed noise is N(0, sig^2)
per seed and is the only randomness.  So the optimum (u*, w*) = (mu, nu) at the new setup, the loss gap of any
recipe, and every decoy's gap are computed, not sampled.

DISCOVERABILITY (condition D1 in `_conditions`): with the reference design, the wd sweep at the oracle's
measurement horizon shows curvature >= 4 sd of its own second difference; nothing in the text suggests it.
TRAP STRENGTH (condition T1): at the cheapest proxy length, halving or doubling wd moves the loss by <= 2 sd
of one seed - a quick proxy check reads "wd barely matters".
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

BUDGET = 224.0
FULL_COST = 64.0           # credits for ONE seed of the new run at full length; cost is linear in tokens
MAX_SEEDS = 4
EPS_LOSS = 0.0008          # R1: expected-loss gap to the best achievable (lr, wd) at the new setup
WIDTH_CAP = 0.011          # R2: absolute width of the loss interval
K_DECOY = 2.5              # every deterministic decoy must be >= K_DECOY * EPS_LOSS off the optimum
PROXY_DEPTH = 4            # octaves below the target length where the "cheap proxy" lives
TOKENS_MIN_FRAC = 1.0 / 64 # shortest run allowed, as a fraction of the CURRENT setup's token budget

VARIANTS = {"batch4x": (2, 0), "batch8x": (3, 0), "tokens4x": (0, 2), "tokens8x": (0, 3),
            "batch2x_tokens8x": (1, 3)}
_VORDER = ["batch4x", "tokens4x", "batch8x", "tokens8x", "batch2x_tokens8x"]
RECITED_RHO = (0.0, 0.5, 1.0)                       # lr vs batch: none / sqrt / linear
RECITED_WD = {"wd_keep": None, "lrwd_keep": (0.0, 0.0), "steps_pow_half": (0.5, 0.5),
              "steps_pow_one": (1.0, 1.0), "batch1_tokens_half": (1.0, 0.5)}   # lr*wd = w0 + cB*B - cT*T


def _sig3(x):
    return float("%.3g" % x)


# ------------------------------------------------------------------ hidden mechanics
def mu(p, B, T):
    return p["u0"] + p["rho"] * B - p["alpha"] * T


def nu(p, B, T):
    return p["w0"] - p["beta"] * (T - B)


def curv_w(p, T):
    return p["C1"] * 2.0 ** (p["gamma"] * (T - p["T_new"]))


def floor_loss(p, B, T):
    return p["E"] + p["K"] * 2.0 ** (-p["a"] * T) + p["b"] * B


def exp_loss(p, B, T, u, v):
    w = u + v
    return floor_loss(p, B, T) + p["A"] * (u - mu(p, B, T)) ** 2 + curv_w(p, T) * (w - nu(p, B, T)) ** 2


def penalty(p, u, v):
    """Expected-loss gap of (u, v) at the new setup's full length.  Separable in (u, w)."""
    Bn, Tn = p["B_new"], p["T_new"]
    return p["A"] * (u - mu(p, Bn, Tn)) ** 2 + p["C1"] * (u + v - nu(p, Bn, Tn)) ** 2


def optimum(p):
    Bn, Tn = p["B_new"], p["T_new"]
    u, w = mu(p, Bn, Tn), nu(p, Bn, Tn)
    return u, w - u


# ------------------------------------------------------------------ public reference design (no hidden info)
def _run_cost(p, T, seeds):
    return seeds * FULL_COST * 2.0 ** (T - p["T_new"])


def oracle_design(p):
    """The reference protocol's measurement plan, from PUBLIC facts only (u0, v0, the variant).  Shared by the
    oracle and by `_conditions`, so the solvability condition is computed for the design actually run."""
    Bn, Tn = p["B_new"], p["T_new"]
    Sn = Tn - Bn
    u0, w0 = p["u0"], p["w0"]
    k = PROXY_DEPTH
    # lr: difference-in-differences against the anchor at matched proxy length (same fraction of each setup)
    cu_old = u0 + 0.3 * k                         # prior: short runs like a higher lr
    cu_new = cu_old + 0.5 * Bn - 0.3 * Tn
    lr_sweeps = [dict(B=0, T=-k, c=cu_old, h=1.25, seeds=2), dict(B=Bn, T=Tn - k, c=cu_new, h=1.25, seeds=2)]
    # effective decay: one sweep at a horizon where wd visibly matters, S_m != 0; beta from the anchor.
    # Score (Sn/Sm)^2 * 2^(Tn-Tm): variance ratio x cost ratio under a gamma~1 prior.  One sweep <= 96 credits.
    cands = []
    for Bm in sorted(set([0, Bn])):
        for Tm in (Tn - 1, Tn - 2, Tn - 3, -1, -2):
            Sm = Tm - Bm
            if Sm == 0 or Tm < math.log2(TOKENS_MIN_FRAC) or Tm >= Tn or (Bm, Tm) == (0, 0):
                continue
            cands.append(((Sn / Sm) ** 2 * 2.0 ** (Tn - Tm), Bm, Tm, Sm))
    cands.sort()
    _, Bm, Tm, Sm = cands[0]
    per = _run_cost(p, Tm, 1)
    seeds = max(1, min(MAX_SEEDS, int(96.0 // (3 * per))))
    wd_sweep = dict(B=Bm, T=Tm, S=Sm, c=w0 - 0.5 * Sm, h=max(1.5, 0.6 * abs(Sm)), seeds=seeds)
    return {"lr": lr_sweeps, "wd": wd_sweep, "confirm_seeds": 1}


def _vertex_var(curv, sig, h, seeds, off):
    """Variance of the 3-point parabola vertex (points c-h, c, c+h; `seeds` per point) when the true vertex
    sits `off` from the centre.  First-order delta method: slope noise plus curvature noise x offset."""
    s2 = sig ** 2 / seeds
    v_slope = s2 / (8.0 * curv ** 2 * h ** 2)
    v_curv = off ** 2 * 6.0 * s2 / (4.0 * curv ** 2 * h ** 4)
    return v_slope + v_curv


# ------------------------------------------------------------------ truth and instance conditions
def _recipes(p):
    """Every deterministic recipe a recitation / shortcut would ship, in (u, v)."""
    Bn, Tn = p["B_new"], p["T_new"]
    u0, v0, w0 = p["u0"], p["v0"], p["w0"]
    us, _ = optimum(p)
    out = {}
    lrs = [("rho%g" % r, u0 + r * Bn) for r in RECITED_RHO] + [("measured", us)]
    for lname, u in lrs:
        for wname, cc in RECITED_WD.items():
            if cc is None:
                out["lr_%s__%s" % (lname, wname)] = (u, v0)
            else:
                w = w0 + cc[0] * Bn - cc[1] * Tn
                out["lr_%s__%s" % (lname, wname)] = (u, w - u)
    # proxy optimum at the cheapest proxy length (what "tune short, ship" gives with a perfect sweep)
    Tp = Tn - PROXY_DEPTH
    up = mu(p, Bn, Tp)
    out["proxy_optimum"] = (up, nu(p, Bn, Tp) - up)
    return out


def _ofat_exact(p):
    """Full-length lr line search at the production wd, exact minimiser: u = (A mu + C1 (nu - v0)) / (A + C1)."""
    us, vs = optimum(p)
    ws = us + vs
    A, C1, v0 = p["A"], p["C1"], p["v0"]
    u = (A * us + C1 * (ws - v0)) / (A + C1)
    return u, v0


def _sh_grid(p):
    Bn, Tn = p["B_new"], p["T_new"]
    cu = p["u0"] + 0.5 * Bn - 0.3 * Tn
    return [(cu + du, p["v0"] + dv) for du in (-0.8, 0.0, 0.8) for dv in (-1.0, 0.0, 1.0)]


def _proxy_grid(p):
    Bn, Tn = p["B_new"], p["T_new"]
    cu = p["u0"] + 0.5 * Bn - 0.3 * Tn + 0.3 * 3
    return [(cu + du, p["v0"] + dv) for du in (-0.9, -0.3, 0.3, 0.9) for dv in (-1.2, -0.4, 0.4, 1.2)]


def truth(p):
    us, vs = optimum(p)
    Bn, Tn = p["B_new"], p["T_new"]
    rec = {k: round(penalty(p, *uv), 6) for k, uv in _recipes(p).items()}
    uo, vo = _ofat_exact(p)
    return {"variant": p["variant"], "u_star": round(us, 5), "v_star": round(vs, 5), "w_star": round(us + vs, 5),
            "lr_star": _sig3(2.0 ** us), "wd_star": _sig3(2.0 ** vs),
            "best_loss": round(floor_loss(p, Bn, Tn), 6),
            "decoy_penalty": rec, "ofat_penalty": round(penalty(p, uo, vo), 6),
            "carry_over_penalty": round(penalty(p, p["u0"], p["v0"]), 6)}


def _conditions(p):
    Bn, Tn = p["B_new"], p["T_new"]
    Sn = Tn - Bn
    sig = p["sig"]
    thr = K_DECOY * EPS_LOSS
    info = {}
    # C1: every recited / shortcut recipe is >= K_DECOY eps off
    rec = {k: penalty(p, *uv) for k, uv in _recipes(p).items()}
    bad = [k for k, v in rec.items() if v < thr]
    c_rec = not bad
    # C2: OFAT at full length (exact best case) and the grid searches
    c_ofat = penalty(p, *_ofat_exact(p)) >= thr
    c_sh = min(penalty(p, *uv) for uv in _sh_grid(p)) >= 1.5 * EPS_LOSS
    Tp3 = Tn - 3
    pg = _proxy_grid(p)
    lp = [exp_loss(p, Bn, Tp3, *uv) for uv in pg]
    near = [uv for uv, l in zip(pg, lp) if l <= min(lp) + 2.5 * sig]
    c_pg = min(penalty(p, *uv) for uv in near) >= 1.5 * EPS_LOSS
    # C3: the reference design is solvable: expected penalty <= eps/5 under its own error model
    d = oracle_design(p)
    vu = 0.0
    for sw in d["lr"]:
        off = mu(p, sw["B"], sw["T"]) - sw["c"]
        vu += _vertex_var(p["A"], sig, sw["h"], sw["seeds"], off)
    sw = d["wd"]
    cm = curv_w(p, sw["T"])
    off = nu(p, sw["B"], sw["T"]) - sw["c"]
    vnu = _vertex_var(cm, sig, sw["h"], sw["seeds"], off) * (Sn / sw["S"]) ** 2
    e_pen = p["A"] * vu + p["C1"] * vnu
    c_orc = e_pen <= EPS_LOSS / 5.0 and abs(off) <= 1.2 * sw["h"] and \
        all(abs(mu(p, s["B"], s["T"]) - s["c"]) <= 1.2 * s["h"] for s in d["lr"])
    # C4: R2 is feasible with one full-length seed at 2.8 sigma
    c_r2 = 2 * 2.8 * sig <= WIDTH_CAP
    # D1: discoverability - the wd sweep's second difference is >= 4 of its own sd
    sd2 = sig * math.sqrt(6.0 / sw["seeds"])
    z_d1 = 2 * cm * sw["h"] ** 2 / sd2
    # T1: trap strength - at the cheapest proxy, halving/doubling wd moves the loss by <= 2 sigma of one seed
    z_t1 = curv_w(p, Tn - PROXY_DEPTH) * 1.0 / sig
    # C5: budget covers the reference design
    cost = sum(3 * _run_cost(p, s["T"], s["seeds"]) for s in d["lr"]) + 3 * _run_cost(p, sw["T"], sw["seeds"]) + \
        _run_cost(p, Tn, d["confirm_seeds"])
    c_bud = cost <= BUDGET
    conds = [c_rec, c_ofat, c_sh, c_pg, c_orc, c_r2, z_d1 >= 4.0, z_t1 <= 2.0, c_bud]
    info.update({"rec_below_thr": bad, "tau_rule_penalty": round(rec["lr_measured__steps_pow_one"], 6),
                 "lr_only_penalty": round(rec["lr_measured__wd_keep"], 6),
                 "oracle_expected_penalty": round(e_pen, 7), "disc_z_wd_curvature": round(z_d1, 2),
                 "trap_z_proxy_wd": round(z_t1, 2), "oracle_cost": round(cost, 2),
                 "wd_sweep_at": {"B": sw["B"], "T": sw["T"], "seeds": sw["seeds"]}})
    return conds, info


# ------------------------------------------------------------------ instance construction
def _base(g):
    variant = _VORDER[int(g.integers(len(_VORDER)))]
    Bn, Tn = VARIANTS[variant]
    B0 = int(g.choice([256, 512] if Bn >= 3 else [256, 512, 1024]))
    D0 = int(g.choice([12, 16, 24, 32, 48]))
    lr0 = _sig3(float(np.exp(g.uniform(math.log(1.5e-4), math.log(1.2e-3)))))
    wd0 = _sig3(float(np.exp(g.uniform(math.log(0.02), math.log(0.2)))))
    u0, v0 = math.log2(lr0), math.log2(wd0)
    return {"variant": variant, "B_new": Bn, "T_new": Tn, "B0": B0, "D0": D0, "lr0": lr0, "wd0": wd0,
            "u0": u0, "v0": v0, "w0": u0 + v0,
            "E": round(float(g.uniform(2.25, 2.75)), 4), "K": round(float(g.uniform(0.30, 0.55)), 4),
            "a": round(float(g.uniform(0.20, 0.35)), 4), "b": round(float(g.uniform(0.004, 0.012)), 5)}


def _hidden(g):
    return {"rho": round(float(g.uniform(0.10, 0.90)), 4), "alpha": round(float(g.uniform(0.10, 0.50)), 4),
            "beta": round(float(g.uniform(0.15, 0.95)), 4), "gamma": round(float(g.uniform(0.90, 1.50)), 4),
            "A": round(float(g.uniform(0.030, 0.050)), 5), "C1": round(float(g.uniform(0.050, 0.090)), 5),
            "sig": round(float(g.uniform(0.0012, 0.0018)), 5)}


def sample_params(seed):
    """Construct, don't screen: the public part (variant, setup, production recipe) is drawn once; the hidden
    world constants are then redrawn (deterministically in the seed) until the closed-form conditions hold.
    Returns the last draw if none does - `instance_gate` then rejects it."""
    g = np.random.default_rng(91000 + seed)
    base = _base(g)
    p = None
    for _ in range(400):
        p = dict(base)
        p.update(_hidden(g))
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
def _batches(p):
    return [p["B0"] * 2 ** i for i in range(p["B_new"] + 1)]


def _parse_run(w, a):
    p = w.p
    bt = a.get("batch")
    try:
        bt = int(bt)
    except Exception:
        raise LabError("batch must be one of %s (sequences of 4096 tokens)" % _batches(p))
    if bt not in _batches(p):
        raise LabError("batch must be one of %s (sequences of 4096 tokens), got %r" % (_batches(p), a.get("batch")))
    d_new = p["D0"] * 2 ** p["T_new"]
    tok = num(a.get("tokens"), "tokens", p["D0"] * TOKENS_MIN_FRAC - 1e-12, d_new + 1e-9)
    lr = num(a.get("lr"), "lr", 1e-7, 1.0)
    wd = num(a.get("wd"), "wd", 1e-6, 10.0)
    seeds = int(num(a.get("seeds", 1), "seeds", 1, MAX_SEEDS, integer=True))
    B = math.log2(bt / p["B0"])
    T = math.log2(tok / p["D0"])
    return B, T, math.log2(lr), math.log2(wd), seeds, bt, tok, lr, wd


def _cost_run(w, a):
    B, T, u, v, seeds, *_ = _parse_run(w, a)
    return round(_run_cost(w.p, T, seeds), 6)


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
    NAME = "r_retune"
    ARTIFACTS = ["recipe.json"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "credits"
    OPS = {"run": (_cost_run, _run_run, "train with the given batch, token count, lr and wd; returns final val loss")}

    def public_spec(self):
        p = self.p
        d_new = p["D0"] * 2 ** p["T_new"]
        return {"ops": {"run": {
            "args": {"batch": "global batch in sequences of 4096 tokens, one of %s" % _batches(p),
                     "tokens": "training tokens in billions, %g .. %g" % (p["D0"] * TOKENS_MIN_FRAC, d_new),
                     "lr": "AdamW peak learning rate", "wd": "AdamW weight decay (decoupled, PyTorch convention)",
                     "seeds": "1..%d independent seeds" % MAX_SEEDS},
            "cost": "seeds x %g x tokens / %g credits" % (FULL_COST, d_new),
            "noise": "seed-to-seed std of the final val loss: %.4f" % p["sig"],
            "returns": "final validation loss per seed and the mean"}},
            "current_setup": {"batch": p["B0"], "tokens_B": p["D0"], "lr": p["lr0"], "wd": p["wd0"]},
            "new_setup": {"batch": p["B0"] * 2 ** p["B_new"], "tokens_B": d_new},
            "deliverable": "/app/recipe.json (see /app/docs/recipe_format.md)"}

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
            # longest run length (octaves below target) at which the agent varied wd at a fixed lr
            by = {}
            for r in runs:
                a = r["args"]
                try:
                    key = (int(a["batch"]), round(float(a["tokens"]), 6), round(float(a["lr"]), 12))
                    by.setdefault(key, set()).add(round(float(a["wd"]), 12))
                except Exception:
                    pass
            wd_depths = [math.log2(k[1] / p["D0"]) - Tn for k, s in by.items() if len(s) >= 2]
            diag.update({"n_runs": len(runs), "n_full_length": sum(1 for f in fracs if f > -1e-6),
                         "shortest_octaves_below": round(min(fracs), 2) if fracs else None,
                         "wd_varied_at_fixed_lr_max_octave": round(max(wd_depths), 2) if wd_depths else None,
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


def _vertex(c, h, ys):
    ym, y0, yp = ys
    den = ym - 2 * y0 + yp
    if den <= 1e-12:
        return c + h * (-1 if ym < yp else 1)          # no curvature seen: the better edge
    x = c + h * (ym - yp) / (2.0 * den)
    return float(min(c + 2 * h, max(c - 2 * h, x)))


def _sweep_u(sess, B, T, c, h, w, seeds):
    """lr sweep holding lr*wd fixed (u varies, w fixed): separable, vertex = mu(B,T)."""
    ys = [_run(sess, B, T, c + d, w - (c + d), seeds) for d in (-h, 0.0, h)]
    return _vertex(c, h, ys)


def _sweep_w(sess, B, T, u, c, h, seeds):
    ys = [_run(sess, B, T, u, c + d - u, seeds) for d in (-h, 0.0, h)]
    return _vertex(c, h, ys)


def _write(art_dir, u, v, lo, hi):
    json.dump({"lr": float("%.6g" % 2.0 ** u), "wd": float("%.6g" % 2.0 ** v),
               "loss": {"lo": round(lo, 5), "hi": round(hi, 5)}}, open(os.path.join(art_dir, "recipe.json"), "w"))


def _confirm(sess, art_dir, u, v, seeds=1):
    p = sess.w.p
    # write what is shipped first (rounded), then measure exactly that
    uu, vv = math.log2(float("%.6g" % 2.0 ** u)), math.log2(float("%.6g" % 2.0 ** v))
    y = _run(sess, p["B_new"], p["T_new"], uu, vv, seeds)
    half = 2.8 * p["sig"] / math.sqrt(seeds)
    _write(art_dir, uu, vv, y - half, y + half)


def _lr_did(sess):
    p = sess.w.p
    d = oracle_design(p)
    w_fix = p["w0"] + 0.5 * PROXY_DEPTH                       # any fixed lr*wd: the sweep is separable
    m = [_sweep_u(sess, s["B"], s["T"], s["c"], s["h"], w_fix, s["seeds"]) for s in d["lr"]]
    return p["u0"] + (m[1] - m[0])


def strat_oracle(sess, art_dir, rng):
    """Generic protocol, public facts only.
        lr   : sweep lr at fixed lr*wd on the CURRENT and the NEW setup at the same proxy fraction (1/16 of each
               setup's length); the anchor's optimum is known, so the difference of the two vertices is the lr
               shift at full length (difference-in-differences cancels the proxy's horizon bias).
        decay: sweep lr*wd at ONE longer horizon where wd visibly matters (oracle_design picks it), read the
               tau exponent off the anchor (w0 is optimal at the current setup's full length), extrapolate.
        check: one full-length seed of the shipped recipe for the loss interval."""
    p = sess.w.p
    d = oracle_design(p)
    Sn = p["T_new"] - p["B_new"]
    us = _lr_did(sess)
    sw = d["wd"]
    u_m = us                                                    # separable: any lr works for the w-sweep
    nm = _sweep_w(sess, sw["B"], sw["T"], u_m, sw["c"], sw["h"], sw["seeds"])
    beta = (p["w0"] - nm) / sw["S"]
    ws = p["w0"] - beta * Sn
    _confirm(sess, art_dir, us, ws - us, d["confirm_seeds"])


def strat_carry_over(sess, art_dir, rng):
    """P1 ABLATION: ship the production recipe unchanged."""
    p = sess.w.p
    _confirm(sess, art_dir, p["u0"], p["v0"])


def strat_recite_rules(sess, art_dir, rng):
    """P2 ABLATION: sqrt batch rule for lr, textbook constant-timescale rule for wd, no measurement of either."""
    p = sess.w.p
    u = p["u0"] + 0.5 * p["B_new"]
    w = p["w0"] - (p["T_new"] - p["B_new"])
    _confirm(sess, art_dir, u, w - u)


def strat_lr_only(sess, art_dir, rng):
    """P3 ABLATION: the careful lr retune (anchor-calibrated DiD), production wd kept."""
    p = sess.w.p
    us = _lr_did(sess)
    _confirm(sess, art_dir, us, p["v0"])


def strat_lr_did_tau(sess, art_dir, rng):
    """P4 ABLATION: the careful lr retune, wd from the textbook rule lr*wd*steps = const (exponent 1)."""
    p = sess.w.p
    us = _lr_did(sess)
    w = p["w0"] - (p["T_new"] - p["B_new"])
    _confirm(sess, art_dir, us, w - us)


def strat_proxy_direct(sess, art_dir, rng):
    """P5 ABLATION: tune both knobs properly (separable sweeps) - at the cheap proxy length - and ship."""
    p = sess.w.p
    Bn, Tp = p["B_new"], p["T_new"] - PROXY_DEPTH
    cu = p["u0"] + 0.5 * Bn - 0.3 * p["T_new"] + 0.3 * PROXY_DEPTH
    w_fix = p["w0"] + 0.5 * PROXY_DEPTH
    up = _sweep_u(sess, Bn, Tp, cu, 1.0, w_fix, 2)
    wp = _sweep_w(sess, Bn, Tp, up, p["w0"] + 0.5 * PROXY_DEPTH, 1.5, 2)
    _confirm(sess, art_dir, up, wp - up)


def strat_proxy_did_both(sess, art_dir, rng):
    """Legitimate but underpowered: DiD for both knobs at the cheap proxy length, where wd is nearly invisible.
    Unbiased; the gate checks it fails on most salts (NOISY_FAIL)."""
    p = sess.w.p
    k = PROXY_DEPTH
    us = _lr_did(sess)
    Sn = p["T_new"] - p["B_new"]
    ws_old = _sweep_w(sess, 0, -k, p["u0"] + 0.3 * k, p["w0"] + 0.5 * k, 2.0, 2)
    ws_new = _sweep_w(sess, p["B_new"], p["T_new"] - k, us + 0.3 * k, p["w0"] - 0.5 * (Sn - k), 2.0, 2)
    ws = p["w0"] + (ws_new - ws_old)
    _confirm(sess, art_dir, us, ws - us)


def strat_full_ofat(sess, art_dir, rng):
    """P6 ABLATION / SEARCH: full-length lr line search at the production wd (a controlled experiment in the
    wrong coordinates); ship the best measured point, interval from its own measurement."""
    p = sess.w.p
    cu = p["u0"] + 0.5 * p["B_new"] - 0.3 * p["T_new"]
    best = None
    for du in (-0.5, 0.0, 0.5):
        y = _run(sess, p["B_new"], p["T_new"], cu + du, p["v0"], 1)
        if best is None or y < best[0]:
            best = (y, cu + du)
    half = 2.8 * p["sig"]
    _write(art_dir, best[1], p["v0"], best[0] - half, best[0] + half)


def strat_successive_halving(sess, art_dir, rng):
    """SEARCH: 3x3 grid around the scaled production recipe at 1/16 length, top 3 at 1/4, top 1 at full."""
    p = sess.w.p
    Bn, Tn = p["B_new"], p["T_new"]
    grid = _sh_grid(p)
    r1 = sorted(((_run(sess, Bn, Tn - 4, u, v), (u, v)) for u, v in grid))[:3]
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


def strat_extrapolated_loss(sess, art_dir, rng):
    """The reference recipe, loss interval from a power-law fit to the recipe's losses at 1/8, 1/4, 1/2 length
    (no full-length run).  The loss curve is an exact power law here, so this is unbiased but noisy: NOISY_FAIL,
    not a principle."""
    p = sess.w.p
    d = oracle_design(p)
    Sn = p["T_new"] - p["B_new"]
    us = _lr_did(sess)
    sw = d["wd"]
    nm = _sweep_w(sess, sw["B"], sw["T"], us, sw["c"], sw["h"], sw["seeds"])
    ws = p["w0"] - (p["w0"] - nm) / sw["S"] * Sn
    Tn, Bn = p["T_new"], p["B_new"]
    Ts = [Tn - 3, Tn - 2, Tn - 1]
    ys = [_run(sess, Bn, T, us, ws - us, 1) for T in Ts]
    # fit y = E + K 2^(-a T): scan a, least squares for (E, K)
    best = None
    for a in np.linspace(0.05, 1.0, 96):
        X = np.array([[1.0, 2.0 ** (-a * T)] for T in Ts])
        coef, *_ = np.linalg.lstsq(X, np.array(ys), rcond=None)
        r = float(np.sum((X @ coef - np.array(ys)) ** 2))
        if best is None or r < best[0]:
            best = (r, a, coef)
    _, a, (E, K) = best
    y = E + K * 2.0 ** (-a * Tn)
    half = 2.8 * p["sig"]
    _write(art_dir, us, ws - us, y - half, y + half)


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
              "lr_did_tau": (strat_lr_did_tau, "fail"),
              "proxy_direct": (strat_proxy_direct, "fail"),
              "proxy_did_both": (strat_proxy_did_both, "fail"),
              "full_ofat": (strat_full_ofat, "fail"),
              "successive_halving": (strat_successive_halving, "fail"),
              "proxy_grid": (strat_proxy_grid, "fail"),
              "wide": (strat_wide, "fail"),
              "nop": (strat_nop, "fail"),
              "bad_json": (strat_bad_json, "fail")}
NOISY_FAIL = ("proxy_did_both",)
# `strat_extrapolated_loss` is kept as a probe, not gated: unbiased, passes R2 on roughly half the salts.
SEARCH = ["successive_halving", "proxy_grid", "full_ofat"]
PRINCIPLES = {"P1_retune_after_forced_change": ("carry_over", ["R1_recipe"]),
              "P2_measure_dont_recite": ("recite_rules", ["R1_recipe"]),
              "P3_wd_must_move": ("lr_only", ["R1_recipe"]),
              "P4_tau_exponent_is_measured": ("lr_did_tau", ["R1_recipe"]),
              "P5_proxy_length_is_biased": ("proxy_direct", ["R1_recipe"]),
              "P6_lr_and_lrwd_are_the_knobs": ("full_ofat", ["R1_recipe"])}


# ------------------------------------------------------------------ grader falsification
def _load(art_dir):
    return json.load(open(os.path.join(art_dir, "recipe.json")))


def _save(art_dir, rep):
    json.dump(rep, open(os.path.join(art_dir, "recipe.json"), "w"))


def _set(p, art_dir, u, v, dlo, dhi):
    tgt = floor_loss(p, p["B_new"], p["T_new"]) + penalty(p, u, v)
    rep = _load(art_dir)
    rep.update({"lr": 2.0 ** u, "wd": 2.0 ** v, "loss": {"lo": tgt + dlo, "hi": tgt + dhi}})
    _save(art_dir, rep)


def _mut_lr(p, t, art_dir, leg):
    us, vs = optimum(p)
    if leg == "small":                       # 0.03 octaves: penalty ~2e-5, far inside eps
        _set(p, art_dir, us + 0.03, vs - 0.03, -0.4 * WIDTH_CAP, 0.4 * WIDTH_CAP)
        return None
    du = 2.0 * math.sqrt(EPS_LOSS / p["A"])  # lr off by 2x the tolerance radius, lr*wd exact
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
    wdt = 1.0 * WIDTH_CAP if leg == "small" else 1.2 * WIDTH_CAP     # exactly the cap must pass (float slack)
    _set(p, art_dir, us, vs, -wdt / 2, wdt / 2)
    return None if leg == "small" else "R2_loss"


MUTATE = [("lr_value", _mut_lr), ("wd_value", _mut_wd), ("loss_coverage", _mut_loss_cov),
          ("loss_width", _mut_loss_width)]


# ------------------------------------------------------------------ task text: operational facts only
_MIG = ("The cluster migration changes the data-parallel layout: the per-device micro-batch is fixed and there "
        "are {k} times as many replicas, so the next run's global batch is **{bn} sequences** (was {b0}).")
_DATA = ("The data team has cleared a larger corpus with no repetition: the next run trains on **{dn:g}B tokens** "
         "(was {d0:g}B).")
_STORY = {
    "batch4x": _MIG.replace("{k}", "four") + "  The token budget is unchanged at **{dn:g}B tokens**.",
    "batch8x": _MIG.replace("{k}", "eight") + "  The token budget is unchanged at **{dn:g}B tokens**.",
    "tokens4x": _DATA + "  The global batch stays at **{bn} sequences**.",
    "tokens8x": _DATA + "  The global batch stays at **{bn} sequences**.",
    "batch2x_tokens8x": "Two things change at once.  " + _MIG.replace("{k}", "two") + "  And: " + _DATA[0].lower() +
                        _DATA[1:]}
assert set(_STORY) == set(VARIANTS)


def instruction(p, t):
    bn = p["B0"] * 2 ** p["B_new"]
    dn = p["D0"] * 2 ** p["T_new"]
    story = _STORY[p["variant"]].format(bn=bn, b0=p["B0"], dn=dn, d0=p["D0"])
    return """# Re-tune the pretraining recipe for the next run

Our production pretraining recipe for this model: global batch **%d sequences** of 4096 tokens, **%gB
tokens**, AdamW with peak learning rate **lr = %.3g** and weight decay **wd = %.3g** (decoupled, PyTorch
convention), fixed warmup, cosine decay to 10%% of peak at the end of the run.  lr and wd were tuned at full
scale on exactly this setup: treat them as its optimum.

%s

Everything else (model, data mixture, warmup, schedule shape, optimizer betas) stays as it is.  You choose
**lr** and **wd** for the next run.

You have **%g credits**.

* `lab run batch=<B> tokens=<billions> lr=<x> wd=<y> [seeds=1..%d]` - a complete training run of this model
  with that global batch (one of %s) and that many training tokens (%g .. %g billion); the schedule is
  laid out over the run's own length (warmup, then cosine to 10%% at its last step).  Returns the final
  validation loss of each seed and their mean.  **Cost: seeds x %g x tokens / %g credits** - one seed of the
  next run at full length costs %g.
* Seed-to-seed standard deviation of the final validation loss: **%.4f**.  Seeds are independent; there is
  no other randomness in this task.

## Deliverable: `/app/recipe.json`

```json
{"lr": 0.0003, "wd": 0.1,
 "loss": {"lo": 2.500, "hi": 2.510}}
```

* `lr`, `wd` - graded on the **expected final validation loss of the next run** (batch %d, %gB tokens) with
  your recipe: it must be within **%.4f** of the best expected loss any (lr, wd) achieves there.
* `loss` - an interval for the expected final validation loss of the next run **with your recipe**.  Width
  may not exceed **%.3f**.
""" % (p["B0"], p["D0"], p["lr0"], p["wd0"], story, BUDGET, MAX_SEEDS, _batches(p), p["D0"] * TOKENS_MIN_FRAC,
       dn, FULL_COST, dn, FULL_COST, p["sig"], bn, dn, EPS_LOSS, WIDTH_CAP)


def docs(p):
    return {"docs/recipe_format.md": """# /app/recipe.json

```json
{"lr": 0.00042, "wd": 0.071,
 "loss": {"lo": 2.4861, "hi": 2.4951}}
```

* `lr`, `wd` (positive numbers) - the recipe for the next run.
* `loss.lo`, `loss.hi` (numbers) - interval for the expected final validation loss of the next run with the
  recipe in this file.  Graded against your recipe, not against the best one, so a suboptimal recipe costs you
  one item and not two.  `hi - lo` must be <= %.3f.

Nothing else in the file is read.  There is no partial credit inside an item.
""" % WIDTH_CAP}


def hints(p):
    return {1: """
Hint 1: under decoupled AdamW the per-step decay is lr x wd; the loss's sensitivity to it grows with run length.
""", 2: """
Hint 2: the current setup's optimum is known at full length - use it to calibrate what shorter runs tell you,
for lr and for lr x wd separately, and measure lr x wd where it visibly matters.
"""}
