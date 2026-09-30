"""T07 checkpoint-ledger: a scaling law built from intermediate checkpoints (O7), carried to production.

Notebook: the team trained one long cosine run per size and logged intermediate checkpoints.  They fitted
L(N, D) treating every checkpoint as if it were a finished run of f*D tokens (the Kaplan-style shortcut).
Under C11 a checkpoint at fraction f carries a schedule penalty ca*r(f)^zeta that shrinks as the learning
rate decays, so early checkpoints look worse than finished short runs and the fitted data exponent is
biased; the team's compute-optimal allocation is wrong (O7).  Even finished cosine runs carry a residual
penalty ca*rmin^zeta, which a fit on cosine finals silently absorbs into E; the production question is
about a WSD run (no residual), so the offset must be measured (WSD cooldown branches give unbiased
finished-run losses cheaply; stable-phase WSD checkpoints give ca directly).  The team's WSD pilot looks
worse than cosine at every mid-run checkpoint (the stable phase carries the full penalty) although its
finished loss is lower - the notes misread this (O1).  Production may be stopped early at an unknown
fraction of its steps (known unknown, O5): the loss at the stop is a set.
"""
import math
import numpy as np
from .. import world as W
from .. import queries as Q
from ..common import draw_card, textbook, box_for, run_rows, c1_from_r, c1_to_r, R_BOX
from .. import fit as F

ID = "t07-checkpoint-ledger"
TITLE = "Scaling law from intermediate checkpoints, carried to a production run that may be stopped early"
CARDS = ["C1", "C11", "C10"]
OBSTACLES = ["O7", "O1", "O5", "O9"]
CLAIM = ("recognise that intermediate checkpoints and finished runs are different measurements, infer how the "
         "gap depends on the schedule, design unbiased measurements within a tight budget, and carry both the "
         "scaling law and the schedule effect to a production run with an unknown stopping point")
EXEMPT_LOAD_BEARING = {"C10": "noise card: shapes tolerances, not answers (declared exemption)"}

NB_SIZES = [2e7, 5e7, 1.2e8, 3e8]
NB_TPP = 14.0                                  # notebook runs: D = 14 N (the largest fits the per-run cap)
NB_CKPTS = [0.1, 0.2, 0.3, 0.5, 0.7]
PILOT_N = 1.2e8
PROD_N, PROD_D = 3e9, 6e10
C_PROD = 6 * PROD_N * PROD_D
FSTOP = (0.5, 0.9)                             # known unknown: production stopping point
Q5_PAIRS = [(0.5, 0.3), (0.6, 0.4), (0.7, 0.5), (0.8, 0.6), (0.6, 0.3), (0.7, 0.4), (0.8, 0.5),
            (0.9, 0.6), (0.9, 0.7), (0.5, 0.35), (0.75, 0.5), (0.85, 0.6)]
Q5_BAND = 0.008
OFFSET_MIN = 0.012                             # cosine-final residual ca*rmin^zeta: the offset trap must bite


def rcos(f, rmin=0.1):
    return rmin + (1 - rmin) * 0.5 * (1 + math.cos(math.pi * f))


def draw(rng):
    p = {}
    p.update(draw_card(rng, "C1"))
    p.update(draw_card(rng, "C10", {"sigma0": (0.003, 0.007)}))
    for _ in range(400):
        p.update(draw_card(rng, "C11"))
        if p["ca"] * 0.1 ** p["zeta"] >= OFFSET_MIN:
            break
    p["f_stop"] = float(rng.uniform(*FSTOP))   # scenario unknown: not read by the world; keys use the range
    return p


def spec(p):
    return {"knobs": {"N": {"type": "float", "min": 1e7, "max": 3e8, "log": True},
                      "D": {"type": "float", "min": 2e8, "max": 1e11, "log": True},
                      "sched": {"type": "choice", "values": ["cosine", "wsd"]}},
            "fixed": {"B": 5e5},
            "caps": {"run_flops": 8e18, "total_flops": 2e19, "max_runs": 10},
            "metrics": ["loss"], "checkpoints": True}


def LAB_EXTRA(w):
    return ["",
            "Schedules (`sched`): **cosine** - after a short warmup the learning rate follows a half cosine from "
            "the peak down to 10% of the peak at the last step.  **wsd** (warmup-stable-decay) - the learning rate "
            "stays at the peak and decays linearly to 0 over the last 20% of the steps.  Every run uses a peak "
            "learning rate tuned by the lab for its N and D, so there is no lr knob.",
            "",
            "Intermediate measurements (both optional, give them as lists of fractions in [0.02, 1], at most 20 "
            "each; on the command line write `ckpts=0.25,0.5` or, for a single fraction, `ckpts=0.5,`):",
            "",
            "- `ckpts`: the validation loss of the run's own checkpoint after a fraction f of its D tokens, "
            "evaluated as the run passes that point.  Free (no extra FLOPs).",
            "- `cooldowns` (wsd only): for each fraction f the lab forks the run at 0.8*f*D tokens (still in the "
            "stable phase) and decays the learning rate linearly to 0 over the next 0.2*f*D tokens; the branch's "
            "final validation loss is reported.  Extra cost per branch: 6*N*0.2*f*D FLOPs (the main run is charged "
            "in full as usual).",
            "",
            "Results list them under `checkpoints` and `cooldown_branches` (each with `frac`, `tokens`, `loss`).  "
            "The notebook's intermediate measurements are in `notebook/checkpoints.json`."]


def known_unknowns(p):
    return [{"name": "production stopping point", "param": "f_stop", "range": FSTOP,
             "text": "The production run is planned as N=3e9 on D=6e10 tokens with the cosine schedule.  Its cluster "
                     "reservation may end before the run completes: the run will then stop at a fraction f_stop of its "
                     "planned steps, somewhere between 0.5 and 0.9 (the date is not known yet), with no chance to "
                     "change the schedule.  Nothing in this lab depends on f_stop.  Questions about the run 'if it is "
                     "stopped' must cover the whole range."}]


# ------------------------------------------------------------------------------------------ notebook
def _nb_run(sess, req):
    cfg, seed, ex = sess.validate(req)                 # same normalisation as a lab request, so replay is exact
    return sess.execute(cfg, seed, ex)


def points(rows):
    """Flatten rows into measurement points: finals, checkpoints, cooldown branches."""
    N, D, f, s, y = [], [], [], [], []
    for r in rows:
        if r.get("status") != "ok" or r.get("loss") is None:
            continue
        c = r["config"]; sc = {"cosine": 0, "wsd": 1, "constant": 2}[c["sched"]] if isinstance(c["sched"], str) else int(c["sched"])
        N.append(c["N"]); D.append(c["D"]); f.append(1.0); s.append(sc); y.append(r["loss"])
        for ck in r.get("checkpoints") or []:
            N.append(c["N"]); D.append(c["D"]); f.append(ck["frac"]); s.append(sc); y.append(ck["loss"])
        for cb in r.get("cooldown_branches") or []:
            N.append(c["N"]); D.append(cb["frac"] * c["D"]); f.append(1.0); s.append(1); y.append(cb["loss"])
    return {k: np.array(v, float) for k, v in (("N", N), ("D", D), ("f", f), ("sched", s))}, np.array(y, float)


def _L(p, N, D):
    return float(W.core_loss(p, {"N": N, "D": D}))


def _team_fit(rows, rng):
    """The team's shortcut: every checkpoint of a cosine run is a finished run of f*D tokens (no penalty)."""
    cos = [r for r in rows if r["config"]["sched"] == "cosine"]
    pts, y = points(cos)
    pts = {"N": pts["N"], "D": pts["D"] * pts["f"], "f": np.ones_like(pts["f"]), "sched": np.full_like(pts["f"], 1.0)}
    return _fit_pts(pts, y, rng, fixed={"ca": 0.0, "zeta": 1.0})


def _opt_N(p, grid=None):
    grid = np.exp(np.linspace(np.log(1e8), np.log(1e12), 801)) if grid is None else grid
    L = [_L(p, n, C_PROD / (6 * n)) for n in grid]
    return float(grid[int(np.argmin(L))])


def notebook(p, sess, rng):
    rows = []
    for N in NB_SIZES:
        rows.append(_nb_run(sess, {"N": N, "D": NB_TPP * N, "sched": "cosine", "ckpts": list(NB_CKPTS), "seed": 0}))
    rows.append(_nb_run(sess, {"N": PILOT_N, "D": NB_TPP * PILOT_N, "sched": "wsd", "ckpts": list(NB_CKPTS), "seed": 0}))
    team = _team_fit(rows, rng)
    n_team = _opt_N(W.full(team))
    # decision options: four sizes spaced by 2x around the true optimum, the true one at a random position
    n_true = _opt_N(p)
    j = int(rng.integers(0, 4)); u = float(rng.uniform(-0.3, 0.3))
    opts = {}
    for k, lab in enumerate("ABCD"):
        opts[lab] = float("%.2g" % (n_true * 2.0 ** (k - j + u)))
    fa, fb = _q5_pair(p, rng)
    cz = rows[2]; pw = rows[4]
    ck_c = {c["frac"]: c["loss"] for c in cz["checkpoints"]}
    ck_w = {c["frac"]: c["loss"] for c in pw["checkpoints"]}
    worse = sum(1 for fr in NB_CKPTS if ck_w[fr] > ck_c[fr])
    fin_gap = pw["loss"] - cz["loss"]
    L_team = _L(W.full(team), PROD_N, PROD_D)
    notes = f"""# Lab notes (scaling team)

- Runs 1-4: one cosine run per model size (N = 2e7, 5e7, 1.2e8, 3e8), each on D = 14 N tokens, seed 0, with
  checkpoints evaluated at 10, 20, 30, 50 and 70% of training (notebook/checkpoints.json).  Every checkpoint
  is a (N, tokens-seen, loss) point, so four runs give us 24 points for the scaling law at the price of four.
- Fit of L(N, D) = E + A/N^alpha + B/D^beta to all 24 points: E = {team['E']:.3f}, alpha = {team['alpha']:.3f},
  beta = {team['beta']:.3f}.  At the production compute 6*N*D = {C_PROD:.3g} FLOPs this puts the compute-optimal
  size at N = {n_team:.2g}; for the planned production run (N = 3e9, D = 6e10) it predicts a final loss of
  {L_team:.3f}.
- Run 5: WSD pilot at N = 1.2e8 (same D, same checkpoints).  It was worse than the cosine run at {worse} of the
  5 checkpoints; the final loss differed by {fin_gap:+.4f}, within what we would expect from seeds.  WSD
  buys us nothing; production stays on cosine.
- Production plan: N = 3e9, D = 6e10, cosine.  The reservation may be cut short (see the manual).
"""
    ctx = {"opts": opts, "n_team": n_team, "team": {k: float(team[k]) for k in ("E", "alpha", "beta")},
           "L_team": L_team, "worse": worse, "fin_gap": float(fin_gap), "q5": (fa, fb)}
    return rows, notes, ctx


def _q5_margin(p, fa, fb):
    """(loss of the production cosine run at checkpoint fa) - (final loss of a finished WSD run on fb*D_prod)."""
    ck = _L(p, PROD_N, fa * PROD_D) + p["ca"] * rcos(fa, p["rmin"]) ** p["zeta"]
    return ck - _L(p, PROD_N, fb * PROD_D)


def _q5_pair(p, rng):
    ok = [pr for pr in Q5_PAIRS if abs(_q5_margin(p, *pr)) >= Q5_BAND]
    if not ok:
        return Q5_PAIRS[int(rng.integers(len(Q5_PAIRS)))]
    return ok[int(rng.integers(len(ok)))]


# ------------------------------------------------------------------------------------------ items
def _stop_delta(p, f):
    """Loss of the production cosine run stopped at f, minus its final loss had it completed."""
    return (_L(p, PROD_N, f * PROD_D) - _L(p, PROD_N, PROD_D)
            + p["ca"] * (rcos(f, p["rmin"]) ** p["zeta"] - rcos(1.0, p["rmin"]) ** p["zeta"]))


def items(p, ctx, tol=None):
    tol = tol or {}
    opts = ctx["opts"]
    losses = {k: _L(p, n, C_PROD / (6 * n)) for k, n in opts.items()}
    grid = np.linspace(FSTOP[0], FSTOP[1], 41)
    sd = [_stop_delta(p, f) for f in grid]
    fa, fb = ctx["q5"]
    return [
        Q.decision("q1", "Among finished runs at the production compute 6*N*D = %.3g FLOPs (D = C/(6N), same schedule "
                         "for all), which model size gives the lowest final loss?  Options: " % C_PROD +
                   ", ".join("%s: N=%.2g" % (k, v) for k, v in opts.items()), list(opts), losses,
                   tol.get("q1", 0.004), cards=["C1"], obstacles=["O7", "O9"]),
        Q.point("q2", "Final validation loss (nats/token) of the production model N=3e9 trained on D=6e10 tokens with "
                      "the WSD schedule, run to completion.", "nats",
                _L(p, PROD_N, PROD_D), tol.get("q2", 0.01), cards=["C1", "C11"], obstacles=["O7", "O9"], floor=0.004),
        Q.point("q3", "The production cosine run (N=3e9, D=6e10) is evaluated at its checkpoint after 40% of its tokens.  "
                      "By how much (nats/token) is that checkpoint's loss higher than the final loss of a separate WSD run "
                      "of the same model trained to completion on 0.4*6e10 = 2.4e10 tokens?", "nats",
                p["ca"] * rcos(0.4, p["rmin"]) ** p["zeta"], tol.get("q3", 0.005), cards=["C11"], obstacles=["O7"],
                floor=0.002),
        Q.interval("q4", "If the production cosine run is stopped early (f_stop as documented), by how much (nats/token) "
                         "will its loss at the stop exceed the final loss it would have reached had it completed?  "
                         "Give the set over the documented range of f_stop.", "nats",
                   min(sd), max(sd), tol.get("q4", 0.005), cards=["C1", "C11"], obstacles=["O5", "O7", "O9"],
                   floor=0.003),
        Q.verdict("q5", "The production cosine run's checkpoint after %d%% of its tokens has a lower loss than a finished "
                        "WSD run of the same model trained to completion on %d%% of the production tokens."
                  % (round(100 * fa), round(100 * fb)),
                  [_q5_margin(p, fa, fb) < 0], cards=["C1", "C11"], obstacles=["O7", "O9"]),
    ]


# ------------------------------------------------------------------------------------------ fitting
FIT_KEYS = ["E", "rN", "alpha", "rD", "beta", "ca", "zeta"]
TB = c1_to_r(textbook(["C1", "C11"]))


def _fit_pts(pts, y, rng, fixed=None, keys=None, n_starts=8, init_from=None):
    keys = [k for k in (keys or FIT_KEYS) if k not in (fixed or {})]
    w = (pts["N"] / 1e8) ** 0.3
    box = box_for(keys, widen=1.6, overrides=dict(R_BOX, ca=(0.01, 0.8), zeta=(0.3, 2.5)))
    init = dict(TB); init.update(init_from or {})
    init = {k: init[k] for k in box.names if k in init}
    init = {k: min(max(v, (math.exp(box.lo[i]) if box.log[i] else box.lo[i]) * 1.001 if v > 0 else v),
                   (math.exp(box.hi[i]) if box.log[i] else box.hi[i]) * 0.999)
            for i, k in enumerate(box.names) for v in [init[k]]}
    fx = dict(fixed or {})

    def model(q):
        return W.val_loss(W.full(c1_from_r(q)), pts)
    return c1_from_r(F.fit(model, box, fx, y, w, rng, n_starts=n_starts, init=init)[0])


def fit_rows(rows, rng, fixed=None, keys=None, n_starts=8, init_from=None):
    pts, y = points(rows)
    return _fit_pts(pts, y, rng, fixed=fixed, keys=keys, n_starts=n_starts, init_from=init_from)


def answers_from(ph, ctx, collapse=False):
    its = items(W.full(ph), dict(ctx))
    out = {}
    for it in its:
        if it["kind"] in ("point", "set"):
            lo, hi = it["key"]["lo"], it["key"]["hi"]
            if collapse:
                lo = hi = (lo + hi) / 2
            out[it["id"]] = {"lo": lo, "hi": hi}
        elif it["kind"] == "decision":
            out[it["id"]] = {"choice": it["key"]["choice"]}
        else:
            out[it["id"]] = {"verdict": it["key"]["verdict"]}
    return out


WSD_CK = [0.25, 0.5, 0.7, 0.85, 0.9, 0.95]
WSD_CD = [0.125, 0.25, 0.5]


def oracle_design(rows_nb):
    """Four WSD runs spanning the lab's sizes, each with stable/cooldown-phase checkpoints and three cooldown
    branches: 16 unbiased finished-run losses, a direct handle on ca (stable checkpoint vs branch at the same f)
    and on the shape zeta (checkpoints inside the decay).  About 1.7e19 of the 2e19 budget."""
    reqs = []; s = 100
    for N, D in ((3e8, 3e9), (1.5e8, 5e9), (6e7, 8e9), (2e7, 1.5e10)):
        reqs.append({"N": N, "D": D, "sched": "wsd", "ckpts": list(WSD_CK), "cooldowns": list(WSD_CD), "seed": s}); s += 1
    return reqs


def oracle(sess, rows_nb, ctx, rng, drop=None):
    own = run_rows(sess, oracle_design(rows_nb))
    keys = [k for k in FIT_KEYS if k not in (drop or {})]
    ph = fit_rows(rows_nb + own, rng, fixed=drop, keys=keys)
    ph = fit_rows(rows_nb + own, rng, fixed=drop, keys=keys, init_from=c1_to_r(ph), n_starts=4)
    return answers_from(ph, ctx), ph


def _grid_session(p, tag):
    from ..lab import Session
    return Session(W.full(p), spec(p), "rival/" + tag)


def rivals(p, rows_nb, ctx, rng):
    out = {}
    out["B0_textbook"] = answers_from(dict(textbook(CARDS)), ctx)
    # B1: the notes' reading - checkpoints are finished runs, schedules do not matter
    out["B1_notes_reading"] = answers_from(dict(_team_fit(rows_nb, rng), ca=0.0), ctx)
    # B2: an IsoFLOP-style grid of finished cosine runs, fitted with a plain Chinchilla law (no schedule term):
    # right allocation, but the cosine residual lands in E and the checkpoint questions get zero penalty
    sess = _grid_session(p, "B2")
    grid = []
    for C, Ns in ((1e18, (2e7, 6e7, 1.5e8)), (3e18, (4e7, 1e8, 3e8)), (6e18, (6e7, 1.5e8, 3e8))):
        for N in Ns:
            grid.append({"N": N, "D": C / (6 * N), "sched": "cosine", "seed": 300 + len(grid)})
    own = run_rows(sess, grid)
    fin = [r for r in rows_nb + own if r["config"]["sched"] == "cosine"]
    fin = [dict(r, checkpoints=None) for r in fin]
    out["B2_cosine_finals_no_schedule"] = answers_from(fit_rows(fin, rng, fixed={"ca": 0.0, "zeta": 1.0}), ctx)
    # B3: schedule penalty proportional to the learning rate (zeta = 1, the linear textbook form), rest fitted
    sess3 = _grid_session(p, "B3")
    own3 = run_rows(sess3, oracle_design(rows_nb))
    out["B3_linear_in_lr"] = answers_from(fit_rows(rows_nb + own3, rng, fixed={"zeta": 1.0}), ctx)
    # B4: the same grid as B2 plus the notebook checkpoints, fitted with the full law (a legitimate route)
    out["B4_cosine_grid_full_law"] = answers_from(fit_rows(rows_nb + own, rng), ctx)
    return out


def rival_designs(p, rows_nb, rng):
    return {}


DROP = {"C11:ca": {"ca": 0.0, "zeta": 1.0}, "C1:textbook": {k: TB[k] for k in ("E", "rN", "alpha", "rD", "beta")}}
# B3 (linear-in-lr penalty) is right exactly when the world's zeta is near 1; B4 is a legitimate alternative
# design (finished cosine grid + the notebook's checkpoints under the full law) - both reported, not gated.
INFO_RIVALS = {"B3_linear_in_lr", "B4_cosine_grid_full_law"}


def wellposed(w):
    ctx = w["ctx"]; p = w["pf"]
    its = items(p, ctx)
    q1 = [it for it in its if it["id"] == "q1"][0]
    team_choice = min(ctx["opts"], key=lambda k: abs(math.log(ctx["opts"][k] / ctx["n_team"])))
    if team_choice == q1["key"]["choice"]:
        return False, "the team's biased allocation already picks the right option (O7 does not bite on q1)"
    if abs(_q5_margin(p, *ctx["q5"])) < Q5_BAND:
        return False, "q5 margin below band"
    return True, "team picks %s, truth %s; q5 margin %.4f" % (team_choice, q1["key"]["choice"], _q5_margin(p, *ctx["q5"]))


def public_values(ctx):
    t = ctx["team"]
    return ([t["E"], t["alpha"], t["beta"], ctx["n_team"], ctx["L_team"], ctx["fin_gap"], C_PROD, PROD_N, PROD_D]
            + list(ctx["opts"].values()) + list(ctx["q5"]) + list(FSTOP) + NB_SIZES + NB_CKPTS)
