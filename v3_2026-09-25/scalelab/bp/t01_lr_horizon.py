"""T01 lr-horizon: a learning-rate rule fitted along the Chinchilla diagonal is applied off-diagonal.

Notebook: LR sweeps at five model sizes, every run at D = 20N (O1: model size and token horizon co-vary).
The lab notes attribute the whole drift of the optimum to model size (a 'muP width bug') and plan a
heavily over-trained production run (1B params, 1T tokens, 4M-token batches) with that rule (O9).
Truth: the optimum depends on both N (weakly) and D (strongly).  Separating them needs off-diagonal
sweeps.  The batch-size dependence of the optimum has zero footprint in the lab (batch is pinned at
0.5M) and is documented only as a range (O5).
"""
import math
import numpy as np
from .. import world as W
from .. import queries as Q
from ..common import draw_card, textbook, box_for, run_rows, rows_xy, c1_from_r, c1_to_r, R_BOX
from .. import fit as F

ID = "t01-lr-horizon"
TITLE = "Learning-rate transfer to an over-trained production run"
CARDS = ["C1", "C2", "C10"]
OBSTACLES = ["O1", "O9", "O5"]
CLAIM = "separate two causes that co-vary in prior runs by designing off-diagonal experiments; state what the lab cannot tell"
EXEMPT_LOAD_BEARING = {"C10": "noise card: shapes tolerances, not answers (declared exemption)"}

NB_SIZES = [1.5e7, 3e7, 6e7, 1.2e8, 2.4e8]
PROD_N, PROD_D, PROD_B, LAB_B = 1e9, 1e12, 4e6, 5e5
GB_RANGE = (0.0, 0.5)


def draw(rng):
    p = {}
    p.update(draw_card(rng, "C1"))
    # both exponents material: the notes' reading (all N) and the muP folklore (all D) must both be wrong
    p.update(draw_card(rng, "C2", {"gN": (0.08, 0.18), "gD": (0.15, 0.35)}))
    p["gB"] = float(rng.uniform(*GB_RANGE))       # zero footprint in the lab (B pinned); key uses the range
    # seed noise of final validation loss: 0.003-0.007 nats at 1e8 params (run-to-run spread of small-model
    # pretraining loss is of this order; the wider library range makes the lab too noisy for this budget)
    p.update(draw_card(rng, "C10", {"sigma0": (0.003, 0.007)}))
    return p


def spec(p):
    return {"knobs": {"N": {"type": "float", "min": 1e7, "max": 3e8},
                      "D": {"type": "float", "min": 2e8, "max": 1e11},
                      "lr": {"type": "float", "min": 1e-4, "max": 5e-2}},
            "fixed": {"B": LAB_B, "wd": 0.1, "wu": 0.01},
            "caps": {"run_flops": 6e18, "total_flops": 1.2e20, "max_runs": 40},
            "metrics": ["loss"]}


def known_unknowns(p):
    return [{"name": "lr-vs-batch exponent", "param": "gB", "range": GB_RANGE,
             "text": "The lab's data loader is pinned to 0.5M-token batches, so nothing you can run here depends on how "
                     "the optimal learning rate moves with batch size. Our stack's documentation only guarantees that, "
                     "between 0.5M and 4M tokens, the optimal peak learning rate scales as (B/0.5M)^g for some fixed "
                     "g in [0, 0.5] (0 = no change, 0.5 = square-root rule); g does not depend on N or D."}]


def _team_fit(rows):
    """What the team did: per-size parabola in ln(lr), then a power law in N."""
    ests = []
    for N in NB_SIZES:
        rr = [r for r in rows if abs(r["config"]["N"] - N) < 1 and r["status"] == "ok"]
        x = np.log([r["config"]["lr"] for r in rr]); y = np.array([r["loss"] for r in rr])
        a, b, c = np.polyfit(x, y, 2)
        ests.append(-b / (2 * a))
    s, i = np.polyfit(np.log(NB_SIZES), ests, 1)
    return -s, i                                                     # eta* ~ exp(i) * N^-s


def notebook(p, sess, rng):
    rows = []
    for N in NB_SIZES:
        D = 20 * N
        c = float(W.eta_star(p, N, D)) * math.exp(rng.normal(0, 0.12))
        for k in (-2, -1, 0, 1, 2):
            lr = float("%.4g" % min(5e-2, c * 2.0 ** k))           # notebook configs are runnable in the lab
            rows.append(sess.execute({"N": N, "D": D, "lr": lr, **spec(p)["fixed"]}, seed=0))
    s, i = _team_fit(rows)
    lr_notes = math.exp(i) * PROD_N ** (-s)
    notes = f"""# Lab notes (pretraining team)

- All five LR sweeps below were run at the Chinchilla ratio (D = 20 N), batch 0.5M tokens, seed 0.
- Per-size optimum from a parabola fit in ln(lr), then a power law across sizes:
  **eta*(N) = {math.exp(i):.4g} * N^-{s:.3f}**.  Our muP port still has the known width bug, so the optimum drifts
  with width; once that is accounted for, muP makes the optimum independent of training length.
- Production plan: N = 1e9, D = 1e12 tokens (1000 tokens/param), batch 4M tokens.
  Planned peak LR from the rule above: **lr_prod = {lr_notes:.4g}**.
- Loss is fairly flat near the optimum, so a factor-of-two error in LR is cheap (see sweeps).
"""
    return rows, notes, {"lr_notes": lr_notes, "team_slope": s, "opts": _options(p, rng),
                         "q5_drop": _q5_threshold(p, rng), "x6": _q6_value(p, lr_notes, rng)}


# Probe values in verdict questions are drawn independently of the truth and redrawn only if they fall
# within a small band around the verdict boundary (a question whose answer flips inside the measurement
# error is ill-posed).  The band is narrow, so the redraw leaks almost nothing about the key.
Q5_BAND, Q6_BAND = 0.06, 0.08


def _q5_threshold(p, rng):
    d = 1 - 10 ** (-p["gN"])
    for _ in range(50):
        thr = float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]))
        if abs(thr - d) >= Q5_BAND:
            return thr
    return thr


def _q6_value(p, lr_notes, rng):
    edges = [math.log10(float(W.eta_star(dict(p, gB=g), PROD_N, PROD_D, PROD_B))) for g in GB_RANGE]
    for _ in range(50):
        x = float("%.3g" % (lr_notes * 2.0 ** rng.uniform(-1.5, 1.5)))
        if min(abs(math.log10(x) - e) for e in edges) >= Q6_BAND:
            return x
    return x


def _pen(p, N, D, lr, B=LAB_B):
    u = math.log(lr / float(W.eta_star(p, N, D, B)))
    return (p["k_lo"] if u < 0 else p["k_hi"]) * u * u


def items(p, ctx, tol=None):
    tol = tol or {}
    lrn = ctx["lr_notes"]
    l10 = lambda x: math.log10(float(x))
    e1 = l10(W.eta_star(p, PROD_N, 2e10))
    e2 = l10(W.eta_star(p, PROD_N, PROD_D))
    pb = dict(p); lo_hi = []
    for g in GB_RANGE:
        pb["gB"] = g; lo_hi.append(l10(W.eta_star(pb, PROD_N, PROD_D, PROD_B)))
    grid = np.linspace(*GB_RANGE, 101)
    x6 = ctx["x6"]
    above = [x6 > float(W.eta_star(dict(p, gB=g), PROD_N, PROD_D, PROD_B)) for g in grid]
    losses = {k: _pen(p, PROD_N, Q7_D, v) for k, v in ctx["opts"].items()}
    ratio10 = 10 ** (-p["gN"]); thr = ctx["q5_drop"]     # threshold drawn per instance so the key varies
    Lprod = float(W.core_loss(p, {"N": PROD_N, "D": PROD_D, "B": LAB_B, "lr": float(W.eta_star(p, PROD_N, PROD_D))}))
    its = [
        Q.point("q1", "log10 of the loss-minimising peak learning rate for N=1e9, D=2e10 tokens, batch 0.5M tokens.",
                "log10(lr)", e1, tol.get("q1", 0.06), cards=["C2"], obstacles=[], floor=0.03),
        Q.point("q2", "log10 of the loss-minimising peak learning rate for N=1e9, D=1e12 tokens, batch 0.5M tokens.",
                "log10(lr)", e2, tol.get("q2", 0.06), cards=["C2"], obstacles=["O1", "O9"], floor=0.03),
        Q.point("q3", "Excess final loss (nats/token) of training N=1e9, D=1e12, batch 0.5M at the notes' planned lr_prod "
                      "= %.4g instead of at the loss-minimising learning rate." % lrn,
                "nats", _pen(p, PROD_N, PROD_D, lrn), tol.get("q3", 0.01), cards=["C2"], obstacles=["O9"], floor=0.005),
        Q.interval("q4", "log10 of the loss-minimising peak learning rate for the production run as planned: "
                         "N=1e9, D=1e12 tokens, batch 4M tokens.", "log10(lr)", min(lo_hi), max(lo_hi),
                   tol.get("q4", 0.06), cards=["C2"], obstacles=["O5", "O9"], floor=0.03),
        Q.verdict("q5", "Holding the number of training tokens fixed (batch 0.5M), making the model 10x larger lowers its "
                        "loss-minimising learning rate by at least %d%%." % round(100 * thr), [ratio10 <= 1 - thr],
                  cards=["C2"], obstacles=["O1"]),
        Q.verdict("q6", "A colleague proposes peak lr = %.3g for the production run as planned (N=1e9, D=1e12, batch 4M). "
                        "This value is above that run's loss-minimising learning rate." % x6, above, cards=["C2"], obstacles=["O5"]),
        Q.decision("q7", "Which peak learning rate gives the lowest final loss for N=1e9, D=2e11, batch 0.5M? Options: "
                   + ", ".join("%s: %.4g" % (k, v) for k, v in ctx["opts"].items()),
                   list(ctx["opts"]), losses, tol.get("q7", 0.004), cards=["C2"], obstacles=["O9"]),
        Q.point("q8", "Final validation loss (nats/token) of N=1e9, D=1e12, batch 0.5M trained at its loss-minimising "
                      "learning rate.", "nats", Lprod, tol.get("q8", 0.02), cards=["C1"], obstacles=["O3"], floor=0.005),
    ]
    return its


Q7_D = 2e11     # q7 is asked at a horizon other than q2's so its option grid cannot be used to snap q2


def _options(p, rng):
    e = float(W.eta_star(p, PROD_N, Q7_D))
    off = rng.uniform(-0.3, 0.3); pos = int(rng.integers(0, 4))
    vals = [e * 2.0 ** (off + k - pos) for k in range(4)]
    return {"A": float("%.3g" % vals[0]), "B": float("%.3g" % vals[1]), "C": float("%.3g" % vals[2]), "D": float("%.3g" % vals[3])}


# ---------------------------------------------------------------- model used by oracle and rivals
FIT_KEYS = ["E", "rN", "alpha", "rD", "beta", "eta0", "gN", "gD", "k_lo", "k_hi"]   # C1 in r-form (common.c1_from_r)
TB_C1 = c1_to_r(textbook(["C1"]))


def _model(cfg):
    return lambda q: W.core_loss(W.full(c1_from_r(q)), dict(cfg, B=LAB_B))


def _stage_init(rows):
    """Robust start: per-(N,D) parabola minima in ln(lr), then a linear fit of ln(eta*) on ln N, ln D."""
    groups = {}
    for r in rows:
        if r["status"] == "ok":
            groups.setdefault((round(r["config"]["N"]), round(r["config"]["D"])), []).append(r)
    X, Y, K = [], [], []
    for (N, D), rr in groups.items():
        if len(rr) < 4:
            continue
        x = np.log([r["config"]["lr"] for r in rr]); y = np.array([r["loss"] for r in rr])
        a, b, c = np.polyfit(x, y, 2)
        if a <= 0 or not (x.min() - 0.5 < -b / (2 * a) < x.max() + 0.5):
            continue
        X.append([1.0, -np.log(N / 1e8), -np.log(D / 2e9)]); Y.append(-b / (2 * a)); K.append(a)
    init = {}
    if len(X) >= 3:
        sol = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]
        if np.linalg.matrix_rank(np.array(X)) == 3:
            init.update({"eta0": float(np.exp(sol[0])), "gN": float(sol[1]), "gD": float(sol[2])})
        else:
            init.update({"eta0": float(np.exp(sol[0]))})
        k = float(np.median(K)); init.update({"k_lo": k * 0.7, "k_hi": k * 1.4})
    return init


def fit_rows(rows, rng, fixed=None, keys=None, n_starts=10, init_from=None):
    keys = [k for k in (keys or FIT_KEYS) if k not in (fixed or {})]
    cfg, y, ok = rows_xy(rows, ["N", "D", "lr"])
    w = (cfg["N"] / 1e8) ** 0.3
    box = box_for(keys, widen=1.6, overrides=dict(R_BOX, gN=(-0.1, 0.4), gD=(-0.1, 0.6)))
    fx = {"gB": 0.0}; fx.update(fixed or {})
    init = dict(TB_C1)
    init.update({k: v for k, v in (init_from or _stage_init(ok)).items() if k in keys})
    init = {k: min(max(v, (math.exp(box.lo[i]) if box.log[i] else box.lo[i]) * 1.001),
                   (math.exp(box.hi[i]) if box.log[i] else box.hi[i]) * 0.999)
            for i, k in enumerate(box.names) for v in [init.get(k, None)] if v is not None}
    init = init if len(init) == len(box.names) else None
    return c1_from_r(F.fit(_model(cfg), box, fx, y, w, rng, n_starts=n_starts, init=init)[0])


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


ORACLE_POINTS = ((1.5e8, 2.2e9), (5e7, 1.5e10), (2e7, 5e10))


def oracle_design(rows_nb):
    """Stage 1: coarse off-diagonal sweeps centred on the notes' rule with a generic horizon correction."""
    s, i = _team_fit(rows_nb)
    reqs = []; seed = 100
    for N, D in ORACLE_POINTS:
        c = math.exp(i) * N ** (-s) * (D / (20 * N)) ** (-0.25)
        for k in (-2, -1, 0, 1, 2):
            reqs.append({"N": N, "D": D, "lr": min(5e-2, c * 2.0 ** k), "seed": seed}); seed += 1
    return reqs


def oracle(sess, rows_nb, ctx, rng, drop=None):
    """Two-stage: coarse sweeps, fit, then replicate points at +-0.5 and +-1 ln around each fitted optimum."""
    own = run_rows(sess, oracle_design(rows_nb))
    ph = fit_rows(rows_nb + own, rng, fixed=drop, n_starts=6)
    seed = 200; reqs = []
    for N, D in ORACLE_POINTS:
        e = float(W.eta_star(W.full(ph), N, D))
        for u in (-1.0, -0.4, 0.4, 1.0):
            reqs.append({"N": N, "D": D, "lr": float(min(5e-2, e * math.exp(u))), "seed": seed}); seed += 1
    own += run_rows(sess, reqs)
    ph = fit_rows(rows_nb + own, rng, fixed=drop, init_from=c1_to_r(ph))
    return answers_from(ph, ctx), ph


def rivals(p, rows_nb, ctx, rng):
    out = {}
    tb = W.full(dict(textbook(CARDS)))
    out["B0_textbook"] = answers_from(tb, ctx)
    # B1: the notes' reading - all drift attributed to N, no horizon effect; curvature from the notebook
    ph = fit_rows(rows_nb, rng, fixed={"gD": 0.0}, keys=[k for k in FIT_KEYS if k != "gD"])
    out["B1_notebook_naive"] = answers_from(ph, ctx)
    out["B2_notebook_allcards"] = answers_from(fit_rows(rows_nb, rng), ctx)
    # B5: notebook + the literature's reading (drift is all horizon, gN = 0) - the 'well-read' shortcut
    out["B5_notebook_horizon_only"] = answers_from(fit_rows(rows_nb, rng, fixed={"gN": 0.0}), ctx)
    return out


def rival_designs(p, rows_nb, rng):
    """Designs that spend the same budget differently (evaluated with the full correct model)."""
    reqs = []
    for j in range(30):
        N = float(np.exp(rng.uniform(np.log(1e7), np.log(3e8))))
        D = float(np.exp(rng.uniform(np.log(2e8), np.log(min(1e11, 6e18 / (6 * N))))))
        e = float(W.eta_star(W.full(textbook(CARDS)), N, D))
        reqs.append({"N": N, "D": D, "lr": float(min(5e-2, e * np.exp(rng.uniform(-2, 2)))), "seed": 500 + j})
    diag = []                                            # strictly on the notebook's diagonal D = 20 N
    for j, N in enumerate([2e7, 4e7, 8e7, 1.2e8, 1.6e8]):
        e = float(W.eta_star(W.full(textbook(CARDS)), N, 20 * N))
        for k in (-1, 0, 1):
            diag.append({"N": N, "D": 20 * N, "lr": e * 2.0 ** k, "seed": 700 + 3 * j + k})
    return {"B3_random_design": reqs, "B4_more_diagonal": diag}


DROP = {"C2:gD": {"gD": 0.0}, "C2:gN": {"gN": 0.0}, "C1:textbook": dict(TB_C1)}
# B3: a library-aware random design is reported, not required to fail (it samples off the diagonal).
# B2 / B4: the notebook's diagonal alone does not identify the N/D split, so a free fit lands at an
# arbitrary point of the gN+gD valley; whether that point happens to be close is luck, not a strategy.
# They are reported (their kill rate across instances is in the suite log), not gated.  The *strategies*
# that commit to a point of the valley - the notes' reading (B1, gD=0) and the literature's (B5, gN=0) -
# are gated.
INFO_RIVALS = {"B3_random_design", "B2_notebook_allcards", "B4_more_diagonal"}


def public_values(ctx):
    return [ctx["lr_notes"], ctx["team_slope"], ctx["q5_drop"], ctx["x6"]] + list(ctx["opts"].values())
