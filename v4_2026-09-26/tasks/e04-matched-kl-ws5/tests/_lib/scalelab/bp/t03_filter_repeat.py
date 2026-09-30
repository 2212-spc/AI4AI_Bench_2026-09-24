"""T03 filter-repeat: two effects cancel in the team's ablation (masking), then diverge at production.

Notebook: the team's quality-filter ablation ran on a small raw subsample for several epochs.  Filtering
raises the value of each token (C7) but shrinks the unique pool, so the ablation repeats data more (C6).
At the ablation's epoch count the two effects cancel by construction (O2): the notes conclude 'filtering
does nothing' and plan a 5-10 epoch production run without it.  At production the balance is different
(the regime is reachable in the lab only if one emulates scarcity with a small `sub`; O3).  Separating
the two effects needs a fresh-data filter sweep and a repetition sweep.  The next corpus snapshot's
size is documented only as a range (O5): production quantities that depend on it are sets.
"""
import math
import numpy as np
from .. import world as W
from .. import queries as Q
from ..common import draw_card, textbook, box_for, run_rows, rows_xy, c1_from_r, c1_to_r, R_BOX, loguni
from .. import fit as F

ID = "t03-filter-repeat"
TITLE = "Quality filtering for a data-constrained production run"
CARDS = ["C1", "C6", "C7", "C10"]
OBSTACLES = ["O2", "O3", "O5"]
CLAIM = ("decompose two effects that cancel in the team's ablation by designing experiments that isolate each; "
         "carry a documented range through a nonlinear model")
EXEMPT_LOAD_BEARING = {"C10": "noise card: shapes tolerances, not answers (declared exemption)"}

U0 = 2e11                                  # raw unique tokens of the current corpus (stated in the manual)
PROD_N, PROD_D = 1e9, 2e11
UP_RANGE = (2e10, 1e11)                    # next snapshot, raw unique tokens (known unknown)
UP_Q = 4e10                                # snapshot size stated in the point / decision questions
NB_SIZES = [1.5e7, 3e7, 6e7, 1.2e8, 2.4e8]
NB_N, NB_D, NB_QS = 5e7, 8e9, (0.0, 0.3, 0.6)
Q_OPTS = {"A": 0.0, "B": 0.3, "C": 0.6, "D": 0.85}
Q6_OPTS = (0.3, 0.6, 0.85)
MASK_EPOCHS = (3.0, 12.0)                  # the ablation's epoch count at q=0 must be a plausible ablation


def _G(x, Rs):
    return x if x <= 1 else 1 + Rs * (1 - math.exp(-(x - 1) / Rs))


def _ratio(e, q, p):
    """Deff(q)/Deff(0) for a run of D = e*P tokens on a raw subsample of P tokens."""
    m = 1 + p["mu"] * q ** p["nu"]
    return m * (1 - q) * _G(e / (1 - q), p["Rs"]) / _G(e, p["Rs"])


def mask_epochs(p, q=0.6):
    lo, hi = 0.05, 500.0
    if _ratio(hi, q, p) > 1:
        return None
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if _ratio(mid, q, p) > 1:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def draw(rng):
    p = {}
    p.update(draw_card(rng, "C1"))
    p.update(draw_card(rng, "C10", {"sigma0": (0.003, 0.007)}))
    for _ in range(200):                  # masking construction: the ablation's epoch count must be plausible
        p.update(draw_card(rng, "C6")); p.update(draw_card(rng, "C7"))
        e = mask_epochs(p)
        if e is not None and MASK_EPOCHS[0] <= e <= MASK_EPOCHS[1]:
            break
    p["U0"] = U0
    p["U_prod"] = loguni(rng, *UP_RANGE)   # scenario unknown: not read by the world, keys use the range
    return p


def spec(p):
    return {"knobs": {"N": {"type": "float", "min": 1e7, "max": 3e8},
                      "D": {"type": "float", "min": 2e8, "max": 1e11},
                      "q": {"type": "float", "min": 0.0, "max": 0.9, "default": 0.0},
                      "sub": {"type": "float", "min": 1e8, "max": U0, "default": U0}},
            "fixed": {"B": 5e5, "wd": 0.1, "wu": 0.01},
            "caps": {"run_flops": 6e18, "total_flops": 8e19, "max_runs": 40},
            "metrics": ["loss"]}


def LAB_EXTRA(w):
    return ["",
            "Data knobs: `q` is the fraction of the raw corpus removed by the quality classifier (lowest-scoring "
            "documents first).  `sub` is the number of raw unique tokens sampled from the corpus *before* filtering "
            "(an ablation subset); the filter then keeps a fraction 1-q of it.  Default: the whole current corpus, "
            "%s raw unique tokens.  A run whose D exceeds its unique tokens sub*(1-q) repeats data (several epochs).  "
            "Every run uses a learning rate tuned by the lab for its configuration, so there is no lr knob." % "2e11"]


def known_unknowns(p):
    return [{"name": "next corpus snapshot size", "param": "U_prod", "range": UP_RANGE,
             "text": "Production trains on the next snapshot of the same corpus (same quality distribution; the "
                     "classifier and its q scale are unchanged).  Its deduplication is still running: it will hold "
                     "between 2e10 and 1e11 raw unique tokens, and nothing in this lab depends on the exact number.  "
                     "Questions that name a snapshot size use that size; questions about the snapshot 'as it will be' "
                     "must cover the whole range."}]


# ------------------------------------------------------------------------------------------ notebook
def notebook(p, sess, rng):
    rows = []
    for N in NB_SIZES:
        rows.append(sess.execute({"N": N, "D": 20 * N, "q": 0.0, "sub": U0, **spec(p)["fixed"]}, seed=0))
    e = mask_epochs(p)
    P = float("%.2g" % (NB_D / e))
    for q in NB_QS:
        for s in (0, 1):
            rows.append(sess.execute({"N": NB_N, "D": NB_D, "q": q, "sub": P, **spec(p)["fixed"]}, seed=s))
    ab = {q: np.mean([r["loss"] for r in rows[5:] if abs(r["config"]["q"] - q) < 1e-9]) for q in NB_QS}
    d3, d6 = ab[0.3] - ab[0.0], ab[0.6] - ab[0.0]
    ep = NB_D / P
    notes = f"""# Lab notes (data team)

- Runs 1-5: scaling runs on fresh data (q = 0, whole corpus, D = 20 N), seed 0.
- Runs 6-11: quality-filter ablation.  N = 5e7, D = 8e9 tokens on our standard ablation subset
  (sub = {P:.2g} raw unique tokens, the subset we use for every data ablation), filter levels q = 0, 0.3,
  0.6, seeds 0 and 1.  Mean loss change vs q = 0: {d3:+.4f} nats at q = 0.3, {d6:+.4f} nats at q = 0.6.
  Seed-to-seed spread is about 0.01 nats, so filtering buys us nothing measurable.  Decision: drop the
  classifier, keep every token.
- Production plan: N = 1e9, D = 2e11 tokens on the next corpus snapshot (2e10-1e11 raw unique tokens once
  dedup finishes), q = 0.  That is 2-10 epochs; repetition up to ~4 epochs is nearly free in the
  literature and we accept the small cost beyond that.
"""
    return rows, notes, {"P": P, "ep": ep, "d3": float(d3), "d6": float(d6),
                         "thr4": _q4_threshold(p, rng), "q6q": float(rng.choice(Q6_OPTS))}


def _prod(p, q, U, N=PROD_N, D=PROD_D):
    return float(W.core_loss(W.full(p), {"N": N, "D": D, "q": q, "sub": U}))


def _gain_fresh(p):
    """Filter gain with unlimited unique data (no repetition at any q), N=1e9, D=1e11."""
    pi = dict(p, U0=float("inf"))
    return _prod(pi, 0.0, float("inf"), D=1e11) - _prod(pi, 0.6, float("inf"), D=1e11)


Q4_BAND = 0.006


def _q4_threshold(p, rng):
    g = _gain_fresh(p)
    for _ in range(60):
        thr = float(rng.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]))
        if abs(thr - g) >= max(Q4_BAND, 0.25 * abs(g)):
            return thr
    return thr


def _delta(p, q, U):
    """Loss change from filtering at q (positive = filtering hurts) for the production run on U raw unique tokens."""
    return _prod(p, q, U) - _prod(p, 0.0, U)


def _repcost(p, U):
    """Loss added by repetition to the unfiltered production run on U raw unique tokens."""
    return _prod(p, 0.0, U) - _prod(dict(p, U0=float("inf")), 0.0, float("inf"))


def items(p, ctx, tol=None):
    tol = tol or {}
    grid = np.exp(np.linspace(*np.log(UP_RANGE), 41))
    d_q5 = [_delta(p, 0.5, U) for U in grid]
    q6q = ctx["q6q"]
    beats = [_delta(p, q6q, U) < 0 for U in grid]
    losses = {k: _prod(p, q, UP_Q) for k, q in Q_OPTS.items()}
    thr = ctx["thr4"]
    return [
        Q.point("q1", "Repetition cost: by how much (nats/token) does repetition raise the final loss of the unfiltered "
                      "(q=0) production run N=1e9, D=2e11 on a snapshot of 4e10 raw unique tokens, compared with the same "
                      "run on unlimited unique data?  (A positive number; 0 if repetition were free.)", "nats",
                _repcost(p, UP_Q), tol.get("q1", 0.005), cards=["C6"], obstacles=["O3"], floor=0.0015),
        Q.point("q2", "Filtering effect: final loss of the production run (N=1e9, D=2e11, snapshot of 4e10 raw unique "
                      "tokens) at q=0.5 minus its final loss at q=0, in nats/token (negative = filtering helps).", "nats",
                _delta(p, 0.5, UP_Q), tol.get("q2", 0.005), cards=["C6", "C7"], obstacles=["O2", "O3"], floor=0.0015),
        Q.decision("q3", "For the production run N=1e9, D=2e11 on a snapshot of 4e10 raw unique tokens, which filter "
                         "level gives the lowest final loss?  Options: " +
                   ", ".join("%s: q=%.2g" % (k, v) for k, v in Q_OPTS.items()), list(Q_OPTS), losses,
                   tol.get("q3", 0.004), cards=["C6", "C7"], obstacles=["O2", "O3"]),
        Q.verdict("q4", "With enough unique data that nothing is repeated, filtering at q=0.6 lowers the final loss of "
                        "N=1e9 trained on D=1e11 tokens by at least %.3f nats (compared with q=0)." % thr,
                  [_gain_fresh(p) >= thr], cards=["C7"], obstacles=["O2"]),
        Q.interval("q5", "Filtering effect on the next snapshot as it will be: final loss of N=1e9, D=2e11 at q=0.5 minus "
                         "at q=0, in nats/token (negative = filtering helps); the snapshot size is the documented known "
                         "unknown.", "nats", min(d_q5), max(d_q5), tol.get("q5", 0.005), cards=["C6", "C7"],
                   obstacles=["O5"], floor=0.0015),
        Q.verdict("q6", "For the production run as planned (N=1e9, D=2e11, next snapshot), filtering at q=%.2g gives a "
                        "lower final loss than no filtering." % q6q, beats, cards=["C6", "C7"], obstacles=["O5", "O2"]),
    ]


# ------------------------------------------------------------------------------------------ fitting
FIT_KEYS = ["E", "rN", "alpha", "rD", "beta", "Rs", "mu", "nu"]
TB_C1 = c1_to_r(textbook(["C1"]))
KNOBS = ["N", "D", "q", "sub"]


def _model(cfg):
    return lambda q: W.core_loss(W.full(dict(c1_from_r(q), U0=U0)), cfg)


def fit_rows(rows, rng, fixed=None, keys=None, n_starts=8, init_from=None):
    keys = [k for k in (keys or FIT_KEYS) if k not in (fixed or {})]
    cfg, y, ok = rows_xy(rows, KNOBS)
    w = (cfg["N"] / 1e8) ** 0.3
    box = box_for(keys, widen=1.6, overrides=dict(R_BOX, Rs=(2.0, 200.0), mu=(0.02, 3.0), nu=(0.9, 2.9)))
    init = dict(TB_C1, Rs=15.4, mu=0.6, nu=1.5)
    init.update(init_from or {})
    init = {k: init[k] for k in box.names if k in init}
    init = {k: min(max(v, (math.exp(box.lo[i]) if box.log[i] else box.lo[i]) * 1.001 if v > 0 else v),
                   (math.exp(box.hi[i]) if box.log[i] else box.hi[i]) * 0.999)
            for i, k in enumerate(box.names) for v in [init[k]]}
    fx = dict(fixed or {})
    return c1_from_r(F.fit(_model(cfg), box, fx, y, w, rng, n_starts=n_starts, init=init)[0])


def answers_from(ph, ctx, collapse=False):
    ph = dict(ph); ph.setdefault("U0", U0)
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


def oracle_design(rows_nb):
    reqs = []; s = 100
    for N, D in ((2e7, 4e8), (2e7, 4e9), (2e7, 4e10), (6e7, 1.5e9), (6e7, 1.5e10), (2e8, 1e9), (2e8, 4.5e9), (3e8, 3e9)):
        reqs.append({"N": N, "D": D, "q": 0.0, "seed": s}); s += 1
    for q in (0.0, 0.3, 0.6, 0.85):                          # fresh-data filter sweep
        for k in range(2):
            reqs.append({"N": 5e7, "D": 5e9, "q": q, "seed": s}); s += 1
    for ep in (2.0, 5.0, 12.0, 30.0):                        # repetition sweep (q = 0, small subsample)
        for k in range(2):
            reqs.append({"N": 5e7, "D": 5e9, "q": 0.0, "sub": 5e9 / ep, "seed": s}); s += 1
    return reqs


def oracle(sess, rows_nb, ctx, rng, drop=None):
    own = run_rows(sess, oracle_design(rows_nb))
    keys = [k for k in FIT_KEYS if k not in (drop or {})]
    ph = fit_rows(rows_nb + own, rng, fixed=drop, keys=keys)
    ph = fit_rows(rows_nb + own, rng, fixed=drop, keys=keys, init_from=c1_to_r(ph), n_starts=4)
    return answers_from(ph, ctx), ph


def rivals(p, rows_nb, ctx, rng):
    out = {}
    out["B0_textbook"] = answers_from(dict(textbook(CARDS)), ctx)
    # B1: the notes' reading - filtering is worthless (mu = 0); repetition fitted to the notebook
    out["B1_notebook_naive"] = answers_from(fit_rows(rows_nb, rng, fixed={"mu": 0.0, "nu": 1.0}), ctx)
    out["B2_notebook_allcards"] = answers_from(fit_rows(rows_nb, rng), ctx)
    # B5: the literature's repetition constant (Muennighoff R*=15.4) plus the notebook for the rest
    out["B5_literature_Rs"] = answers_from(fit_rows(rows_nb, rng, fixed={"Rs": 15.4}), ctx)
    return out


def rival_designs(p, rows_nb, rng):
    reqs = []
    for j in range(24):
        N = float(np.exp(rng.uniform(np.log(1e7), np.log(3e8))))
        D = float(np.exp(rng.uniform(np.log(2e8), np.log(min(1e11, 6e18 / (6 * N))))))
        reqs.append({"N": N, "D": D, "q": float(rng.uniform(0, 0.9)),
                     "sub": float(np.exp(rng.uniform(np.log(1e9), np.log(U0)))), "seed": 500 + j})
    conf = []                                                   # the obvious check: replicate the ablation
    P = [r for r in rows_nb if r["config"]["sub"] < U0][0]["config"]["sub"]
    for q in NB_QS:
        for s in range(4):
            conf.append({"N": NB_N, "D": NB_D, "q": q, "sub": P, "seed": 700 + s})
    for N, D in ((2e7, 4e9), (6e7, 1.5e10), (2e8, 4.5e9)):
        conf.append({"N": N, "D": D, "q": 0.0, "seed": 720})
    return {"B3_random_design": reqs, "B4_replicate_ablation": conf}


DROP = {"C6:Rs": {"Rs": float("inf")}, "C7:mu": {"mu": 0.0, "nu": 1.0}, "C1:textbook": dict(TB_C1)}
# B3 random, B2 notebook-only and B4 (replicating the masked ablation with more seeds) cannot separate the
# quality and repetition constants: where their fit lands in the (mu, nu, Rs) valley is luck.  Reported,
# not gated.  B5 (literature Rs) is right exactly when the world's Rs is near 15.4: reported, not gated,
# so the instance distribution is not pushed away from the literature value.
INFO_RIVALS = {"B3_random_design", "B2_notebook_allcards", "B4_replicate_ablation", "B5_literature_Rs"}


def public_values(ctx):
    return [ctx["P"], ctx["ep"], ctx["d3"], ctx["d6"], ctx["thr4"], ctx["q6q"], U0, UP_Q] + list(UP_RANGE)
