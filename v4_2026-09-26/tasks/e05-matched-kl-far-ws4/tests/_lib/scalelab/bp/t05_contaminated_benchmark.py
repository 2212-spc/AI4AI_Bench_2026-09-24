"""T05 contaminated-benchmark: books in the data mixture, a public QA benchmark partly leaked into the books
corpus, and a small clean split.  (O4 measurement artifact, O1 confounded notebook, O9 scale transfer, O5.)

Notebook: the team swept the books fraction b at their largest lab configuration and read the 2000-item
public benchmark.  Its score keeps rising with b, so they plan production at the best-looking b.  Under C9
the public score mixes real knowledge (a sigmoid of a books-adjusted loss, peaking at b* = gb/(2 hb)) with
memorised leaked items (a fraction phi of the benchmark; memorisation grows with model size and with the
number of books tokens seen, 1 - exp(-km (N/1e8)^xi b D / 1e9)).  The 250-item clean split measures the
real part but is noisy, and the notes dismiss it.  At production scale memorisation is nearly saturated,
so the public score overstates the model by ~phi (1 - a), far more than the notebook's gap suggests (O9).
The books fraction of a small release (lab-size model, 3x more tokens) is not decided yet (known unknown):
its contamination gap is a set.

Useful designs: small models trained on many books tokens have clean accuracy ~0, so their public score is
pure memorisation (phi * mem); b = 0 runs give contamination-free public scores (2000 items) for the
loss->accuracy map; the clean split with several seeds for the books curve.
"""
import math
import numpy as np
from .. import world as W
from .. import queries as Q
from ..common import draw_card, textbook, box_for, run_rows, c1_from_r, c1_to_r, R_BOX, loguni
from .. import fit as F

ID = "t05-contaminated-benchmark"
TITLE = "Books in the mixture: real knowledge gains versus benchmark memorisation, carried to production"
CARDS = ["C1", "C9", "C10"]
OBSTACLES = ["O4", "O1", "O9", "O5"]
CLAIM = ("recognise that a benchmark score mixes real capability with memorised leaked items, design runs that "
         "separate the two, and carry both - with their different scaling - to production and to an undecided mixture")
EXEMPT_LOAD_BEARING = {"C10": "noise card: shapes tolerances, not answers (declared exemption)",
                       "C1": "enters every item only through loss -> accuracy; the free sigmoid (L50, sa) absorbs a "
                             "textbook C1 over a 3x extrapolation (measured: drop:C1:textbook almost never killed), "
                             "so C1 is context, not load (declared exemption)"}

LAB_N, LAB_D = 3e8, 1e10
NB_B = [0.0, 0.15, 0.3, 0.45, 0.6]
PROD_N, PROD_D = 1e9, 3e10
REL_N, REL_D = 3e8, 3e10                  # small release: lab-size model, 3x the lab's longest run at that size
B_OPTS = {"A": 0.05, "B": 0.2, "C": 0.35, "D": 0.6}
BREL = (0.02, 0.4)                        # known unknown: books fraction of the small release
QA_ITEMS, QA_CLEAN_ITEMS = 2000, 250
Q5_BAND = 0.01


def _logit(a):
    return math.log(a / (1 - a))


def _core(p, N, D):
    return float(W.core_loss(W.full(p), {"N": N, "D": D}))


def draw(rng):
    for _ in range(5000):
        p = {}
        p.update(draw_card(rng, "C1"))
        p.update(draw_card(rng, "C10", {"sigma0": (0.004, 0.008)}))
        L_lab, L_prod = _core(p, LAB_N, LAB_D), _core(p, PROD_N, PROD_D)
        a_lab = float(rng.uniform(0.3, 0.5)); a_prod = float(rng.uniform(0.5, 0.78))   # clean accuracy at b = 0
        sa = (L_lab - L_prod) / (_logit(a_prod) - _logit(a_lab))
        if not 0.1 <= sa <= 0.4:
            continue
        bstar = float(rng.uniform(0.1, 0.5)); g = float(rng.uniform(0.25, 0.6))          # logit gain at b*
        dstar = g * sa
        xi = float(rng.uniform(0.3, 1.0)); x_lab = loguni(rng, 0.8, 2.5)                 # exposure at the notebook's b = 0.6
        p.update({"L50": L_prod + sa * _logit(a_prod), "sa": sa, "hb": dstar / bstar ** 2, "gb": 2 * dstar / bstar,
                  "phi": float(rng.uniform(0.15, 0.35)), "xi": xi,
                  "km": x_lab / ((LAB_N / 1e8) ** xi * 0.6 * LAB_D / 1e9), "vb": float(rng.uniform(0.05, 0.3))})
        p["b_rel"] = float(rng.uniform(*BREL))     # scenario unknown: not read by the world; keys use the range
        return p
    raise RuntimeError("no admissible draw")


def spec(p):
    return {"knobs": {"N": {"type": "float", "min": 1e7, "max": 3e8, "log": True},
                      "D": {"type": "float", "min": 2e8, "max": 3e10, "log": True},
                      "b": {"type": "float", "min": 0.0, "max": 0.8}},
            "fixed": {"B": 5e5},
            "caps": {"run_flops": 2e19, "total_flops": 1.5e20, "max_runs": 16},
            "metrics": ["loss", "qa", "qa_clean"], "qa_items": QA_ITEMS, "qa_clean_items": QA_CLEAN_ITEMS}


def LAB_EXTRA(w):
    return ["",
            "Data mixture (`b`): the fraction of training tokens drawn from the **books** corpus; the rest is web text.  "
            "Every run uses a learning rate tuned by the lab, so there is no lr knob.",
            "",
            "Metrics: `loss` is the validation loss on held-out **web** text.  `qa_acc` is exact-match accuracy on "
            "the public QA benchmark (2000 short-answer questions, released on the web several years ago).  "
            "`qa_clean_acc` is accuracy on the benchmark's clean split: 250 questions of the same kind and difficulty "
            "that the eval team wrote after every training corpus was frozen, so they occur in no training data.  "
            "The web corpus was deduplicated against the public benchmark; the books corpus was not audited.  "
            "Both accuracies are measured once per run (the reported number includes the finite-sample scatter of "
            "the question set and the run's seed effect)."]


def known_unknowns(p):
    return [{"name": "books fraction of the small release", "param": "b_rel", "range": BREL,
             "text": "The small release (N=3e8 trained on D=3e10 tokens) will use a books fraction that the data "
                     "team will fix later, somewhere between 0.02 and 0.4.  Nothing in this lab depends on that choice.  "
                     "Questions about the small release must cover the whole range."}]


# ------------------------------------------------------------------------------------------ world helpers
def _acc(p, N, D, b):
    a, agg = W.qa_acc(W.full(p), {"N": N, "D": D, "b": b})
    return float(a), float(agg)


def _nb_run(sess, req):
    cfg, seed, ex = sess.validate(req)
    return sess.execute(cfg, seed, ex)


def notebook(p, sess, rng):
    rows = [_nb_run(sess, {"N": LAB_N, "D": LAB_D, "b": b, "seed": 0}) for b in NB_B]
    qa = [r["qa_acc"] for r in rows]; qc = [r["qa_clean_acc"] for r in rows]; ls = [r["loss"] for r in rows]
    j = int(np.argmax(qa)); b_team = NB_B[j]
    team_opt = min(B_OPTS, key=lambda k: abs(B_OPTS[k] - b_team))
    tab = "\n".join("  | %.2f | %.4f | %.4f | %.4f |" % (b, x, c, l) for b, x, c, l in zip(NB_B, qa, qc, ls))
    notes = f"""# Lab notes (data-mixture team)

- Runs 1-5: N = 3e8, D = 1e10 (our largest lab configuration), seed 0, books fraction b swept:

  | b | public QA | clean split | web loss |
  |---|---|---|---|
{tab}

- The public benchmark (2000 questions) is our decision metric.  It is best at b = {b_team:.2f}
  ({qa[j] - qa[0]:+.3f} over no books).  The clean split has only 250 questions (roughly +-3 points of
  sampling noise per run), so its wiggles are not informative; we do not use it for decisions.
- Web loss rises by {ls[j] - ls[0]:+.4f} nats at b = {b_team:.2f}; acceptable for the benchmark gain.
- Production plan: N = 1e9, D = 3e10, b = {b_team:.2f}.  A small release (N = 3e8, D = 3e10) will follow;
  its books fraction is not fixed yet (see the manual).
"""
    ctx = {"b_team": b_team, "team_opt": team_opt, "qa": qa, "qc": qc, "ls": ls}
    return rows, notes, ctx


# ------------------------------------------------------------------------------------------ items
def _gap_rel(p, b):
    a, agg = _acc(p, REL_N, REL_D, b)
    return agg - a


def items(p, ctx, tol=None):
    tol = tol or {}
    bt = ctx["b_team"]
    losses = {k: -_acc(p, PROD_N, PROD_D, b)[0] for k, b in B_OPTS.items()}
    a_t, agg_t = _acc(p, PROD_N, PROD_D, bt)
    grid = np.linspace(BREL[0], BREL[1], 31)
    gm = [_gap_rel(p, b) for b in grid]
    a6 = _acc(p, PROD_N, PROD_D, 0.6)[0]; a0 = _acc(p, PROD_N, PROD_D, 0.0)[0]
    return [
        Q.decision("q1", "For the production model (N=1e9 trained on D=3e10 tokens), which books fraction gives the "
                         "highest accuracy on the clean split?  Options: " +
                   ", ".join("%s: b=%.2f" % (k, v) for k, v in B_OPTS.items()), list(B_OPTS), losses,
                   tol.get("q1", 0.004), cards=["C9"], obstacles=["O4", "O1"]),
        Q.point("q2", "Accuracy on the clean split (fraction in [0, 1]) of the production model as planned in the notes "
                      "(N=1e9, D=3e10, b=%.2f)." % bt, "fraction",
                a_t, tol.get("q2", 0.02), cards=["C1", "C9"], obstacles=["O4", "O9"], floor=0.01),
        Q.point("q3", "Accuracy on the public benchmark (fraction in [0, 1]) of the same production model "
                      "(N=1e9, D=3e10, b=%.2f)." % bt, "fraction",
                agg_t, tol.get("q3", 0.02), cards=["C1", "C9"], obstacles=["O4", "O9"], floor=0.01),
        Q.interval("q4", "For the small release (N=3e8, D=3e10, books fraction as documented), by how much will its "
                         "public-benchmark accuracy exceed its clean-split accuracy (public minus clean, as a fraction; "
                         "noise-free expected values)?  Give the set over the documented range of its books fraction.",
                   "fraction", min(gm), max(gm), tol.get("q4", 0.01), cards=["C9"], obstacles=["O4", "O9", "O5"],
                   floor=0.004),
        Q.verdict("q5", "A production model (N=1e9, D=3e10) trained with b=0.6 has a higher expected clean-split "
                        "accuracy than the same model trained with no books (b=0).", [a6 > a0],
                  cards=["C9"], obstacles=["O4", "O9"]),
    ]


# ------------------------------------------------------------------------------------------ fitting
C9K = ["L50", "sa", "gb", "hb", "phi", "km", "xi", "vb"]
FIT_KEYS = ["E", "rN", "alpha", "rD", "beta"] + C9K
TB = c1_to_r(textbook(["C1", "C9"]))
C9SPEC = {"L50": (1.8, 4.2, False), "sa": (0.04, 0.7, True), "gb": (0.0, 3.0, False), "hb": (0.0, 12.0, False),
          "phi": (0.0, 0.6, False), "km": (1e-3, 2.0, True), "xi": (0.0, 1.6, False), "vb": (0.0, 0.6, False)}


def _box(keys):
    """C1 keys: card ranges widened 1.6x (as elsewhere); C9 keys: explicit physical bounds (no negative rates)."""
    c1 = [k for k in keys if k not in C9SPEC]
    spec = {}
    if c1:
        b1 = box_for(c1, widen=1.6, overrides=R_BOX)
        for i, k in enumerate(b1.names):
            lo, hi = b1.lo[i], b1.hi[i]
            spec[k] = (math.exp(lo), math.exp(hi), True) if b1.log[i] else (lo, hi, False)
    for k in keys:
        if k in C9SPEC:
            spec[k] = C9SPEC[k]
    return F.Box({k: spec[k] for k in keys})


def _stack(rows, use_qa=True, use_clean=True):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("loss") is not None]
    cfg = {k: np.array([float(r["config"][k]) for r in ok]) for k in ("N", "D", "b")}
    n = len(ok)
    y = [np.array([r["loss"] for r in ok])]; wt = [(cfg["N"] / 1e8) ** 0.3 / 0.006]; tags = ["loss"]
    if use_qa:
        y.append(np.array([r["qa_acc"] for r in ok])); wt.append(np.full(n, 1 / 0.012)); tags.append("qa")
    if use_clean:
        y.append(np.array([r["qa_clean_acc"] for r in ok])); wt.append(np.full(n, 1 / 0.032)); tags.append("qc")
    return cfg, np.concatenate(y), np.concatenate(wt), tags


def fit_rows(rows, rng, fixed=None, keys=None, n_starts=8, init_from=None, use_qa=True, use_clean=True):
    fx = dict(fixed or {})
    keys = [k for k in (keys or FIT_KEYS) if k not in fx]
    cfg, y, wt, tags = _stack(rows, use_qa, use_clean)
    box = _box(keys)
    init = dict(TB); init.update(init_from or {})
    init = {k: init[k] for k in box.names if k in init}
    for i, k in enumerate(box.names):
        if k in init:
            lo = math.exp(box.lo[i]) if box.log[i] else box.lo[i]
            hi = math.exp(box.hi[i]) if box.log[i] else box.hi[i]
            init[k] = min(max(init[k], lo + 1e-3 * (hi - lo)), hi - 1e-3 * (hi - lo))

    def model(q):
        pp = W.full(c1_from_r(q))
        out = [W.val_loss(pp, cfg)]
        a, agg = W.qa_acc(pp, cfg)
        if "qa" in tags:
            out.append(agg)
        if "qc" in tags:
            out.append(a)
        return np.concatenate([np.broadcast_to(np.asarray(o, float), cfg["N"].shape) for o in out])
    return c1_from_r(F.fit(model, box, fx, y, wt, rng, n_starts=n_starts, init=init)[0])


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


def oracle_design(rows_nb):
    """b = 0 runs across N and D (loss law + contamination-free public scores for the loss->accuracy map);
    small models on many books tokens (clean accuracy ~0, so the public score is phi * memorisation) at
    several sizes and exposures; the notebook configuration re-run at other b and seeds for the books curve.
    13 runs, about 1.16e20 of the 1.5e20 budget."""
    B = [(1e7, 1e9, 0.0), (3e7, 3e9, 0.0), (1e8, 3e9, 0.0), (3e8, 3e9, 0.0), (1e8, 1e10, 0.0), (3e8, 1e10, 0.0),
         (3e8, 1e10, 0.3), (3e8, 1e10, 0.8), (1e8, 3e10, 0.6), (3e8, 3e9, 0.6), (1e8, 3e10, 0.25),
         (3e7, 3e10, 0.6), (1e7, 3e10, 0.8)]
    return [{"N": N, "D": D, "b": b, "seed": 100 + i} for i, (N, D, b) in enumerate(B)]


def oracle(sess, rows_nb, ctx, rng, drop=None):
    own = run_rows(sess, oracle_design(rows_nb))
    keys = [k for k in FIT_KEYS if k not in (drop or {})]
    ph = fit_rows(rows_nb + own, rng, fixed=drop, keys=keys, n_starts=16)
    ph = fit_rows(rows_nb + own, rng, fixed=drop, keys=keys, init_from=c1_to_r(ph), n_starts=4)
    return answers_from(ph, ctx), ph


def _grid_session(p, tag):
    from ..lab import Session
    return Session(W.full(p), spec(p), "rival/" + tag)


def rivals(p, rows_nb, ctx, rng):
    out = {}
    out["B0_textbook"] = answers_from(dict(textbook(CARDS)), ctx)
    sess = _grid_session(p, "R")
    own = run_rows(sess, oracle_design(rows_nb))
    rows = rows_nb + own
    # B1: the notes' reading - the public benchmark is the model's accuracy (no contamination term, clean split unused)
    out["B1_public_is_truth"] = answers_from(fit_rows(rows, rng, fixed={"phi": 0.0, "km": 0.02, "xi": 0.6},
                                                      use_clean=False), ctx)
    # B2: contamination measured, but memorisation assumed independent of model size (depends on exposure only)
    out["B2_size_free_memorisation"] = answers_from(fit_rows(rows, rng, fixed={"xi": 0.0}), ctx)
    # B3 (minimal pair): the true world everywhere, except that the contamination gap is carried over from the
    # notebook configuration (N=3e8, D=1e10) at the same b - 'the benchmark overstates us by what we measured in
    # the lab'.  Built on the truth rather than on a noisy fit, so it can only be killed through q3/q4, the items
    # the shortcut touches (a fit-based base was killed by fit noise on q1 alone - a spurious kill).
    pt = W.full(p)
    b3 = answers_from(pt, ctx)
    a_lab, agg_lab = _acc(pt, LAB_N, LAB_D, ctx["b_team"])
    b3["q3"] = {"lo": _acc(pt, PROD_N, PROD_D, ctx["b_team"])[0] + agg_lab - a_lab}
    b3["q3"]["hi"] = b3["q3"]["lo"]
    gl = [_acc(pt, LAB_N, LAB_D, b) for b in np.linspace(BREL[0], BREL[1], 31)]
    b3["q4"] = {"lo": min(g[1] - g[0] for g in gl), "hi": max(g[1] - g[0] for g in gl)}
    out["B3_lab_gap_transfer"] = b3
    return out


def rival_designs(p, rows_nb, rng):
    return {}


DROP = {"C9:phi": {"phi": 0.0, "km": 0.02, "xi": 0.6},
        "C9:books": {"gb": 0.0, "hb": 0.0},
        "C1:textbook": {k: TB[k] for k in ("E", "rN", "alpha", "rD", "beta")}}
INFO_RIVALS = {"B2_size_free_memorisation", "drop:C1:textbook"}   # C1 exempt (see EXEMPT_LOAD_BEARING)


def wellposed(w):
    ctx = w["ctx"]; p = w["pf"]
    its = items(p, ctx)
    q1 = [it for it in its if it["id"] == "q1"][0]
    if ctx["b_team"] < 0.3:
        return False, "notebook argmax b=%.2f: the team's story needs a books-heavy plan" % ctx["b_team"]
    if ctx["team_opt"] == q1["key"]["choice"]:
        return False, "the team's plan already picks the clean-optimal option"
    m = _acc(p, PROD_N, PROD_D, 0.6)[0] - _acc(p, PROD_N, PROD_D, 0.0)[0]
    if abs(m) < Q5_BAND:
        return False, "q5 margin %.4f below band" % m
    return True, "team b=%.2f (%s), truth %s; q5 margin %.4f" % (ctx["b_team"], ctx["team_opt"], q1["key"]["choice"], m)


def public_values(ctx):
    vals = [LAB_N, LAB_D, PROD_N, PROD_D, REL_N, REL_D, ctx["b_team"]] + NB_B + list(B_OPTS.values()) + list(BREL)
    vals += ctx["qa"] + ctx["qc"] + ctx["ls"]
    vals += [ctx["qa"][NB_B.index(ctx["b_team"])] - ctx["qa"][0], ctx["ls"][NB_B.index(ctx["b_team"])] - ctx["ls"][0]]
    return vals
