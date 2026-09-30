"""T02 stability-edge: a learning rate that is safe in every lab sweep diverges at production scale, and
the fix the team dismissed (qk-layernorm) can only be bounded, not measured, in this lab.

Notebook: LR sweeps along D = 20 N at four sizes (warmup 2%, qk-layernorm off) plus a paired qk on/off
check at one size.  The largest-LR points at the biggest sizes diverged; the notes call that the usual
instability far above the optimum.  qk-layernorm made no difference to loss, so the notes drop it.  They
extrapolate the loss-optimal LR to a 7B run and plan to use it.

Truth: the divergence edge falls with N faster than the optimal LR does (O3/O9): at 7B the edge is
below the optimum, so the loss-optimal LR is not runnable without qk-layernorm.  With qk-layernorm on,
no lab run can diverge (the lab's LR cap sits below the qk-on edge everywhere), so the qk factor Q is
*censored*: the lab gives a lower bound Q >= lr_cap / edge_off(most unstable lab config), the stack's
documentation gives the upper bound (O5).  The tightest lower bound needs the qk-on run at the lab
corner that is least stable without qk (largest N, zero warmup, LR cap) - a design choice.  The edge is
deterministic in this lab (div_jitter = 0) so the identification set has crisp endpoints.
"""
import math
import numpy as np
from .. import world as W
from .. import queries as Q
from ..common import draw_card, textbook, loguni

ID = "t02-stability-edge"
TITLE = "Stability of a 7B production run planned from small-scale sweeps"
CARDS = ["C1", "C2", "C4", "C10"]
OBSTACLES = ["O3", "O5", "O9"]
CLAIM = ("bound a quantity the lab only observes censored (a stability edge that never binds in the lab), choose the "
         "experiment that makes the bound tightest, and state exactly what remains unknown")
EXEMPT_LOAD_BEARING = {"C1": "loss floor under the LR bowl; no item asks about loss levels (declared exemption)",
                       "C10": "noise card: shapes the notebook, not the answers (declared exemption)"}

NB_SIZES = [2e7, 4e7, 8e7, 1.6e8]
PROD_N, PROD_D, PROD_WU, HOT_WU = 7e9, 1.4e11, 0.02, 0.05
LAB_B = 5e5
N_MIN, N_MAX, LR_MAX, WU_MAX, D_MIN = 1e7, 3e8, 5e-2, 0.05, 2e8
Q_DOC = (2.0, 60.0)
BAND = 0.1          # verdict probes are kept at least this far (in ln lr) from their truth boundary
LN10 = math.log(10)


def _edge(p, N, wu):
    return float(W.eta_max(p, N, wu, 0))


def _eta(p, N, D):
    return float(W.eta_star(p, N, D))


def draw(rng):
    for _ in range(5000):
        p = {}
        p.update(draw_card(rng, "C1"))
        p.update(draw_card(rng, "C2", {"gN": (0.0, 0.08), "gD": (0.08, 0.22)}))
        p["gB"] = 0.0
        p.update(draw_card(rng, "C10", {"sigma0": (0.003, 0.007)}))
        c4 = draw_card(rng, "C4", {"delta": (0.4, 0.6)})
        c4["c_wu"] = 0.0                       # warmup has no loss cost here (only its stability effect)
        p.update(c4)
        pf = W.full(p)
        # place the edge: at production it sits a factor ratio_prod below the loss-optimal LR
        ratio_prod = float(rng.uniform(0.45, 0.75))
        base = (PROD_N / 1e8) ** (-p["delta"]) * (1 + PROD_WU / p["w0"]) ** p["omega"]
        p["h0"] = ratio_prod * _eta(pf, PROD_N, PROD_D) / base
        pf = W.full(p)
        lab_ratio = _edge(pf, N_MAX, PROD_WU) / _eta(pf, N_MAX, 20 * N_MAX)
        qlo = LR_MAX / _edge(pf, N_MAX, 0.0)
        wu_gain = p["omega"] * math.log(1 + PROD_WU / p["w0"])
        lo_ok = _edge(pf, NB_SIZES[0], PROD_WU) <= LR_MAX / 1.3      # edge measurable across the lab's N range
        if lab_ratio >= 1.3 and 5.0 <= qlo <= 25.0 and wu_gain >= 0.15 and Q_DOC[1] / qlo >= 2.0 and lo_ok:
            p["Q"] = loguni(rng, qlo * 1.25, Q_DOC[1])
            return p
    raise RuntimeError("no admissible draw")


def spec(p):
    return {"knobs": {"N": {"type": "float", "min": N_MIN, "max": N_MAX},
                      "D": {"type": "float", "min": D_MIN, "max": 1e11},
                      "lr": {"type": "float", "min": 1e-4, "max": LR_MAX},
                      "wu": {"type": "float", "min": 0.0, "max": WU_MAX},
                      "qk": {"type": "choice", "values": [0, 1]}},
            "fixed": {"B": LAB_B, "wd": 0.1},
            "caps": {"run_flops": 6e18, "total_flops": 3e19, "max_runs": 40},
            "metrics": ["loss"], "div_jitter": 0.0}


def known_unknowns(p):
    qlo = LR_MAX / _edge(W.full(p), N_MAX, 0.0)
    # range = the part of the documented range the lab cannot distinguish (zero footprint, certified by G4);
    # the documented range itself is stated in the text.  Values below qlo are ruled out by the lab.
    return [{"name": "qk-layernorm stability factor", "param": "Q", "range": (qlo * (1 + 1e-6), Q_DOC[1]),
             "text": "Our stack's documentation states that turning on qk-layernorm multiplies the learning rate "
                     "at which a run diverges by a constant factor between 2 and 60, the same factor at every "
                     "model size, token count and warmup; qk-layernorm does not change the loss of a run that "
                     "does not diverge.  Nobody has measured the factor for our stack."}]


def LAB_EXTRA(w):
    return ["Production context: N=7e9, D=1.4e11 tokens, batch 0.5M tokens, warmup 2% of steps (`wu`=0.02), same "
            "optimizer and schedule as the lab."]


# ------------------------------------------------------------------------------------------ notebook
def _team_fit(rows):
    """What the team did: per-size parabola in ln(lr) over non-diverged runs, power law across sizes."""
    ests, ns = [], []
    for N in NB_SIZES:
        rr = [r for r in rows if abs(r["config"]["N"] - N) < 1 and r["status"] == "ok" and r["config"]["qk"] == 0
              and r["config"].get("seed", 0) == 0]
        if len(rr) < 3:
            continue
        x = np.log([r["config"]["lr"] for r in rr]); y = np.array([r["loss"] for r in rr])
        a, b, c = np.polyfit(x, y, 2)
        if a > 0:
            ests.append(-b / (2 * a)); ns.append(N)
    s, i = np.polyfit(np.log(ns), ests, 1)
    return -s, i


def _nb_run(sess, req):
    cfg, seed, ex = sess.validate(req)                 # same normalisation as a lab request, so replay is exact
    return sess.execute(cfg, seed, ex)


def notebook(p, sess, rng):
    rows = []
    mids = {}
    for N in NB_SIZES:
        D = 20 * N
        c = _eta(p, N, D) * math.exp(rng.normal(0, 0.12)); mids[N] = c
        for k in (-2, -1, 0, 1, 2):
            lr = float("%.4g" % min(LR_MAX, c * 1.6 ** k))
            rows.append(_nb_run(sess, {"N": N, "D": D, "lr": lr, "wu": PROD_WU, "qk": 0, "seed": 0}))
    N = 8e7; lr = float("%.4g" % mids[N])
    for qk, seed in ((0, 1), (1, 0), (1, 1)):
        rows.append(_nb_run(sess, {"N": N, "D": 20 * N, "lr": lr, "wu": PROD_WU, "qk": qk, "seed": seed}))
    s, i = _team_fit(rows)
    lr_notes = float("%.3g" % (math.exp(i) * PROD_N ** (-s)))
    ndiv = sum(r["status"] == "diverged" for r in rows)
    divtxt = ("No run diverged, so stability is not a concern at these learning rates." if ndiv == 0 else
              "%s of the highest-LR runs diverged (loss spike early in training).  That is the usual instability far "
              "above the\n  optimum and does not matter for runs at the optimum." % ("One" if ndiv == 1 else str(ndiv)))
    notes = f"""# Lab notes (pretraining team)

- LR sweeps at four sizes along the Chinchilla ratio (D = 20 N), batch 0.5M, warmup 2%, qk-layernorm off, seed 0.
  {divtxt}
- Per-size optimum from a parabola in ln(lr) over the runs that finished, then a power law across sizes:
  **eta*(N) = {math.exp(i):.4g} * N^-{s:.3f}** (along D = 20 N).
- qk-layernorm on vs off at 8e7 params, at that size's optimum, two seeds each: no difference in loss beyond seed
  noise.  It costs ~3% step time, so production drops it.
- Production plan: N = 7e9, D = 1.4e11 tokens (20 tokens/param), batch 0.5M, warmup 2%, qk-layernorm off,
  peak LR from the rule above: **lr_prod = {lr_notes:.3g}**.
"""
    ctx = {"lr_notes": lr_notes, "team_slope": s, "team_icpt": math.exp(i)}
    ctx.update(_probes(p, rng))
    return rows, notes, ctx


def _probes(p, rng):
    """Probe values for verdict and decision items (drawn from the truth, kept out of a band around the
    verdict boundary so no answer flips inside the measurement error)."""
    pf = W.full(p); e1 = _edge(pf, PROD_N, PROD_WU); e5 = _edge(pf, PROD_N, HOT_WU)
    qlo = LR_MAX / _edge(pf, N_MAX, 0.0)
    qa = math.exp(rng.uniform(math.log(Q_DOC[0]) + 0.35, math.log(qlo) - 0.25))           # supported only via the lab bound
    if rng.uniform() < 0.7:
        qb = math.exp(rng.uniform(math.log(qlo) + 0.25, math.log(Q_DOC[1]) - 0.25))       # undetermined
    else:
        qb = math.exp(rng.uniform(math.log(Q_DOC[1]) + 0.25, math.log(Q_DOC[1]) + 1.0))   # refuted
    if rng.uniform() < 0.5:
        qa, qb = qb, qa                                                                    # order carries no signal
    y7 = e5 * math.exp(rng.choice([-1, 1]) * rng.uniform(0.12, 0.35))
    s = math.log(1.25); off = float(rng.uniform(-s + 0.07, -0.07)); pos = int(rng.integers(1, 3))
    opts = {}
    for k, name in enumerate("ABCD"):
        opts[name] = float("%.3g" % (e1 * math.exp(s * (k - pos) + off)))
    return {"x4": float("%.3g" % (qa * e1)), "x5": float("%.3g" % (qb * e1)), "x7": float("%.3g" % y7), "opts": opts}


def wellposed(w):
    p, ctx = w["pf"], w["ctx"]
    e1 = _edge(p, PROD_N, PROD_WU)
    r3 = math.log(ctx["lr_notes"] / e1)
    return abs(r3) >= BAND, "ln(lr_notes/edge_prod) = %.3f (band %.2f)" % (r3, BAND)


# ------------------------------------------------------------------------------------------ keys
def derived(p):
    """Everything the items depend on, as logs.  The oracle and rivals produce the same dict."""
    pf = W.full(p)
    return {"le_prod": math.log(_edge(pf, PROD_N, PROD_WU)), "le_hot": math.log(_edge(pf, PROD_N, HOT_WU)),
            "le_min": math.log(_edge(pf, N_MAX, 0.0)), "leta_prod": math.log(_eta(pf, PROD_N, PROD_D)),
            "k_lo": pf["k_lo"], "k_hi": pf["k_hi"]}


def _qset(d):
    """ln Q identification set: lab lower bound (never below the documented minimum), documented maximum."""
    lo = d.get("lq_lo", max(math.log(Q_DOC[0]), math.log(LR_MAX) - d["le_min"]))
    hi = d.get("lq_hi", math.log(Q_DOC[1]))
    return lo, hi


def items_d(d, ctx, tol=None):
    tol = tol or {}
    lq_lo, lq_hi = _qset(d)
    e1 = math.exp(d["le_prod"])
    grid = np.linspace(lq_lo, lq_hi, 201)
    stable4 = [ctx["x4"] <= math.exp(g) * e1 for g in grid]
    stable5 = [ctx["x5"] <= math.exp(g) * e1 for g in grid]

    def loss(v):
        if v > e1:
            return 10.0                                  # diverged: worst outcome
        u = math.log(v) - d["leta_prod"]
        return (d["k_lo"] if u < 0 else d["k_hi"]) * u * u

    losses = {k: loss(v) for k, v in ctx["opts"].items()}
    return [
        Q.point("q1", "log10 of the learning rate above which the production run (N=7e9, D=1.4e11, batch 0.5M, warmup "
                      "0.02, qk-layernorm off) diverges.", "log10(lr)", d["le_prod"] / LN10, tol.get("q1", 0.05),
                cards=["C4"], obstacles=["O3", "O9"], floor=0.025),
        Q.interval("q2", "log10 of the learning rate above which the same production run diverges with qk-layernorm on.",
                   "log10(lr)", (lq_lo + d["le_prod"]) / LN10, (lq_hi + d["le_prod"]) / LN10, tol.get("q2", 0.05),
                   cards=["C4"], obstacles=["O5"], floor=0.025),
        Q.verdict("q3", "The production run exactly as planned in the lab notes (qk-layernorm off, warmup 0.02, peak lr = "
                        "%.3g) diverges." % ctx["lr_notes"], [ctx["lr_notes"] > e1], cards=["C4"], obstacles=["O3"]),
        Q.verdict("q4", "The production run with qk-layernorm on (warmup 0.02) and peak lr = %.3g does not diverge."
                  % ctx["x4"], stable4, cards=["C4"], obstacles=["O5"]),
        Q.verdict("q5", "The production run with qk-layernorm on (warmup 0.02) and peak lr = %.3g does not diverge."
                  % ctx["x5"], stable5, cards=["C4"], obstacles=["O5"]),
        Q.decision("q6", "Keeping qk-layernorm off and warmup 0.02, which peak learning rate gives the production run "
                         "the lowest final loss?  A run that diverges counts as the worst possible outcome.  Options: "
                   + ", ".join("%s: %.3g" % (k, v) for k, v in ctx["opts"].items()),
                   list(ctx["opts"]), losses, tol.get("q6", 0.004), cards=["C2", "C4"], obstacles=["O3"]),
        Q.verdict("q7", "With qk-layernorm off but warmup raised to 5%% of steps (wu=0.05), the production run at peak "
                        "lr = %.3g does not diverge." % ctx["x7"],
                  [ctx["x7"] <= math.exp(d["le_hot"])], cards=["C4"], obstacles=["O9"]),
    ]


def items(p, ctx, tol=None):
    return items_d(derived(p), ctx, tol)


def answers_from(d, ctx, collapse=False):
    out = {}
    for it in items_d(d, dict(ctx)):
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


# ------------------------------------------------------------------------------------------ oracle
def _nb_edge_guess(rows, N, wu):
    """Rough edge at (N, wu) from the notebook: bracket at the largest size with a divergence, then a generic
    slope of 0.5 in N and no warmup correction.  Only used to start a bracket search."""
    best = None
    for n in sorted(NB_SIZES, reverse=True):
        rr = [r for r in rows if abs(r["config"]["N"] - n) < 1 and r["config"]["qk"] == 0]
        ok = [r["config"]["lr"] for r in rr if r["status"] == "ok"]
        dv = [r["config"]["lr"] for r in rr if r["status"] == "diverged"]
        if ok and dv:
            best = (n, math.sqrt(max(ok) * min(dv))); break
    if best is None:
        n = NB_SIZES[-1]
        best = (n, 2 * max(r["config"]["lr"] for r in rows if abs(r["config"]["N"] - n) < 1))
    return best[1] * (N / best[0]) ** (-0.5)


def find_edge(sess, N, wu, guess, steps, seed0, spread=1.5):
    """Bracket then bisect (in ln lr) the qk-off divergence edge at (N, wu) with cheap short runs."""
    seed = [seed0]

    def div(lr):
        r = sess.run({"N": N, "D": D_MIN, "lr": float(min(LR_MAX, lr)), "wu": wu, "qk": 0, "seed": seed[0]})
        seed[0] += 1
        return r["status"] == "diverged"

    lo, hi = guess / spread, min(LR_MAX, guess * spread)
    while not div(hi):
        if hi >= LR_MAX:
            return None                                   # censored: never diverges in the lab
        lo = hi; hi = min(LR_MAX, hi * spread ** 2)
    while div(lo):
        hi = lo; lo = lo / spread ** 2
    for _ in range(steps):
        m = math.sqrt(lo * hi)
        if div(m):
            hi = m
        else:
            lo = m
    return math.log(math.sqrt(lo * hi))


def _diag_eta(rows):
    """ln eta*(N) along D = 20 N: grid over (a, g) with an inner linear fit of per-size floors and k_lo, k_hi."""
    ok = [r for r in rows if r["status"] == "ok"]
    Ns = sorted({r["config"]["N"] for r in ok})
    x = np.log([r["config"]["lr"] for r in ok]); ln = np.log([r["config"]["N"] / 1e8 for r in ok])
    y = np.array([r["loss"] for r in ok])
    Ind = np.array([[1.0 if abs(r["config"]["N"] - n) < 1 else 0.0 for n in Ns] for r in ok])

    def sse(a, g):
        u = x - (a - g * ln)
        X = np.column_stack([Ind, np.where(u < 0, u * u, 0.0), np.where(u >= 0, u * u, 0.0)])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((X @ c - y) ** 2).sum()), c

    best = None
    A = np.linspace(math.log(3e-4), math.log(3e-2), 61); G = np.linspace(-0.1, 0.6, 36)
    for _ in range(3):
        for a in A:
            for g in G:
                s, c = sse(a, g)
                if best is None or s < best[0]:
                    best = (s, a, g, c)
        da, dg = A[1] - A[0], G[1] - G[0]
        A = np.linspace(best[1] - 2 * da, best[1] + 2 * da, 21); G = np.linspace(best[2] - 2 * dg, best[2] + 2 * dg, 21)
    _, a, g, c = best
    return a, g, max(float(c[-2]), 1e-4), max(float(c[-1]), 1e-4)


def oracle(sess, rows_nb, ctx, rng, drop=None):
    drop = drop or {}
    # 1) qk-off edge at the largest lab size, production warmup (bracket from the notebook, generic slope)
    g = _nb_edge_guess(rows_nb, N_MAX, PROD_WU)
    l_max = find_edge(sess, N_MAX, PROD_WU, g, 6, 100, spread=1.6)
    # 2) same warmup at the smallest notebook size -> delta;  3) zero and 5% warmup at the largest size
    l_small = find_edge(sess, NB_SIZES[0], PROD_WU, math.exp(l_max) * (N_MAX / NB_SIZES[0]) ** 0.5, 6, 200, spread=1.6)
    if "omega" in drop:                                    # ablation: assume warmup does not move the edge
        l_min = l_hot = l_max
    else:
        l_min = find_edge(sess, N_MAX, 0.0, math.exp(l_max) / 1.4, 6, 300, spread=1.4)
        l_hot = find_edge(sess, N_MAX, HOT_WU, math.exp(l_max) * 1.2, 6, 400, spread=1.4)
    # 4) the qk-on run at the least stable corner the lab allows
    r = sess.run({"N": N_MAX, "D": D_MIN, "lr": LR_MAX, "wu": 0.0, "qk": 1, "seed": 500})
    delta = drop.get("delta", (l_small - l_max) / math.log(N_MAX / NB_SIZES[0]))
    shift = -delta * math.log(PROD_N / N_MAX)
    d = {"le_prod": l_max + shift, "le_hot": l_hot + shift, "le_min": l_min}
    if r["status"] == "diverged":                          # not reachable for admissible draws; keep it honest
        d["lq_lo"] = d["lq_hi"] = math.log(LR_MAX) - d["le_min"]
    a, gg, klo, khi = _diag_eta(rows_nb)
    d["leta_prod"] = a - gg * math.log(PROD_N / 1e8)
    d["k_lo"], d["k_hi"] = (0.0, 0.0) if "k" in drop else (klo, khi)
    return answers_from(d, ctx), d


# ------------------------------------------------------------------------------------------ rivals
def rivals(p, rows_nb, ctx, rng):
    """Each rival measures perfectly (true edges) and errs only in its reasoning."""
    d = derived(p); pf = W.full(p); out = {}
    tb = dict(textbook(CARDS)); tb["gB"] = 0.0
    out["B0_textbook"] = answers_from(derived(tb), ctx)
    # B1: the notes' reading - the edge does not move with scale and qk-layernorm does nothing
    b1 = dict(d, le_prod=math.log(_edge(pf, N_MAX, PROD_WU)), le_hot=math.log(_edge(pf, N_MAX, HOT_WU)), lq_lo=0.0, lq_hi=0.0)
    out["B1_notes_reading"] = answers_from(b1, ctx)
    # B2: correct edges, but the qk factor taken from the documentation alone (ignores what the lab rules out)
    out["B2_doc_range_only"] = answers_from(dict(d, lq_lo=math.log(Q_DOC[0])), ctx)
    # B3: lower bound from qk-on runs at the production warmup only (not the least stable lab corner)
    out["B3_bound_at_prod_warmup"] = answers_from(dict(d, lq_lo=math.log(LR_MAX / _edge(pf, N_MAX, PROD_WU))), ctx)
    # B4: lower bound from a mid-size qk-on run (1e8 params, zero warmup)
    out["B4_bound_at_mid_size"] = answers_from(dict(d, lq_lo=math.log(LR_MAX / _edge(pf, 1e8, 0.0))), ctx)
    # B5: 'qk-on never diverged up to the cap, so the edge is the cap-implied bound' - lower bound read as a point
    lq = math.log(LR_MAX) - d["le_min"]
    out["B5_bound_as_point"] = answers_from(dict(d, lq_lo=lq, lq_hi=lq), ctx)
    # B6: qk-on probed only up to 60% of the LR cap
    out["B6_probe_below_cap"] = answers_from(dict(d, lq_lo=math.log(0.6 * LR_MAX) - d["le_min"]), ctx)
    # B7: correct scale dependence, but warmup assumed not to move the edge (qk bound and 5% warmup both off)
    out["B7_no_warmup_effect"] = answers_from(dict(d, le_hot=d["le_prod"], le_min=math.log(_edge(pf, N_MAX, PROD_WU))), ctx)
    return out


def rival_designs(p, rows_nb, rng):
    return {}


DROP = {"C4:delta": {"delta": 0.0}, "C4:warmup": {"omega": 0.0}, "C2:bowl": {"k": 0.0}}
# B4 and B6 are weaker-bound shortcuts whose error depends on the draw; reported, not gated.
INFO_RIVALS = {"B4_bound_at_mid_size", "B6_probe_below_cap"}


def public_values(ctx):
    return [ctx["lr_notes"], ctx["team_slope"], ctx["team_icpt"], ctx["x4"], ctx["x5"], ctx["x7"]] + list(ctx["opts"].values())
