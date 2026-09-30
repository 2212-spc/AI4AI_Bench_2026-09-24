"""EvalLab: a simulated *evaluation* service (benchmark scoring, LLM-judge comparisons, arena games).

Cost unit: **eval credits** = one model call on one benchmark item.  Per-call cost differs by model, so
a budget buys very different amounts of evidence depending on the design.

The world is scalar-parametrised: the item bank is generated procedurally from `bank_seed`, so
`hidden/world.json` stays small and every key is recomputable.

Services (`svc` knob)
  score   accuracy of one model on one split under one prompt format, optionally repeated
  judge   pairwise LLM-judge comparison of two models on one split, in one or both presentation orders
  arena   pairwise "games" between two models judged by the same judge, reported as win counts

Mechanism cards implemented here (ids as in cards.py, which is authoritative): E1 (2PL response),
E2 (format / answer-extraction), E3 (contamination memorisation), E4 (judge bias), E5 (Bradley-Terry
ratings and Ford's condition), E6 (two-level eval variance: between-call ability jitter),
E7 (scored-denominator aggregation / Simpson), E8 (n-gram contamination detector with calibration error).
"""
import hashlib, math
import numpy as np

NAME = "evallab"
COST_UNIT = "eval credits"
COST_TEXT = "credits = items x repetitions x per-call cost of the model(s) involved"
EXTRA_FIELDS = ()

# Gauss-Hermite nodes for the between-call ability jitter (5 nodes, weight-normalised)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(5)
_GH_W = _GH_W / _GH_W.sum()

MODEL = dict(theta=0.0, kappa=0.0, length=300.0, fmt={}, slice_off=None, ext=0.0, cost=1.0, family="x")

BASE = dict(
    n_items=600, n_slices=3, bank_seed=1,
    a_mu=1.25, a_sd=0.30, b_mu=0.0, b_sd=0.95, slice_b=None,   # E1
    dup_frac=0.0, dup_lam=6.0,                                  # E3
    fmt_ext=None,                                               # E2  fmt -> {"base": e, "by_slice": [...]}
    fmt_kappa=None,                                             # E3  fmt -> multiplier on memorisation
    models=None,
    judge=None,                                                 # E4
    sig_call=0.0,                                               # E6 between-call ability jitter (in theta units)
    tie_rate=0.0,                                               # E5
    det_a=1.1, det_b=-2.2, det_s=0.9, det_fast_cost=0.05, det_scan_cost=40.0,   # E8 contamination detector
)
JUDGE = dict(cq=2.2, dpos=0.0, phi=0.0, psi=0.0, sigl=120.0, family="", cost=1.0)
FMTS = ("mc_letter", "mc_cloze", "free_exact", "free_judge")


def full(p):
    out = dict(BASE); out.update(p or {})
    out["slice_b"] = list(out["slice_b"] or [0.0] * out["n_slices"])
    out["fmt_ext"] = dict(out["fmt_ext"] or {})
    out["fmt_kappa"] = dict(out["fmt_kappa"] or {})
    out["judge"] = dict(JUDGE, **(out["judge"] or {}))
    ms = {}
    for k, v in (out["models"] or {}).items():
        m = dict(MODEL); m.update(v)
        m["fmt"] = dict(m["fmt"] or {})
        m["slice_off"] = list(m["slice_off"] or [0.0] * out["n_slices"])
        ms[k] = m
    out["models"] = ms
    return out


# ----------------------------------------------------------------------------------- item bank
_BANK = {}


def _h(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def bank(p):
    """Item parameters, generated from bank_seed alone: (a, b, slice, dup)."""
    key = (p["bank_seed"], p["n_items"], p["n_slices"], p["a_mu"], p["a_sd"], p["b_mu"], p["b_sd"],
           p["dup_frac"], p["dup_lam"])
    if key not in _BANK:
        n = int(p["n_items"]); rng = np.random.default_rng(_h("bank", *key))
        a = np.clip(rng.lognormal(math.log(max(p["a_mu"], 1e-3)), p["a_sd"], n), 0.25, 5.0)
        b = rng.normal(p["b_mu"], p["b_sd"], n)
        g = rng.integers(0, int(p["n_slices"]), n)
        dup = np.where(rng.random(n) < p["dup_frac"], rng.poisson(p["dup_lam"], n) + 1, 0)
        _BANK[key] = (a, b, g, dup.astype(float))
    return _BANK[key]


def subset(p, split, n):
    """Deterministic nested prefixes of a fixed permutation, so every design is paired across models."""
    a, b, g, dup = bank(p)
    idx = np.arange(len(a))
    if split != "all":
        idx = idx[g == int(str(split).lstrip("s"))]
    order = np.random.default_rng(_h("perm", p["bank_seed"], split)).permutation(len(idx))
    idx = idx[order]
    return idx[:int(min(n, len(idx)))]


def n_available(p, split):
    a, b, g, dup = bank(p)
    return int(len(a) if split == "all" else (g == int(str(split).lstrip("s"))).sum())


# ----------------------------------------------------------------------------------- response model
def extract_ok(p, m, fmt, idx):
    """E2: deterministic per (item, model, format) answer-extraction success."""
    fe = p["fmt_ext"].get(fmt)
    if not fe:
        return np.ones(len(idx), dtype=bool)
    a, b, g, dup = bank(p)
    by = fe.get("by_slice") or [0.0] * p["n_slices"]
    rate = np.clip(fe.get("base", 0.0) + np.array([by[int(x)] for x in g[idx]]) + p["models"][m]["ext"], 0.0, 0.95)
    u = np.array([((_h("ext", p["bank_seed"], m, fmt, int(j)) % 10 ** 9) + 0.5) / 10 ** 9 for j in idx])
    return u >= rate


def p_correct(p, m, fmt, idx, dtheta=0.0):
    """E1 + E3: per-item probability of being scored correct, given the item is extractable."""
    a, b, g, dup = bank(p)
    mm = p["models"][m]
    off = np.array([mm["slice_off"][int(x)] for x in g[idx]])
    sb = np.array([p["slice_b"][int(x)] for x in g[idx]])
    th = mm["theta"] + off + mm["fmt"].get(fmt, 0.0) + dtheta
    pc = 1.0 / (1.0 + np.exp(-a[idx] * (th - b[idx] - sb)))
    kap = mm["kappa"] * p["fmt_kappa"].get(fmt, 1.0)
    if kap > 0:
        mem = 1.0 - np.exp(-kap * dup[idx])
        pc = pc + (1.0 - pc) * mem
    return np.clip(pc, 0.0, 1.0)


def score_truth(p, m, split, fmt, n=None, clean_only=False, denom="scored"):
    """Noise-free accuracy of `svc=score` with infinite repetitions (the quantity keys are built from).

    denom="scored"   : correct / extractable items      (what the service reports)
    denom="attempted": correct / all items in the split  (extraction failures counted wrong)
    """
    idx = subset(p, split, n if n is not None else 10 ** 9)
    ok = extract_ok(p, m, fmt, idx)
    if clean_only:
        a, b, g, dup = bank(p)
        ok = ok & (dup[idx] == 0)
    sel = idx[ok] if denom == "scored" else idx
    if len(sel) == 0:
        return float("nan"), 0
    pc = np.zeros(len(sel))
    for x, w in zip(_GH_X, _GH_W):
        pc += w * p_correct(p, m, fmt, sel, dtheta=p["sig_call"] * x)
    if denom == "attempted":
        pc = pc * extract_ok(p, m, fmt, sel)
    return float(pc.mean()), int(len(sel))


def score_truth_slice(p, m, split, fmt, n, sl):
    """Noise-free accuracy restricted to the items of one slice *within the requested design*."""
    idx = subset(p, split, n if n is not None else 10 ** 9)
    a, b, g, dup = bank(p)
    idx = idx[g[idx] == int(sl)]
    ok = extract_ok(p, m, fmt, idx)
    sel = idx[ok]
    if len(sel) == 0:
        return float("nan"), 0
    pc = np.zeros(len(sel))
    for x, w in zip(_GH_X, _GH_W):
        pc += w * p_correct(p, m, fmt, sel, dtheta=p["sig_call"] * x)
    return float(pc.mean()), int(len(sel))


def _between_shift(p, m, split, fmt, n, sl):
    """One-sigma accuracy shift a between-call ability jitter produces on one slice of a design."""
    if p["sig_call"] <= 0:
        return 0.0
    idx = subset(p, split, n if n is not None else 10 ** 9)
    a, b, g, dup = bank(p)
    idx = idx[g[idx] == int(sl)]
    sel = idx[extract_ok(p, m, fmt, idx)]
    if len(sel) == 0:
        return 0.0
    return abs(0.5 * float(p_correct(p, m, fmt, sel, dtheta=p["sig_call"]).mean()
                           - p_correct(p, m, fmt, sel, dtheta=-p["sig_call"]).mean()))


def score_sd(p, m, split, fmt, n, reps=1):
    """E6: sd of one reported accuracy = sqrt(within/(n_scored*reps) + between/reps)."""
    idx = subset(p, split, n)
    ok = extract_ok(p, m, fmt, idx)
    sel = idx[ok]
    if len(sel) == 0:
        return float("nan")
    pc = p_correct(p, m, fmt, sel)
    within = float((pc * (1 - pc)).mean()) / len(sel)
    if p["sig_call"] > 0:
        d = 0.5 * (p_correct(p, m, fmt, sel, dtheta=p["sig_call"]).mean()
                   - p_correct(p, m, fmt, sel, dtheta=-p["sig_call"]).mean())
        between = float(d) ** 2
    else:
        between = 0.0
    return math.sqrt(max(within + between, 1e-12) / max(int(reps), 1))


# ----------------------------------------------------------------------------------- judge / arena
def _len_of(p, m, idx):
    mm = p["models"][m]
    return mm["length"] * (1.0 + 0.0 * idx)


def judge_p(p, a_m, b_m, idx, order):
    """E4: probability the judge prefers a_m, item by item, in presentation `order`."""
    J = p["judge"]
    qa = p_correct(p, a_m, "free_judge", idx); qb = p_correct(p, b_m, "free_judge", idx)
    la = p["models"][a_m]["length"]; lb = p["models"][b_m]["length"]
    d = J["cq"] * (qa - qb) + J["phi"] * math.tanh((la - lb) / max(J["sigl"], 1e-9))
    if order == "ab":
        d = d + J["dpos"]
    elif order == "ba":
        d = d - J["dpos"]
    d = d + J["psi"] * ((1.0 if p["models"][a_m]["family"] == J["family"] else 0.0)
                        - (1.0 if p["models"][b_m]["family"] == J["family"] else 0.0))
    return 1.0 / (1.0 + np.exp(-d))


def judge_truth(p, a_m, b_m, split, n=None, order="both"):
    idx = subset(p, split, n if n is not None else 10 ** 9)
    if order == "both":
        q = 0.5 * (judge_p(p, a_m, b_m, idx, "ab") + judge_p(p, a_m, b_m, idx, "ba"))
    else:
        q = judge_p(p, a_m, b_m, idx, order)
    return float(q.mean() * (1 - p["tie_rate"]) + 0.5 * p["tie_rate"])


def bt_p(p, i, j):
    """E5: Bradley-Terry win probability from the judge's mean preference over the whole bank."""
    return judge_truth(p, i, j, "all", None, "both")


# ----------------------------------------------------------------------------------- E8 corpus detector
def det_fast(p, idx):
    """A cheap n-gram overlap score per item: a fixed monotone function of how often the item occurs in
    the pretraining corpus, plus item-level detector noise.  Deterministic (a property of the item)."""
    a, b, g, dup = bank(p)
    z = np.array([(_h("det", p["bank_seed"], int(j)) % 10 ** 9 + 0.5) / 10 ** 9 for j in idx])
    z = np.sqrt(2.0) * _erfinv(2.0 * z - 1.0)                       # deterministic standard normal
    raw = p["det_b"] + p["det_a"] * np.log1p(dup[idx]) + p["det_s"] * z
    return 1.0 / (1.0 + np.exp(-raw))


def _erfinv(y):
    y = np.clip(np.asarray(y, dtype=float), -1 + 1e-12, 1 - 1e-12)
    a = 0.147
    ln = np.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t)


def clean_frac(p, split, n=None):
    """Fraction of the requested design that occurs zero times in the corpus (the E3/E8 target)."""
    idx = subset(p, split, n if n is not None else 10 ** 9)
    a, b, g, dup = bank(p)
    return float((dup[idx] == 0).mean())


# ----------------------------------------------------------------------------------- backend protocol
def check(sess, cfg):
    from ..lab import LabError
    p = sess.p; svc = cfg.get("svc", "score")
    if svc == "score":
        if cfg["model"] not in p["models"]:
            raise LabError("unknown model %r" % cfg["model"])
        av = n_available(p, cfg.get("split", "all"))
        if cfg["n"] > av:
            raise LabError("split %s has only %d items" % (cfg.get("split", "all"), av))
    elif svc == "corpus":
        av = n_available(p, cfg.get("split", "all"))
        if cfg["n"] > av:
            raise LabError("split %s has only %d items" % (cfg.get("split", "all"), av))
    else:
        for k in ("a", "b"):
            if cfg.get(k) not in p["models"]:
                raise LabError("unknown model %r for %s" % (cfg.get(k), k))
        if cfg["a"] == cfg["b"]:
            raise LabError("a and b must be different models")
        av = n_available(p, cfg.get("split", "all"))
        if svc == "judge" and cfg["n"] > av:
            raise LabError("split %s has only %d items" % (cfg.get("split", "all"), av))


def extras(sess, req, cfg):
    return {}


def cost(sess, cfg, extra):
    p = sess.p; svc = cfg.get("svc", "score")
    if svc == "score":
        return float(cfg["n"]) * max(int(cfg.get("reps", 1)), 1) * p["models"][cfg["model"]]["cost"]
    if svc == "corpus":
        per = p["det_scan_cost"] if cfg.get("mode", "fast") == "scan" else p["det_fast_cost"]
        return float(cfg["n"]) * float(per)
    if svc == "judge":
        k = 2 if cfg.get("order", "both") == "both" else 1
        c = p["models"][cfg["a"]]["cost"] + p["models"][cfg["b"]]["cost"] + p["judge"]["cost"] * k
        return float(cfg["n"]) * c
    g = float(cfg.get("games", 0))
    c = p["models"][cfg["a"]]["cost"] + p["models"][cfg["b"]]["cost"] + 2 * p["judge"]["cost"]
    return g * c


def _rng(sess, cfg, seed, tag):
    key = repr(sorted((k, (round(float(v), 12) if isinstance(v, (int, float)) else v)) for k, v in cfg.items()))
    return np.random.default_rng(_h(sess.salt, key, int(seed), tag))


def execute(sess, cfg, seed, extra=None):
    p = sess.p; svc = cfg.get("svc", "score")
    out = {"svc": svc, "config": dict(cfg, seed=seed), "status": "ok"}
    if svc == "score":
        m = cfg["model"]; fmt = cfg.get("fmt", "mc_letter"); split = cfg.get("split", "all")
        n = int(cfg["n"]); reps = max(int(cfg.get("reps", 1)), 1)
        idx = subset(p, split, n); ok = extract_ok(p, m, fmt, idx)
        sel = idx[ok]
        a, b, g, dup = bank(p)
        rng = _rng(sess, cfg, seed, "score")
        accs = []; per_slice = None; detail = []
        for r in range(reps):
            z = rng.normal() if p["sig_call"] > 0 else 0.0
            pc = p_correct(p, m, fmt, sel, dtheta=p["sig_call"] * z)
            corr = rng.random(len(sel)) < pc
            accs.append(round(float(corr.mean()) if len(sel) else float("nan"), 6))
            if r == 0:
                gs = g[sel]; ga = g[idx]
                per_slice = [{"slice": int(s), "n_items": int((ga == s).sum()),
                              "n_scored": int((gs == s).sum()), "n_correct": int(corr[gs == s].sum())}
                             for s in range(int(p["n_slices"]))]
                if int(cfg.get("detail", 0)):
                    pos = {int(j): i for i, j in enumerate(sel)}       # O(n): `sel.index(j)` made this O(n^2)
                    detail = [{"item": int(j), "slice": int(g[j]), "scored": bool(o),
                               "correct": (bool(corr[pos[int(j)]]) if o else None)}
                              for j, o in zip(idx, ok)]
        out.update({"model": m, "fmt": fmt, "split": split, "n_items": int(len(idx)), "n_scored": int(len(sel)),
                    "acc": round(float(np.mean(accs)), 6), "reps": reps, "reps_acc": accs, "by_slice": per_slice})
        if detail:
            out["detail"] = detail
        return out
    if svc == "corpus":
        split = cfg.get("split", "all"); n = int(cfg["n"]); mode = cfg.get("mode", "fast")
        idx = subset(p, split, n)
        a, b, g, dup = bank(p)
        if mode == "scan":
            items = [{"item": int(j), "occurrences": int(dup[j])} for j in idx]
        else:
            ov = det_fast(p, idx)
            items = [{"item": int(j), "overlap": round(float(v), 6)} for j, v in zip(idx, ov)]
        out.update({"mode": mode, "split": split, "n": int(len(idx)), "items": items})
        return out
    if svc == "judge":
        am, bm = cfg["a"], cfg["b"]; split = cfg.get("split", "all"); n = int(cfg["n"])
        order = cfg.get("order", "both"); idx = subset(p, split, n)
        rng = _rng(sess, cfg, seed, "judge")
        orders = ["ab", "ba"] if order == "both" else [order]
        rec = {}
        for o in orders:
            q = judge_p(p, am, bm, idx, o)
            u = rng.random(len(idx)); t = rng.random(len(idx)) < p["tie_rate"]
            wa = int(((u < q) & ~t).sum()); wb = int(((u >= q) & ~t).sum()); ti = int(t.sum())
            rec[o] = {"wins_a": wa, "wins_b": wb, "ties": ti, "n": int(len(idx))}
        tot = {"wins_a": sum(v["wins_a"] for v in rec.values()), "wins_b": sum(v["wins_b"] for v in rec.values()),
               "ties": sum(v["ties"] for v in rec.values()), "n": sum(v["n"] for v in rec.values())}
        out.update({"a": am, "b": bm, "split": split, "order": order, "by_order": rec, **tot,
                    "win_rate_a": round(tot["wins_a"] / max(tot["n"], 1), 6)})
        return out
    am, bm = cfg["a"], cfg["b"]; games = int(cfg.get("games", 0))
    rng = _rng(sess, cfg, seed, "arena")
    q = bt_p(p, am, bm)
    w = int(rng.binomial(games, q))
    out.update({"a": am, "b": bm, "games": games, "wins_a": w, "wins_b": games - w,
                "win_rate_a": round(w / max(games, 1), 6)})
    return out


# ----------------------------------------------------------------------------------- consistency model
def predict_row(p, row):
    """Noise-free prediction + sd for every observable a row reports.  Used by the counterexample
    verifier (verify.py): a witness world must reproduce every observed number within the stated bar."""
    cfg = row.get("config", {}); svc = row.get("svc", cfg.get("svc", "score"))
    out = {}
    if svc == "corpus":
        idx = [it["item"] for it in row.get("items", [])]
        if row.get("mode") == "scan":
            a, b, g, dup = bank(p)
            for it in row.get("items", []):
                out["occ%d" % it["item"]] = (float(dup[it["item"]]), 0.0)
        else:
            ov = det_fast(p, np.array(idx, dtype=int)) if idx else []
            for it, v in zip(row.get("items", []), ov):
                out["ovl%d" % it["item"]] = (float(v), 0.0)
        return out
    if svc == "score":
        m = cfg["model"]; fmt = cfg.get("fmt", "mc_letter"); split = cfg.get("split", "all")
        n = int(cfg["n"]); reps = max(int(cfg.get("reps", 1)), 1)
        mu, ns = score_truth(p, m, split, fmt, n)
        if not (ns == row.get("n_scored")):
            return {"n_scored": (float(ns), 0.0)}          # structural mismatch: sd 0 -> infinite z
        # independent observables: rep-0 slice counts, then the accuracy of each later repetition
        sd1 = score_sd(p, m, split, fmt, n, 1)
        for s in (row.get("by_slice") or []):
            if s["n_scored"] > 0:
                msl, nsl = score_truth_slice(p, m, split, fmt, n, s["slice"])
                if nsl != s["n_scored"]:
                    return {"n_scored_slice%d" % s["slice"]: (float(nsl), 0.0)}
                out["slice%d" % s["slice"]] = (msl * nsl,
                                               math.sqrt(max(msl * (1 - msl), 1e-9) * nsl
                                                         + (_between_shift(p, m, split, fmt, n, s["slice"]) * nsl) ** 2))
        if not (row.get("by_slice") or []):
            out["acc"] = (mu, score_sd(p, m, split, fmt, n, reps))
        for i, a_i in enumerate(row.get("reps_acc") or []):
            if i:
                out["rep%d" % i] = (mu, sd1)
    elif svc == "judge":
        n = row["n"]
        by = row.get("by_order") or {}
        if by:
            for o, v in by.items():
                qo = judge_truth(p, cfg["a"], cfg["b"], cfg.get("split", "all"), int(cfg["n"]), o)
                out["wins_a_" + o] = (qo * v["n"], math.sqrt(max(qo * (1 - qo), 1e-9) * v["n"]))
        else:
            q = judge_truth(p, cfg["a"], cfg["b"], cfg.get("split", "all"), int(cfg["n"]), cfg.get("order", "both"))
            out["wins_a"] = (q * n, math.sqrt(max(q * (1 - q), 1e-9) * n))
    else:
        q = bt_p(p, cfg["a"], cfg["b"]); n = row["games"]
        out["wins_a"] = (q * n, math.sqrt(max(q * (1 - q), 1e-9) * n))
    return out


def observed(row):
    """The numbers predict_row is compared against."""
    svc = row.get("svc", "score")
    out = {}
    if svc == "corpus":
        k = "occurrences" if row.get("mode") == "scan" else "overlap"
        pre = "occ" if row.get("mode") == "scan" else "ovl"
        for it in row.get("items", []):
            out["%s%d" % (pre, it["item"])] = float(it[k])
    elif svc == "score":
        out["n_scored"] = row["n_scored"]
        by = row.get("by_slice") or []
        for s in by:
            if s["n_scored"] > 0:
                out["slice%d" % s["slice"]] = s["n_correct"]
        if not by:
            out["acc"] = row["acc"]
        for i, a_i in enumerate(row.get("reps_acc") or []):
            if i:
                out["rep%d" % i] = a_i
    elif svc == "judge":
        by = row.get("by_order") or {}
        if by:
            for o, v in by.items():
                out["wins_a_" + o] = v["wins_a"]
        else:
            out["wins_a"] = row["wins_a"]
    else:
        out["wins_a"] = row["wins_a"]
    return out


MANUAL = r"""## 1. What the service models (abstraction boundary)

EvalLab is a **simulated** language-model evaluation service.  No real model is ever called; every
request is answered by a simulator.  The simulator is not a replica of any real evaluation stack.  Its
mechanisms are modelled on regularities reported in the evaluation-methodology literature, but **its
constants were drawn fresh for this lab**.  Published numbers (item difficulties, judge agreement
rates, benchmark accuracies, contamination rates) describe other worlds.  Measure; do not recall.

Objects: a fixed **item bank** of benchmark questions, split into **slices**; a pool of **models**; one
**LLM judge**.  Each item has a difficulty and a discrimination; each model has an ability.  A
`score` request reports how many items were answered correctly; `judge` and `arena` requests report
pairwise preferences of the judge.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Item response.*  The chance a model answers an item correctly rises with its ability and falls with the item's difficulty; items differ in how sharply they separate models.
- *Slices.*  A model's ability may differ from slice to slice, and slices may differ in difficulty.
- *Prompt format.*  The same item under a different answer format can give a different result, and the shift can differ by model.
- *Answer extraction.*  Under some formats a model's answer cannot be parsed at all.  Such items are reported as *attempted but not scored*; whether they are counted in a denominator is the analyst's choice, not the service's.
- *Contamination.*  Some items may appear in some models' training data; the resulting advantage grows with exposure and can differ by format.
- *Contamination detection.*  A corpus service can report, per item, either a cheap n-gram **overlap score** or an expensive exact **occurrence count**.  The overlap score is a fixed monotone function of the occurrence count plus item-level detector noise, so it is neither a sound nor a complete test for contamination on any single item.
- *Judge bias.*  A judge's preference can depend on which answer is shown first, on answer length, and on whether the answer comes from its own model family, in addition to answer quality.
- *Ratings.*  Pairwise win rates can be summarised by a Bradley-Terry rating per model; whether the ratings are determined by a set of games depends on which pairs were played.
- *Repetition.*  Repeated scoring of the same model on the same items differs from call to call.  Part of that variation is per-item sampling and part is a shift shared by every item in a call.
- *Selection.*  Nothing in the service selects for you.  If you report the best of several repetitions, that number is not an estimate of the underlying accuracy.

Not modelled: tokenizers, context length, latency, cost of the real API, prompt wording beyond the named formats.

## 2. Guarantees

- **Fixed bank.**  The item bank and its slice labels do not change.  `n` items means a fixed nested
  prefix of one fixed permutation of the split, so two models evaluated at the same `split`/`n` see the
  **same items**, and `n` = 200 is a subset of `n` = 400.
- **Fixed laws.**  Every mechanism is a fixed law whose constants do not depend on which request you
  make.  There are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  An identical request with the same seed returns identical numbers (and is charged
  again).  Different seeds give independent draws.
- **Notebook rows are real service responses.**  The team's notes that accompany them are the team's
  interpretation and may be wrong.
"""

CLI_HELP = r"""## 4. Using the service

    lab spec                                              # knobs, fixed settings, caps (free)
    lab run svc=score model=M1 split=all fmt=mc_letter n=200 reps=1 seed=0
    lab run svc=judge a=M1 b=M2 split=all n=150 order=both seed=1
    lab run svc=arena a=M1 b=M3 games=200 seed=2
    lab run svc=corpus mode=fast n=600 seed=0                 # cheap overlap score for every item
    lab run svc=corpus mode=scan n=60 seed=0                  # exact occurrence counts (expensive)
    lab batch plan.json                                   # a JSON list of requests, run in order
    lab status                                            # credits used / left (free)
    lab history                                           # every request you have made (free)

`score` reports `n_items` (items presented), `n_scored` (items whose answer could be extracted),
`acc` (mean over repetitions of correct / n_scored), `reps_acc` (one accuracy per repetition) and
`by_slice` counts for the first repetition.  Add `detail=1` for per-item outcomes.
`judge` reports win counts, and `by_order` separately for each presentation order.
`corpus` reports one row per item: `overlap` in [0,1] under `mode=fast`, or the exact `occurrences`
count under `mode=scan`.  Both modes address the same fixed corpus and are deterministic: re-asking
returns the same numbers, so repetition buys nothing.  Items are named by index and item indices are
stable across every service, so `corpus` rows can be joined to `score detail=1` rows.
A request above the per-request cap, beyond the request limit, or beyond the remaining budget is
refused and not charged.  Results are also appended to `/app/lab_runs.jsonl`.
"""
