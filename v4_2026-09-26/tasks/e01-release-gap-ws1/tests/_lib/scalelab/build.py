"""Instance builder: world draw -> notebook -> keys -> tolerance calibration -> rival kill matrix -> gates.

Every number the gates certify is produced by the same Session code the lab server runs.

Tolerance rule (per numeric item):  T = max(floor, 2.25 * p90(oracle endpoint error over n_cal replicates)).
Then n_ver *fresh* replicates must pass every item >= 90% of the time (solvability), and every must-kill
rival must fail at least one item *with margin*: a numeric endpoint off by >= 2T, or a wrong categorical
answer on an item the oracle gets right >= 90% of the time.

Replicates differ only in the noise salt of the oracle's *own* runs; the notebook is fixed (it is what the
agent is given).  The shipped instance uses its own salt; the oracle is also run on that salt and must pass.
"""
import hashlib, importlib, json, math, re
import numpy as np
from . import world as W
from . import queries as Q
from . import labs
from . import ctx as CTX
from .lab import Session, LabError
from .cards import CARDS

ALL_BLUEPRINTS = []   # filled by suite.py


def sid(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def load(name):
    return importlib.import_module("scalelab.bp." + name)


def make_world(bp, ws):
    # WORLD_ID lets a controlled variant (e.g. a tight-budget ablation) reuse its base blueprint's world,
    # notebook and noise salt exactly, so matched world seeds differ only in what the variant changes.
    wid = getattr(bp, "WORLD_ID", bp.ID)
    rng = np.random.default_rng(sid(wid, ws, "world"))
    p = bp.draw(rng); sp = bp.spec(p)
    pf = labs.backend(sp.get("lab", "pretrain")).full(p)
    salt = "%016x" % sid(wid, ws, "salt")
    sess = Session(pf, sp, salt)
    rows, notes, ctx = bp.notebook(pf, sess, rng)
    # A blueprint may ship extra files the agent reads (an audit task's analysis script, its logs and the
    # team's reported number).  They are part of the shipped text, so the leakage scanner sees them too.
    files = getattr(bp, "files", lambda *a: {})(pf, ctx, rng)
    return dict(bp=bp, ws=ws, p=p, pf=pf, spec=sp, salt=salt, rows=rows, notes=notes, ctx=ctx, files=files)


def grading_ctx(w, items):
    """The context the cex/prereg graders need, built from this instance's own world and rows.

    Built once per instance and reused by calibration, the kill matrix and the gates, so everything the
    certificate asserts was checked by the same code the exported grader runs."""
    return CTX.make_ctx(w["pf"], w["spec"], w["salt"], w["rows"], items,
                        getattr(w["bp"], "claim_fn", None))


def item_error(it, ans, gctx=None):
    """Numeric: max endpoint error.  Categorical and executed forms: 0 if right else inf."""
    if ans is None:
        return math.inf
    if it["kind"] in ("point", "set"):
        try:
            return max(abs(float(ans["lo"]) - it["key"]["lo"]), abs(float(ans["hi"]) - it["key"]["hi"]))
        except Exception:
            return math.inf
    ok, _ = Q.grade_item(it, ans, gctx)
    return 0.0 if ok else math.inf


def oracle_rep(w, tag, drop=None):
    bp = w["bp"]
    sess = Session(w["pf"], w["spec"], w["salt"] + "/" + tag if tag else w["salt"])
    rng = np.random.default_rng(sid(bp.ID, w["ws"], "oracle", tag, json.dumps(drop, sort_keys=True)))
    ans, ph = bp.oracle(sess, w["rows"], w["ctx"], rng, drop=drop)
    return ans, ph, sess


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(0.9 * len(xs))) - 1)]


def calibrate(w, n_cal=10, n_ver=10):
    bp = w["bp"]
    base = bp.items(w["pf"], w["ctx"])
    gctx = grading_ctx(w, base)
    errs = {it["id"]: [] for it in base}
    for r in range(n_cal):
        ans, _, _ = oracle_rep(w, "cal%d" % r)
        for it in base:
            errs[it["id"]].append(item_error(it, ans.get(it["id"]), gctx))
    tol = {}
    for it in base:
        if it["kind"] in ("point", "set"):
            finite = [e for e in errs[it["id"]] if math.isfinite(e)]
            tol[it["id"]] = max(it.get("floor", 0.0), 2.25 * p90(finite) if finite else math.inf)
    items = bp.items(w["pf"], w["ctx"], tol=tol)
    for it in items:                                   # decision tolerance: only the best option passes
        if it["kind"] == "decision":
            regs = sorted(it["key"]["regret"].values())
            it["gap"] = regs[1]; it["r_tol"] = regs[1] / 3.0
    gctx = grading_ctx(w, items)
    ver = {it["id"]: [] for it in items}; allpass = []
    for r in range(n_ver):
        ans, _, _ = oracle_rep(w, "ver%d" % r)
        g = Q.grade(items, ans, gctx)
        for q, v in g["items"].items():
            ver[q].append(v["pass"])
        allpass.append(g["all_pass"])
    ship_ans, ship_ph, ship_sess = oracle_rep(w, "")          # the agent's own salt
    ship_grade = Q.grade(items, ship_ans, gctx)
    cal_acc = {q: float(np.mean([math.isfinite(e) and e <= (tol.get(q, 0) if q in tol else 0) for e in errs[q]]))
               for q in errs}
    return dict(items=items, tol=tol, cal_errs=errs, cal_acc=cal_acc, gctx=gctx,
                ver_pass={q: float(np.mean(v)) for q, v in ver.items()}, ver_allpass=float(np.mean(allpass)),
                ship_ans=ship_ans, ship_ph=ship_ph, ship_grade=ship_grade,
                ship_usage=ship_sess.status(), ship_log=ship_sess.log)


# ------------------------------------------------------------------------------------------ rivals
def _widen(items, ans, k=3.0):
    out = json.loads(json.dumps(ans))
    for it in items:
        if it["kind"] == "point" and it["id"] in out:
            out[it["id"]]["lo"] -= k * it["tol"]; out[it["id"]]["hi"] += k * it["tol"]
    return out


def rival_answers(w, cal):
    bp = w["bp"]; items = cal["items"]
    rng = np.random.default_rng(sid(bp.ID, w["ws"], "rivals"))
    out = {}
    for k, a in bp.rivals(w["pf"], w["rows"], w["ctx"], rng).items():
        out[k] = a
    for k, reqs in bp.rival_designs(w["pf"], w["rows"], rng).items():
        sess = Session(w["pf"], w["spec"], w["salt"] + "/design/" + k)
        from .common import run_rows
        own = run_rows(sess, reqs)
        out[k] = bp.answers_from(bp.fit_rows(w["rows"] + own, rng), w["ctx"])
    for label, fx in getattr(bp, "DROP", {}).items():
        out["drop:" + label], _, _ = oracle_rep(w, "drop", drop=fx)
    if any(it["kind"] == "set" for it in items):
        out["collapse_sets"] = bp.answers_from(cal["ship_ph"], w["ctx"], collapse=True)
    if any(it["kind"] == "point" for it in items):
        out["widen_points"] = _widen(items, cal["ship_ans"])
    return out


def kill_matrix(w, cal, rivals):
    items = cal["items"]; stable = {q: cal["ver_pass"][q] >= 0.9 for q in cal["ver_pass"]}
    info = set(getattr(w["bp"], "INFO_RIVALS", ()))
    gctx = cal["gctx"]
    mat = {}
    for name, ans in rivals.items():
        g = Q.grade(items, ans, gctx); per = {}; margin = []
        for it in items:
            e = item_error(it, ans.get(it["id"]), gctx)
            if it["kind"] in ("point", "set"):
                ratio = e / it["tol"] if it["tol"] > 0 else math.inf
                per[it["id"]] = {"pass": g["items"][it["id"]]["pass"], "err_over_T": round(ratio, 2) if math.isfinite(ratio) else None}
                if ratio >= 2.0:
                    margin.append(it["id"])
            else:
                # An executed form has no tolerance to be off by: a wrong witness, a plan that fails to
                # discriminate or a misplaced defect is a categorical miss, so the margin condition is
                # simply "fails an item the oracle passes reliably".
                a = ans.get(it["id"])
                per[it["id"]] = {"pass": g["items"][it["id"]]["pass"],
                                 "answer": (list(a.values())[0] if isinstance(a, dict) and len(a) == 1
                                            else g["items"][it["id"]]["why"][:120])}
                if not g["items"][it["id"]]["pass"] and stable[it["id"]]:
                    margin.append(it["id"])
        mat[name] = {"score": g["score"], "killed": not g["all_pass"], "margin_items": margin,
                     "killed_with_margin": bool(margin), "must_kill": name not in info, "items": per}
    return mat


# ------------------------------------------------------------------------- v4 difficulty and form gates
# A blueprint *claims* a difficulty and the gates below verify the claim, so that a knob recorded in
# difficulty.json is a measured property of the instance rather than the author's intention:
#
#   DIFFICULTY = {"depth": 4,                  longest chain the answer path needs (G11)
#                 "nuisance": ["E4", "E6"],    cards that must be modelled and divided out (G12)
#                 "anti_prior": ["q2"]}        items whose truth is opposite the literature prior (G13)
#
# Items may carry "chain" (the declared derivation steps) and "prior_key" (what the published constant
# would give).  A blueprint that claims nothing gets a vacuous pass on G11-G13, which keeps the v3
# pretraining blueprints green.
def _items_of(items, kind):
    return [it for it in items if it["kind"] == kind]


def _claim(bp, key, default=None):
    return (getattr(bp, "DIFFICULTY", {}) or {}).get(key, default)


def depth_gate(w, cal, mat):
    """G11: every declared derivation step is load-bearing.

    For each step of an item's declared chain there must be a rival that takes the shortcut of skipping
    exactly that step (`skip:<qid>:<step>` or `skip:<step>`) and fails the item with margin.  Depth that
    no shortcut rival tests is depth on paper only - the v2 lesson, where a chain the author had in mind
    was collapsible into one regression."""
    items = cal["items"]; miss = []; declared = {}
    for it in items:
        chain = list(it.get("chain") or [])
        if not chain:
            continue
        declared[it["id"]] = len(chain)
        for s in chain:
            cand = [k for k in mat if k in ("skip:%s:%s" % (it["id"], s), "skip:" + s)]
            if not cand or not any(it["id"] in mat[k]["margin_items"] for k in cand):
                miss.append("%s/%s" % (it["id"], s))
    want = _claim(w["bp"], "depth")
    got = max(declared.values()) if declared else 0
    ok = not miss and (want is None or got >= int(want))
    return {"pass": bool(ok), "declared_depth": declared, "max_depth": got, "claimed": want,
            "steps_without_a_killed_shortcut": miss}


def nuisance_gate(w, mat):
    """G12: each card claimed as a nuisance must be load-bearing for the *measurement* while an analyst
    who ignores it gets a wrong answer.  Distinct from G9: `drop:<card>` switches the mechanism off in
    the world, `naive_ignore:<card>` leaves it on and fails to model it."""
    bad = []
    for c in (_claim(w["bp"], "nuisance") or []):
        k = "naive_ignore:" + c
        if k not in mat or not mat[k]["killed_with_margin"]:
            bad.append(c)
    return {"pass": not bad, "nuisance_cards": list(_claim(w["bp"], "nuisance") or []),
            "not_killed": bad}


def anti_prior_gate(w, cal, mat):
    """G13: on an item claimed anti-prior, answering with the published constant must fail with margin.

    For a numeric item the prior answer must also be at least 2T away from the key, so that the item is
    not merely *un*supported by the literature but actually contradicted by this world."""
    items = {it["id"]: it for it in cal["items"]}
    rows = []; bad = []
    for q in (_claim(w["bp"], "anti_prior") or []):
        it = items.get(q)
        if it is None:
            bad.append("%s: no such item" % q); continue
        riv = [k for k in mat if k == "B_prior" or k.startswith("B_prior:")]
        killed = any(q in mat[k]["margin_items"] for k in riv)
        gap = None
        if it["kind"] in ("point", "set") and it.get("prior_key") is not None:
            pk = it["prior_key"]
            gap = max(abs(float(pk.get("lo", pk.get("hi"))) - it["key"]["lo"]),
                      abs(float(pk.get("hi", pk.get("lo"))) - it["key"]["hi"])) / max(it["tol"], 1e-300)
        rows.append({"item": q, "prior_rival_killed": bool(killed), "gap_over_T": None if gap is None else round(gap, 2)})
        if not killed or (gap is not None and gap < 2.0):
            bad.append(q)
    return {"pass": not bad, "items": rows, "failing": bad}


def _draw_witness_param(rng, sp):
    if sp.get("type") == "choice":
        return sp["values"][int(rng.integers(len(sp["values"])))]
    v = float(rng.uniform(sp["lo"], sp["hi"]))
    return int(round(v)) if sp.get("type") == "int" else v


def _witness_corners(sch):
    """Every single parameter railed to an end of its range with the others at the truth's own draw, plus
    the all-low / all-mid / all-high assignments.  These are the assignments an agent tries first; if one
    of them is already a counterexample the item is free."""
    keys = sorted(sch)
    mid = {k: (sch[k]["values"][0] if sch[k].get("type") == "choice"
               else (sch[k]["lo"] + sch[k]["hi"]) / 2.0) for k in keys}
    out = [dict(mid),
           {k: (sch[k]["values"][0] if sch[k].get("type") == "choice" else sch[k]["lo"]) for k in keys},
           {k: (sch[k]["values"][-1] if sch[k].get("type") == "choice" else sch[k]["hi"]) for k in keys}]
    for k in keys:
        for end in ("lo", "hi"):
            if sch[k].get("type") == "choice":
                continue
            c = dict(mid); c[k] = sch[k][end]; out.append(c)
    for c in out:
        for k in keys:
            if sch[k].get("type") == "int":
                c[k] = int(round(c[k]))
    return out


def witness_gate(w, cal, n=600, vol_max=0.10):
    """G14: witness existence, and the *size* of the witness set.

    For an `entailed` item: no consistent world in the declared box falsifies the claim - the answer
    "entailed" is then a fact about the box, certified by dense sampling and by every corner.
    For a `refutable` item: a counterexample exists but is rare (at most `vol_max` of the box), so it has
    to be located by reasoning about which direction the evidence leaves open, not by guessing.  The
    measured volume is recorded as a difficulty statistic.
    """
    from . import verify as V
    gctx = cal["gctx"]; be = gctx["backend"]; base = be.full(w["pf"])
    rows = w["rows"]                       # only the disclosed rows: what the agent starts from
    cf = getattr(w["bp"], "claim_fn", None)
    rng = np.random.default_rng(sid(w["bp"].ID, w["ws"], "witness"))
    out = []; ok_all = True
    for it in _items_of(cal["items"], "cex"):
        sch = it["schema"]["params"]
        if cf is None:
            out.append({"item": it["id"], "error": "blueprint has no claim_fn"}); ok_all = False; continue
        cands = [{k: _draw_witness_param(rng, sch[k]) for k in sch} for _ in range(n)] + _witness_corners(sch)
        n_cons = n_flip = n_both = 0; corner_hit = []
        n_corner = len(cands) - n
        for i, vals in enumerate(cands):
            p = V.apply_witness(base, it["schema"], vals)
            c, _ = V.consistency(be, p, rows, z_bar=it.get("z_bar", V.Z_BAR))
            try:
                f = not bool(cf(it["id"], p))
            except Exception as e:
                out.append({"item": it["id"], "error": "claim_fn failed: %s" % e}); ok_all = False; break
            n_cons += bool(c); n_flip += bool(f); n_both += bool(c and f)
            if c and f and i >= n:
                corner_hit.append(i - n)
        else:
            vol = n_both / float(len(cands))
            rec = {"item": it["id"], "entailed": bool(it["entailed"]), "n_samples": len(cands),
                   "consistent_frac": round(n_cons / float(len(cands)), 4),
                   "flip_frac": round(n_flip / float(len(cands)), 4),
                   "witness_volume": round(vol, 4), "n_corner_witnesses": len(corner_hit),
                   "n_corners": n_corner}
            if it["entailed"]:
                rec["pass"] = n_both == 0
            else:
                rec["pass"] = n_both > 0 and vol <= vol_max and not corner_hit
            ok_all = ok_all and rec["pass"]
            out.append(rec)
    return {"pass": bool(ok_all), "vol_max": vol_max, "items": out}


def plan_gate(w, cal, mat, n_salt=9, need_sd=4.0):
    """G15/G15a: the plan question is well posed and won only by a discriminating plan.

    (a) *World-set validity.*  Every world the plan is graded in must itself be consistent with the
        notebook: it is unfair to require a conclusion in a world the disclosed evidence rules out.
    (b) *Oracle margin.*  The oracle plan's statistic must sit at least `need_sd` standard deviations
        (measured over fresh salts) away from every threshold its own rule uses, in every world.  Without
        this a correct plan fails by luck - the empirical finding from the verifier unit tests.
    (c) *Naive plan fails.*  A plan a competent-but-hasty analyst would write (the blueprint's
        NAIVE_PLANS entry, or a rival that fails the item with margin) must get at least one world wrong.
    """
    from . import verify as V
    gctx = cal["gctx"]; be = gctx["backend"]
    naive = getattr(w["bp"], "NAIVE_PLANS", {}) or {}
    out = []; ok_all = True
    for it in _items_of(cal["items"], "prereg"):
        rec = {"item": it["id"]}
        inconsistent = []
        for wd in it["worlds"]:
            okc, rep = V.consistency(be, be.full(wd["params"]), w["rows"])
            if not okc:
                inconsistent.append({"world": wd["name"], "why": rep.get("reason")})
        rec["worlds_inconsistent_with_notebook"] = inconsistent
        plan = (cal["ship_ans"] or {}).get(it["id"])
        margins = []; stats = {}
        if isinstance(plan, dict) and "rule" in plan:
            cuts = [c for c in plan["rule"].get("cuts", []) if c and c[0] != "else"]
            thr = [float(c[1]) for c in cuts]
            for wd in it["worlds"]:
                vals = []
                for s in range(n_salt):
                    salt = "%s/gate%d" % (w["salt"], s)
                    try:
                        env, _, _ = V.run_plan(gctx["session_for"](wd, salt), plan)
                        vals.append(V.eval_expr(str(plan["rule"]["expr"]), env))
                    except Exception as e:
                        rec["error"] = "plan could not be executed in %s: %s" % (wd["name"], e)
                        break
                if len(vals) < n_salt:
                    break
                mu = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                stats[wd["name"]] = {"mean": round(mu, 6), "sd": round(sd, 6), "label": wd["label"]}
                if thr:
                    d = min(abs(mu - t) for t in thr)
                    margins.append(d / sd if sd > 0 else math.inf)
        else:
            rec["error"] = "the oracle did not answer this item with a plan"
        rec["stats"] = stats
        rec["min_margin_sd"] = None if not margins else (None if math.isinf(min(margins)) else round(min(margins), 2))
        nv = naive.get(it["id"])
        if callable(nv):
            # A naive plan is usually world-dependent (it names runs, and the runs name knob settings the
            # world fixes), so a blueprint may register a callable of the drawn params instead of a literal.
            nv = nv(w["p"])
        if nv is not None:
            r = V.grade_prereg(dict(it, _cost_fn=gctx["cost_fn"]), nv, it["worlds"], gctx["session_for"])
            rec["naive_plan_fails"] = not r["pass"]
        else:
            rec["naive_plan_fails"] = any(it["id"] in v["margin_items"] for v in mat.values())
        rec["pass"] = bool(not inconsistent and "error" not in rec and rec["naive_plan_fails"]
                           and margins and min(margins) >= need_sd)
        ok_all = ok_all and rec["pass"]
        out.append(rec)
    return {"pass": bool(ok_all), "need_sd": need_sd, "n_salt": n_salt, "items": out}


def audit_gate(cal, herring_max=0.25, defect_min=2.0):
    """G16: the audited script's defect matters and its red herrings provably do not.

    Each audit item carries `variants`: the number the analysis produces as shipped, with the defect
    fixed, and with each red herring "fixed" instead.  Fixing the defect must move the number by at
    least `defect_min` accepted widths; fixing a red herring by less than `herring_max` of one, which is
    what makes it *provably* harmless rather than merely believed to be.  The number the team reported
    must also lie outside the accepted answer range, or "the reported number is fine" would pass.
    """
    out = []; ok_all = True
    for it in _items_of(cal["items"], "audit"):
        v = it.get("variants") or {}
        wd = float(it["max_width"])
        rec = {"item": it["id"], "have_variants": sorted(v)}
        if "as_shipped" not in v or "fixed" not in v or wd <= 0:
            rec["pass"] = False; rec["why"] = "item does not ship as_shipped/fixed variants and a width"
        else:
            shift = abs(float(v["fixed"]) - float(v["as_shipped"])) / wd
            herrings = {k: abs(float(x) - float(v["as_shipped"])) / wd for k, x in v.items()
                        if k.startswith("herring:")}
            loud = [k for k, x in herrings.items() if x > herring_max]
            reported_wrong = not (it["lo"] <= float(v["as_shipped"]) <= it["hi"])
            rec.update({"defect_shift_over_width": round(shift, 2),
                        "herring_shift_over_width": {k: round(x, 3) for k, x in herrings.items()},
                        "reported_number_outside_key": reported_wrong,
                        "pass": bool(shift >= defect_min and not loud and reported_wrong)})
            if loud:
                rec["not_harmless"] = loud
        ok_all = ok_all and rec["pass"]
        out.append(rec)
    return {"pass": bool(ok_all), "herring_max": herring_max, "defect_min": defect_min, "items": out}


def difficulty_record(w, cal, g):
    """What difficulty.json ships: the claimed knob settings next to the measured statistics that back
    them, so a matched pair can be checked to differ in exactly one knob."""
    bp = w["bp"]
    d = {"blueprint": bp.ID, "lab": w["spec"].get("lab", "pretrain"), "world_seed": w["ws"],
         "claimed": getattr(bp, "DIFFICULTY", {}) or {},
         "measured": {"depth": g["G11_depth"]["max_depth"],
                      "nuisance": len(_claim(bp, "nuisance") or []),
                      "anti_prior_items": [r["item"] for r in g["G13_anti_prior"]["items"]],
                      "anti_prior_gap_over_T": {r["item"]: r["gap_over_T"] for r in g["G13_anti_prior"]["items"]},
                      "witness_volume": {r["item"]: r.get("witness_volume") for r in g["G14_witness"]["items"]},
                      "plan_margin_sd": {r["item"]: r.get("min_margin_sd") for r in g["G15_plan"]["items"]},
                      "audit_defect_shift": {r["item"]: r.get("defect_shift_over_width") for r in g["G16_audit"]["items"]}},
         "kinds": sorted({it["kind"] for it in cal["items"]}),
         "n_items": len(cal["items"])}
    return d


# ------------------------------------------------------------------------------------------ gates
def _rand_request(spec, rng):
    req = {}
    for k, kd in spec["knobs"].items():
        if kd["type"] == "float":
            lo, hi = kd["min"], kd["max"]
            v = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if lo > 0 else float(rng.uniform(lo, hi))
            if kd.get("int"):
                # an integer knob rejects fractional values, so a fractional draw would make every
                # certificate request a LabError and G4 would pass on error strings alone
                v = float(min(math.floor(hi), max(math.ceil(lo), round(v))))
            req[k] = v
        else:
            req[k] = kd["values"][int(rng.integers(len(kd["values"])))]
    req["seed"] = int(rng.integers(0, 1000))
    return req


def identifiability_certificate(w, n=1500):
    """For each documented known-unknown: every runnable request (random dense sample + every notebook
    config) returns byte-identical results across the unknown's range.  Zero footprint, not 'small'.

    "Runnable" is meant literally: a request above the per-request cost cap is one the agent cannot make,
    so it is recorded as the refusal it would receive rather than executed.  `CERT_N` lets an expensive
    lab (one where a single request can be hundreds of thousands of model calls) trade sample size for
    build time; the property being certified is structural (nothing reads the parameter), and the sample
    is a check on that, not the argument for it."""
    bp = w["bp"]; out = []
    n = int(getattr(bp, "CERT_N", n))
    rng = np.random.default_rng(sid(bp.ID, w["ws"], "cert"))
    reqs = [_rand_request(w["spec"], rng) for _ in range(n)]
    for r in w["rows"]:
        rq = {k: r["config"][k] for k in w["spec"]["knobs"] if k in r["config"]}
        rq["seed"] = r["config"].get("seed", 0); reqs.append(rq)
    extra = getattr(bp, "cert_requests", None)
    if extra:
        reqs += extra(w["spec"], rng)
    cap = Session(w["pf"], w["spec"], w["salt"]).caps["run_cost"]
    for ku in bp.known_unknowns(w["p"]):
        lo, hi = ku["range"]; vals = [lo, (lo + hi) / 2, hi]
        results = []
        for v in vals:
            pv = set_path(json.loads(json.dumps(w["pf"])), ku["param"], v)
            s = Session(pv, w["spec"], w["salt"]); res = []
            for rq in reqs:
                try:
                    cfg, seed, ex = s.validate(rq)
                    c = s.cost(cfg, ex)
                    res.append("above cap: %.6g" % c if c > cap * (1 + 1e-9)
                               else json.dumps(s.execute(cfg, seed, ex), sort_keys=True))
                except LabError as e:
                    res.append("LabError:" + str(e))
            results.append(res)
        n_diff = sum(1 for i in range(len(reqs)) if len({results[j][i] for j in range(len(vals))}) > 1)
        out.append({"param": ku["param"], "range": [lo, hi], "n_requests": len(reqs), "n_differing": n_diff,
                    "zero_footprint": n_diff == 0})
    return out


def replay_gate(w):
    """Rebuild the session from the JSON that will be shipped to the server; re-execute every notebook row."""
    world_json = json.loads(json.dumps({"params": w["p"], "spec": w["spec"], "salt": w["salt"]}))
    s = Session(world_json["params"], world_json["spec"], world_json["salt"])
    bad = 0
    for r in w["rows"]:
        rq = {k: r["config"][k] for k in w["spec"]["knobs"] if k in r["config"]}
        rq["seed"] = r["config"].get("seed", 0)
        for k in ("ckpts", "cooldowns"):
            if k in r["config"]:
                rq[k] = r["config"][k]
        cfg, seed, ex = s.validate(rq)
        again = s.execute(cfg, seed, ex)
        if json.dumps(again, sort_keys=True) != json.dumps(r, sort_keys=True):
            bad += 1
    return {"n_rows": len(w["rows"]), "n_mismatch": bad, "pass": bad == 0}


def _num_patterns(v):
    pats = set()
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) or v == 0:
        return pats
    for f in ("%.3g", "%.4g", "%.5g", "%.3f", "%.4f"):
        s = f % v
        digits = re.sub(r"[^0-9]", "", s.split("e")[0]).lstrip("0")
        if len(digits.rstrip("0")) >= 3:
            pats.add(s)
    return pats


def walk_params(p, prefix=""):
    """Every scalar in a world, with a dotted name.  The v4 labs nest parameters (per-model tables,
    per-draft tables, per-slice lists), and a leakage scan that only looked at top-level scalars would
    miss exactly the interesting ones."""
    out = []
    if isinstance(p, dict):
        for k, v in p.items():
            out += walk_params(v, "%s.%s" % (prefix, k) if prefix else str(k))
    elif isinstance(p, (list, tuple)):
        for i, v in enumerate(p):
            out += walk_params(v, "%s[%d]" % (prefix, i))
    elif isinstance(p, bool):
        pass
    elif isinstance(p, (int, float)):
        out.append((prefix, float(p)))
    return out


def set_path(p, path, v):
    """Write a parameter named either "k", "a.b" or ["a", "b"]."""
    parts = path if isinstance(path, (list, tuple)) else str(path).split(".")
    node = p
    for s in parts[:-1]:
        node = node[s]
    node[parts[-1]] = v
    return p


def leakage_scan(w, items, shipped_text, public_values=()):
    """Hidden parameter values and numeric keys must not appear in anything the agent can read."""
    pub = set()
    for v in public_values:
        pub |= _num_patterns(v)
    hits = []
    cands = [("param:" + k, v) for k, v in walk_params(w["p"])]
    cands += [("key:" + it["id"] + ":" + e, it["key"][e]) for it in items if it["kind"] in ("point", "set") for e in ("lo", "hi")]
    cands += [("key:" + it["id"] + ":" + e, it[e]) for it in items if it["kind"] == "audit" for e in ("lo", "hi")]
    for name, v in cands:
        for pat in _num_patterns(v) - pub:
            if re.search(r"(?<![0-9.])" + re.escape(pat) + r"(?![0-9])", shipped_text):
                hits.append({"what": name, "pattern": pat})
    return {"hits": hits, "pass": not hits}


def oracle_bias_gate(cal, ratio_max=0.90, share_max=0.05, n_min=6):
    """Fail when the reference solution's own error is a *bias* rather than *dispersion*.

    `calibrate` sets each numeric tolerance to `max(floor, 2.25 * p90(oracle error))`, which is the right
    rule for a noisy estimator and the wrong rule for a deterministic offset: an offset inflates the
    tolerance by 2.25x and then parks the oracle at 0.44 of it, so 44% of the band is spent on the
    author's own rounding and the item can no longer resolve anything finer than that.  e05's q5 was
    exactly this - the matched budget solves to 2162.6907 samples while the answer is a count, so every
    repetition of the oracle returned 2163 and missed by 0.309 every single time.  Four frontier runs then
    "missed" by the identical 0.44 T, which looks like a difficulty signal and is not one.

    The statistic that separates the two cases is median/p90 of the oracle's error across the calibration
    repetitions.  A deterministic offset has no spread, so the ratio is 1.000; a real estimator measured
    over 10 repetitions came in at 0.24-0.71 across all 15 v4 instances.  0.90 sits in the gap with room
    on both sides.  Items whose oracle is exact (p90 == 0) are skipped, not failed.

    A second and a third threshold are needed because *some* deterministic residual is unavoidable whenever
    the oracle searches a grid or bisects a fixed number of times.  t02's edge finder leaves one: a
    divergence edge is a hard threshold, so the answer is bounded by the final bracket, and the 40-request
    cap affords only 6 bisections - about 0.005 in log10(lr), identical every repetition.  Nobody can do
    better inside the budget, so that is physics rather than carelessness.  What separates it from e05's q5
    is that t02's tolerance is the author's declared `floor` and merely contains the residual, whereas q5's
    band was *set by* the residual through the 2.25*p90 rule.  So the gate fires only when the offset spends
    more than `share_max` of the band **and** the band came out above the declared floor - i.e. when the
    bias bought itself the tolerance that then hides it.
    """
    bad, seen = {}, {}
    for it in cal["items"]:
        if it["kind"] not in ("point", "set"):
            continue
        e = sorted(x for x in cal["cal_errs"].get(it["id"], []) if math.isfinite(x))
        if len(e) < n_min or e[int(0.9 * (len(e) - 1))] <= 0.0:
            continue                                    # too few reps to judge, or an exact oracle
        med, p90 = float(np.median(e)), e[int(0.9 * (len(e) - 1))]
        tol, floor = it.get("tol") or 0.0, it.get("floor") or 0.0
        share = med / tol if tol else 0.0
        bought = tol > floor * (1.0 + 1e-9)             # the p90 rule, not the declared floor, set this band
        seen[it["id"]] = {"median": med, "p90": p90, "ratio": round(med / p90, 3),
                          "median_over_tol": round(share, 3), "tol_above_floor": bought}
        if med / p90 > ratio_max and share > share_max and bought:
            bad[it["id"]] = seen[it["id"]]
    return {"pass": not bad, "biased_items": bad, "median_over_p90": seen,
            "ratio_max": ratio_max, "share_max": share_max}


def gates(w, cal, mat, cert, replay, leak):
    bp = w["bp"]; items = cal["items"]; g = {}
    if hasattr(bp, "wellposed"):                     # blueprint-declared posedness checks (e.g. verdict bands)
        ok, why = bp.wellposed(w)
        g["G0_wellposed"] = {"pass": bool(ok), "why": why}
    g["G1_solvable"] = {"pass": cal["ver_allpass"] >= 0.8 and min(cal["ver_pass"].values()) >= 0.9 and cal["ship_grade"]["all_pass"],
                        "ver_allpass": cal["ver_allpass"], "min_item_pass": min(cal["ver_pass"].values()),
                        "shipped_oracle_all_pass": cal["ship_grade"]["all_pass"]}
    unkilled = [k for k, v in mat.items() if v["must_kill"] and not v["killed_with_margin"]]
    g["G2_kill_matrix"] = {"pass": not unkilled, "unkilled_must_kill": unkilled}
    narrow = [it["id"] for it in items if it["kind"] == "set" and it["key"]["hi"] - it["key"]["lo"] < 4 * it["tol"]]
    g["G3_set_width"] = {"pass": not narrow, "narrow": narrow}
    g["G4_identifiability"] = {"pass": all(c["zero_footprint"] for c in cert), "certs": cert}
    g["G5_replay"] = replay
    g["G6_leakage"] = leak
    kinds = {it["kind"] for it in items}
    # An "ignorance item" is one whose correct answer is a statement about what cannot be known: a set
    # interval, an `undetermined` verdict, a counterexample question (which asks whether the evidence
    # forces the claim at all), or a plan whose allowed conclusions include `undetermined`.
    ign = (any(it["kind"] == "set" for it in items)
           or any(it["kind"] == "verdict" and it["key"]["verdict"] == "undetermined" for it in items)
           or any(it["kind"] == "cex" for it in items)
           or any(it["kind"] == "prereg" and "undetermined" in it["key"]["labels"] for it in items))
    g["G7_key_types"] = {"pass": len(kinds) >= 3 and ign, "kinds": sorted(kinds), "has_ignorance_item": ign}
    dec = [(it["id"], it["gap"]) for it in items if it["kind"] == "decision"]
    g["G8_decision_gap"] = {"pass": all(gp >= 0.004 for _, gp in dec), "gaps": dec}
    lb = {}
    for c in bp.CARDS:
        if c in getattr(bp, "EXEMPT_LOAD_BEARING", {}):
            lb[c] = "exempt: " + bp.EXEMPT_LOAD_BEARING[c]
        else:
            ds = [k for k in mat if k.startswith("drop:" + c + ":") or k == "drop:" + c]
            lb[c] = "killed by " + ",".join(ds) if ds and all(mat[k]["killed_with_margin"] for k in ds) else "NOT load-bearing"
    g["G9_load_bearing"] = {"pass": all(not v.startswith("NOT") for v in lb.values()), "cards": lb}
    item_kills = {it["id"]: [k for k, v in mat.items() if it["id"] in v["margin_items"]] for it in items}
    g["G10_item_useful"] = {"pass": True, "note": "informational: which rivals each item kills with margin",
                            "free_items": [q for q, ks in item_kills.items() if not ks], "item_kills": item_kills}
    g["G11_depth"] = depth_gate(w, cal, mat)
    g["G12_nuisance"] = nuisance_gate(w, mat)
    g["G13_anti_prior"] = anti_prior_gate(w, cal, mat)
    g["G14_witness"] = witness_gate(w, cal)
    g["G15_plan"] = plan_gate(w, cal, mat)
    g["G16_audit"] = audit_gate(cal)
    g["G17_oracle_bias"] = oracle_bias_gate(cal)
    return g, all(v["pass"] for v in g.values())


def build_instance(bp_name, ws, n_cal=10, n_ver=10, shipped_text_fn=None):
    bp = load(bp_name)
    w = make_world(bp, ws)
    cal = calibrate(w, n_cal, n_ver)
    riv = rival_answers(w, cal)
    mat = kill_matrix(w, cal, riv)
    cert = identifiability_certificate(w)
    rep = replay_gate(w)
    text = shipped_text_fn(w, cal["items"]) if shipped_text_fn else (w["notes"] + json.dumps(w["rows"]))
    pubv = getattr(bp, "public_values", lambda ctx: [])(w["ctx"])
    leak = leakage_scan(w, cal["items"], text, pubv)
    g, ok = gates(w, cal, mat, cert, rep, leak)
    return dict(w=w, cal=cal, rivals=riv, mat=mat, gates=g, ok=ok,
                difficulty=difficulty_record(w, cal, g))


def key_str(it):
    k = it["key"]
    if it["kind"] in ("point", "set"):
        return "[%.4g, %.4g]" % (k["lo"], k["hi"])
    if it["kind"] == "prereg":
        return "labels=" + ",".join(k["labels"])
    if it["kind"] == "audit":
        return "%s [%.4g, %.4g]" % (k["defect"], k["lo"], k["hi"])
    return str(k.get("verdict") or k.get("choice"))


def summary(inst):
    w, cal = inst["w"], inst["cal"]
    lines = ["%s ws=%s  ok=%s" % (w["bp"].ID, w["ws"], inst["ok"])]
    for it in cal["items"]:
        lines.append("  %-4s %-8s key=%-22s T=%-8s ver=%.2f kills=%s" % (
            it["id"], it["kind"], key_str(it), ("%.3g" % it["tol"]) if "tol" in it else ("%.3g" % it.get("r_tol", 0)),
            cal["ver_pass"][it["id"]], ",".join(inst["gates"]["G10_item_useful"]["item_kills"][it["id"]])))
    for name, v in inst["mat"].items():
        lines.append("  rival %-28s score=%.2f margin=%s%s" % (name, v["score"], ",".join(v["margin_items"]) or "-",
                                                              "" if v["must_kill"] else "  (info)"))
    for gname, gv in inst["gates"].items():
        if not gv["pass"]:
            lines.append("  FAIL %s %s" % (gname, {k: v for k, v in gv.items() if k != "certs"}))
    return "\n".join(lines)
