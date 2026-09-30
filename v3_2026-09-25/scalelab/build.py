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
    p = bp.draw(rng); pf = W.full(p); sp = bp.spec(p)
    salt = "%016x" % sid(wid, ws, "salt")
    sess = Session(pf, sp, salt)
    rows, notes, ctx = bp.notebook(pf, sess, rng)
    return dict(bp=bp, ws=ws, p=p, pf=pf, spec=sp, salt=salt, rows=rows, notes=notes, ctx=ctx)


def item_error(it, ans):
    """Numeric: max endpoint error.  Categorical: 0 if right else inf."""
    if ans is None:
        return math.inf
    if it["kind"] in ("point", "set"):
        try:
            return max(abs(float(ans["lo"]) - it["key"]["lo"]), abs(float(ans["hi"]) - it["key"]["hi"]))
        except Exception:
            return math.inf
    ok, _ = Q.grade_item(it, ans)
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
    errs = {it["id"]: [] for it in base}
    for r in range(n_cal):
        ans, _, _ = oracle_rep(w, "cal%d" % r)
        for it in base:
            errs[it["id"]].append(item_error(it, ans.get(it["id"])))
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
    ver = {it["id"]: [] for it in items}; allpass = []
    for r in range(n_ver):
        ans, _, _ = oracle_rep(w, "ver%d" % r)
        g = Q.grade(items, ans)
        for q, v in g["items"].items():
            ver[q].append(v["pass"])
        allpass.append(g["all_pass"])
    ship_ans, ship_ph, ship_sess = oracle_rep(w, "")          # the agent's own salt
    ship_grade = Q.grade(items, ship_ans)
    cal_acc = {q: float(np.mean([math.isfinite(e) and e <= (tol.get(q, 0) if q in tol else 0) for e in errs[q]]))
               for q in errs}
    return dict(items=items, tol=tol, cal_errs=errs, cal_acc=cal_acc,
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
    mat = {}
    for name, ans in rivals.items():
        g = Q.grade(items, ans); per = {}; margin = []
        for it in items:
            e = item_error(it, ans.get(it["id"]))
            if it["kind"] in ("point", "set"):
                ratio = e / it["tol"] if it["tol"] > 0 else math.inf
                per[it["id"]] = {"pass": g["items"][it["id"]]["pass"], "err_over_T": round(ratio, 2) if math.isfinite(ratio) else None}
                if ratio >= 2.0:
                    margin.append(it["id"])
            else:
                per[it["id"]] = {"pass": g["items"][it["id"]]["pass"],
                                 "answer": list(ans.get(it["id"], {"?": None}).values())[0]}
                if not g["items"][it["id"]]["pass"] and stable[it["id"]]:
                    margin.append(it["id"])
        mat[name] = {"score": g["score"], "killed": not g["all_pass"], "margin_items": margin,
                     "killed_with_margin": bool(margin), "must_kill": name not in info, "items": per}
    return mat


# ------------------------------------------------------------------------------------------ gates
def _rand_request(spec, rng):
    req = {}
    for k, kd in spec["knobs"].items():
        if kd["type"] == "float":
            lo, hi = kd["min"], kd["max"]
            req[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if lo > 0 else float(rng.uniform(lo, hi))
        else:
            req[k] = kd["values"][int(rng.integers(len(kd["values"])))]
    req["seed"] = int(rng.integers(0, 1000))
    return req


def identifiability_certificate(w, n=1500):
    """For each documented known-unknown: every runnable request (random dense sample + every notebook
    config) returns byte-identical results across the unknown's range.  Zero footprint, not 'small'."""
    bp = w["bp"]; out = []
    rng = np.random.default_rng(sid(bp.ID, w["ws"], "cert"))
    reqs = [_rand_request(w["spec"], rng) for _ in range(n)]
    for r in w["rows"]:
        rq = {k: r["config"][k] for k in w["spec"]["knobs"] if k in r["config"]}
        rq["seed"] = r["config"].get("seed", 0); reqs.append(rq)
    extra = getattr(bp, "cert_requests", None)
    if extra:
        reqs += extra(w["spec"], rng)
    for ku in bp.known_unknowns(w["p"]):
        lo, hi = ku["range"]; vals = [lo, (lo + hi) / 2, hi]
        results = []
        for v in vals:
            pv = dict(w["pf"]); pv[ku["param"]] = v
            s = Session(pv, w["spec"], w["salt"]); res = []
            for rq in reqs:
                try:
                    cfg, seed, ex = s.validate(rq)
                    res.append(json.dumps(s.execute(cfg, seed, ex), sort_keys=True))
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


def leakage_scan(w, items, shipped_text, public_values=()):
    """Hidden parameter values and numeric keys must not appear in anything the agent can read."""
    pub = set()
    for v in public_values:
        pub |= _num_patterns(v)
    hits = []
    cands = [("param:" + k, v) for k, v in w["p"].items()]
    cands += [("key:" + it["id"] + ":" + e, it["key"][e]) for it in items if it["kind"] in ("point", "set") for e in ("lo", "hi")]
    for name, v in cands:
        for pat in _num_patterns(v) - pub:
            if re.search(r"(?<![0-9.])" + re.escape(pat) + r"(?![0-9])", shipped_text):
                hits.append({"what": name, "pattern": pat})
    return {"hits": hits, "pass": not hits}


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
    ign = any(it["kind"] == "set" for it in items) or any(it["kind"] == "verdict" and it["key"]["verdict"] == "undetermined" for it in items)
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
    return dict(w=w, cal=cal, rivals=riv, mat=mat, gates=g, ok=ok)


def summary(inst):
    w, cal = inst["w"], inst["cal"]
    lines = ["%s ws=%s  ok=%s" % (w["bp"].ID, w["ws"], inst["ok"])]
    for it in cal["items"]:
        k = it["key"]
        ks = ("[%.4g, %.4g]" % (k["lo"], k["hi"])) if it["kind"] in ("point", "set") else (k.get("verdict") or k.get("choice"))
        lines.append("  %-4s %-8s key=%-22s T=%-8s ver=%.2f kills=%s" % (
            it["id"], it["kind"], ks, ("%.3g" % it["tol"]) if "tol" in it else ("%.3g" % it.get("r_tol", 0)),
            cal["ver_pass"][it["id"]], ",".join(inst["gates"]["G10_item_useful"]["item_kills"][it["id"]])))
    for name, v in inst["mat"].items():
        lines.append("  rival %-28s score=%.2f margin=%s%s" % (name, v["score"], ",".join(v["margin_items"]) or "-",
                                                              "" if v["must_kill"] else "  (info)"))
    for gname, gv in inst["gates"].items():
        if not gv["pass"]:
            lines.append("  FAIL %s %s" % (gname, {k: v for k, v in gv.items() if k != "certs"}))
    return "\n".join(lines)
