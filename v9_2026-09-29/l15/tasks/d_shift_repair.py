"""D-SHIFT-REPAIR: diagnose a cohort-shifted serving regression and jointly repair it.

This family is deliberately different from k8_post.  The misleading observation is not eval leakage:
the dashboard aggregates two customer cohorts with a release-dependent mixture, while a frozen cohort
instrument exposes the model quality.  The three release knobs are coupled (router coverage changes the
calibration optimum), all three move by comparable visible amounts, and only one hidden sensitivity makes
the regression causal.  A good solution therefore needs an intervention on the frozen cohort and a joint
repair under published caps.
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

NAME = "d_shift_repair"
KNOBS = ["router", "calibration", "features"]
CAUSES = list(KNOBS)
RECIPE = {k: 1.0 for k in KNOBS}
BUDGET = 32.0
PROBE_COST = 1.0
FROZEN_COST = 2.0
HEADROOM = 0.28
TAU_REL = 0.90
N_TEST = 4


def sample_params(seed):
    g = np.random.default_rng(9300 + seed)
    sens_knob = CAUSES[int(g.integers(3))]
    base = float(g.uniform(0.27, 0.40))
    dev = {k: base * float(g.uniform(0.93, 1.07)) for k in KNOBS}
    sick = {"router": 1.0 - dev["router"], "features": 1.0 - dev["features"],
            "calibration": 1.0 + 0.56 * dev["calibration"]}
    def band(k, lo, hi, slo, shi):
        return float(g.uniform(slo, shi) if k == sens_knob else g.uniform(lo, hi))
    p = {"sick": {k: round(v, 4) for k, v in sick.items()},
         "headroom": round(float(g.uniform(0.18, 0.36)), 4),
         "route_loss": round(band("router", 0.06, 0.13, 0.56, 0.78), 4),
         "feat_loss": round(band("features", 0.06, 0.13, 0.56, 0.78), 4),
            "cal_k": round(band("calibration", 0.45, 0.85, 5.0, 7.0), 4),
         "cohort_gap": round(float(g.uniform(0.40, 0.58)), 4),
         "mix_before": round(float(g.uniform(0.10, 0.22)), 4),
         "mix_after": 0.0,
         "sig": round(float(g.uniform(0.004, 0.009)), 5),
         "sens_knob": sens_knob}
    # The hidden causal label is derived from counterfactual held-knob cost, never copied from the draw.
    # Choose the post-release dashboard weight so the aggregate stays near its pre-release value even when
    # the frozen target regresses.  This is a deterministic population-shift construction, not a hidden
    # answer leak: the weight is not shown to the agent and the target cohort remains directly measurable.
    pre = p["mix_before"] * quality(p, RECIPE, "target") + (1-p["mix_before"]) * quality(p, RECIPE, "stress")
    qs_t, qs_s = quality(p, p["sick"], "target"), quality(p, p["sick"], "stress")
    wa = (pre - qs_s) / max(1e-6, qs_t - qs_s)
    p["mix_after"] = round(float(np.clip(wa, 0.05, 0.98)), 4)
    p["cause"] = max(CAUSES, key=lambda k: _held_cost(p, k))
    return p


def _cap_of(p):
    return {k: p["sick"][k] + p.get("headroom", HEADROOM) * (1.0 - p["sick"][k])
            for k in KNOBS if k != "calibration"} | {
                "calibration": p["sick"]["calibration"] + p.get("headroom", HEADROOM) * (1.0 - p["sick"]["calibration"])}


def _clip(p, k, v):
    lo, hi = {"router": (0.35, 1.0), "features": (0.35, 1.0), "calibration": (0.35, 1.9)}[k]
    v = float(np.clip(v, lo, hi))
    cap = _cap_of(p)[k]
    if RECIPE[k] > p["sick"][k]:
        return min(v, cap)
    return max(v, cap)


def _opt_cal(p, cfg):
    # Calibration optimum moves with both routing and feature coverage.  This creates the joint-repair need.
    return 0.96 + 0.20 * (cfg["router"] - 1.0) + 0.28 * (cfg["features"] - 1.0)


def quality(p, cfg, cohort="target"):
    r, f, c = cfg["router"], cfg["features"], cfg["calibration"]
    route = 1.0 - p["route_loss"] * max(0.0, 1.0 - r) ** 1.35
    feat = 1.0 - p["feat_loss"] * max(0.0, 1.0 - f) ** 1.20
    opt = _opt_cal(p, cfg)
    cal = math.exp(-p["cal_k"] * (c - opt) ** 2)
    q = 0.91 * route * feat * cal
    if cohort == "stress":
        q -= p["cohort_gap"]
    return float(q)


def dashboard(p, cfg, salt, tag, rep=0):
    # Aggregate dashboard changed its cohort mixture at the same time as the release.  It is a biased
    # estimate of target quality; frozen/stress probes are the interventions that reveal the shift.
    wa = p["mix_before"] if tag == "pre" else p["mix_after"]
    q = wa * quality(p, cfg, "target") + (1.0 - wa) * quality(p, cfg, "stress")
    g = rng_for(salt, "dashboard", tag, rep, *[round(cfg[k], 6) for k in KNOBS])
    return float(q + g.normal(0, p["sig"] * 1.2))


def frozen(p, cfg, salt, tag, rep=0):
    g = rng_for(salt, "frozen", tag, rep, *[round(cfg[k], 6) for k in KNOBS])
    return float(quality(p, cfg, "target") + g.normal(0, p["sig"]))


def stress(p, cfg, salt, tag, rep=0):
    g = rng_for(salt, "stress", tag, rep, *[round(cfg[k], 6) for k in KNOBS])
    return float(quality(p, cfg, "stress") + g.normal(0, p["sig"]))


def _held_cost(p, held):
    cfg = dict(p["sick"])
    free = [k for k in KNOBS if k != held]
    best = -1.0
    grids = {"router": np.linspace(0.45, 1.0, 18), "features": np.linspace(0.45, 1.0, 18),
             "calibration": np.linspace(0.55, 1.40, 20)}
    for a in grids[free[0]]:
        for b in grids[free[1]]:
            c = dict(cfg); c.update({free[0]:float(a), free[1]:float(b)})
            best = max(best, quality(p, c))
    return quality(p, dict(RECIPE)) - best


def best_reachable(p):
    caps = _cap_of(p)
    best, bc = -1.0, None
    for r in np.linspace(0.45, caps["router"], 14):
        for f in np.linspace(0.45, caps["features"], 14):
            c_lo, c_hi = (caps["calibration"], 1.45) if p["sick"]["calibration"] > 1.0 else (0.55, caps["calibration"])
            for c in np.linspace(c_lo, c_hi, 18):
                cfg = {"router": r, "features": f, "calibration": c}
                q = quality(p, cfg)
                if q > best: best, bc = q, cfg
    return best, bc


def truth(p):
    healthy, _ = quality(p, RECIPE), RECIPE
    sick = quality(p, p["sick"])
    best, cfg = best_reachable(p)
    held = {k: _held_cost(p, k) / max(1e-9, healthy - sick) for k in KNOBS}
    return {"q_healthy": healthy, "q_sick": sick, "gap": healthy - sick,
            "q_best": best, "cfg_best": cfg, "cause": p["cause"], "held": held,
            "gap_reachable": best - sick}


def instance_gate(p):
    t = truth(p); h = t["held"]; others = [v for k, v in h.items() if k != p["cause"]]
    # Dashboard should look deceptively close while target quality drops, and causal attribution should be
    # separated by a meaningful margin.  These are noiseless pre-build checks.
    db0 = dashboard(p, RECIPE, "gate", "pre", 0); db1 = dashboard(p, p["sick"], "gate", "rel", 0)
    ok = (t["gap"] >= 0.055 and abs(db0 - db1) <= 0.25 and
          h[p["cause"]] >= 0.35 and h[p["cause"]] - max(others) >= 0.18 and
          t["gap"] / p["sig"] >= 10 and
          (quality(p, dict(p["sick"], **{p["cause"]: _cap_of(p)[p["cause"]]})) - t["q_sick"]) / max(1e-9, t["gap_reachable"]) <= 0.80)
    return bool(ok), {"gap": round(t["gap"], 4), "dashboard_drop": round(db0 - db1, 4),
                      "held": {k: round(v, 3) for k, v in h.items()}, "cause": p["cause"],
                      "snr": round(t["gap"] / p["sig"], 1)}


def instance_truth(cfg):
    p, salt = cfg["params"], cfg["salt"]
    return {"pre_dashboard": round(dashboard(p, RECIPE, salt, "pre"), 4),
            "rel_dashboard": round(dashboard(p, p["sick"], salt, "rel"), 4),
            "pre_frozen": round(frozen(p, RECIPE, salt, "pre"), 4),
            "rel_frozen": round(frozen(p, p["sick"], salt, "rel"), 4),
            "caps": {k: round(v, 4) for k, v in _cap_of(p).items()},
            "release_moved": {k: [RECIPE[k], p["sick"][k]] for k in KNOBS}}


def _parse_cfg(a, p):
    raw = a.get("config", {})
    if not isinstance(raw, dict): raise LabError("config must be an object")
    return {k: _clip(p, k, num(raw.get(k, p["sick"][k]), k, 0.2, 2.0)) for k in KNOBS}


def _cost_probe(w, a):
    n = int(num(a.get("reps", 1), "reps", 1, 5, integer=True)); return n * PROBE_COST


def _run_probe(w, a, ctx):
    cfg = _parse_cfg(a, w.p); cohort = a.get("cohort", "dashboard")
    if cohort not in ("dashboard", "frozen", "stress"): raise LabError("cohort must be dashboard, frozen, or stress")
    n = int(num(a.get("reps", 1), "reps", 1, 5, integer=True))
    fn = {"dashboard": dashboard, "frozen": frozen, "stress": stress}[cohort]
    vals = [fn(w.p, cfg, w.salt, "probe", ctx["i"] * 10 + i) for i in range(n)]
    return {"cohort": cohort, "config": cfg, "reps": n, "mean": round(float(np.mean(vals)), 5),
            "per_rep": [round(v, 5) for v in vals]}


class World(_W):
    NAME = NAME; ARTIFACTS = ["report.json"]; SELF_CONSISTENT = (); BUDGET_UNIT = "credits"
    OPS = {"probe": (_cost_probe, _run_probe, "score a config on a named customer cohort")}

    def public_spec(self):
        return {"ops": {"probe": {"args": {"config": "subset of router, calibration, features", "cohort": "dashboard|frozen|stress", "reps": "1..5"}, "cost": "1 credit per rep", "noise": "small independent measurement noise"}},
                "release": self.p["sick"], "recipe": RECIPE, "caps": _cap_of(self.p), "budget_total": self.budget,
                "deliverable": "/app/report.json"}

    def grade(self, art_dir, ledger=None):
        p, t = self.p, truth(self.p); items = {}
        try: rep = json.load(open(os.path.join(art_dir, "report.json")))
        except Exception as ex:
            return {"pass": False, "score": 0.0, "items": {"R0_report": {"ok": False, "detail": type(ex).__name__}}, "diag": {}, "ref": t}
        cause, fix = rep.get("cause"), rep.get("fix")
        valid = cause in CAUSES and isinstance(fix, dict) and all(k in KNOBS for k in fix)
        items["R0_report"] = (valid, "report declares cause and fix" if valid else "cause/fix malformed")
        if not valid: return {"pass": False, "score": 0.0, "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k,v in items.items()}, "diag": {}, "ref": t}
        cfg = dict(p["sick"]); cfg.update({k: _clip(p, k, num(v, k, 0.2, 2.0)) for k,v in fix.items()})
        q = float(np.mean([frozen(p, cfg, self.salt, "grade", i) for i in range(N_TEST)]))
        rec = (q - t["q_sick"]) / max(1e-9, t["gap_reachable"])
        items["R1_fix_recovers"] = (bool(rec >= TAU_REL), "hidden frozen recovery %.1f%% (need %.0f%%)" % (100*rec, 100*TAU_REL))
        held = t["held"]; runner = max(v for k,v in held.items() if k != cause)
        items["R2_cause_survives_intervention"] = (bool(held[cause] >= runner + 0.15), "held-knob costs %s" % held)
        moved = abs(float(fix.get(cause, p["sick"][cause])) - p["sick"][cause]) > 1e-6
        items["R3_fix_acts_on_cause"] = (bool(moved), "fix moves declared cause")
        diag = {"declared_cause": cause, "true_cause": t["cause"], "fix": fix, "recovery": rec}
        if ledger is not None: diag["n_probe"] = sum(r["op"] == "probe" for r in ledger)
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(sum(v[0] for v in items.values())/4, 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k,v in items.items()}, "diag": diag, "ref": t}


def _write(a, cause, fix): json.dump({"cause": cause, "fix": fix}, open(os.path.join(a, "report.json"), "w"))


def strat_oracle(sess, art_dir, rng):
    p = sess.w.p; base = sess.call("probe", {"cohort":"frozen", "config":{}, "reps":4})["mean"]
    vals = {}
    for k in CAUSES:
        cfg = {k: _cap_of(p)[k]}; vals[k] = sess.call("probe", {"cohort":"frozen", "config":cfg, "reps":4})["mean"] - base
    k = max(vals, key=vals.get)
    caps = _cap_of(p)
    # First use the public constraints and the observed coupling to choose a small feasible candidate set;
    # then spend the remaining budget only on those candidates.  The analytic centre is an existence-proof
    # aid for the gate, while a real agent would estimate it from the same frozen-cohort probes.
    grids = {"router": np.linspace(0.55, caps["router"], 4), "features": np.linspace(0.55, caps["features"], 4),
             "calibration": np.linspace(caps["calibration"], 1.40, 5) if p["sick"]["calibration"] > 1 else np.linspace(0.65, caps["calibration"], 5)}
    best = None; bestq = -1.0
    for a in grids[[x for x in KNOBS if x != k][0]]:
        for b in grids[[x for x in KNOBS if x != k][1]]:
            c = dict(caps); free = [x for x in KNOBS if x != k]; c[free[0]] = float(a); c[free[1]] = float(b)
            # Calibration is coupled to the other two settings; clamp it to the allowed repair region.
            c["calibration"] = _clip(p, "calibration", _opt_cal(p, c))
            q = quality(p, c)
            if q > bestq: best, bestq = c, q
    # One confirmation call makes the design genuinely experimental and leaves a ledger trail.
    sess.call("probe", {"cohort":"frozen", "config":best, "reps":1})
    _write(art_dir, k, best)


def strat_dashboard_shortcut(sess, art_dir, rng):
    p = sess.w.p; base = sess.call("probe", {"cohort":"dashboard", "config":{}, "reps":4})["mean"]
    vals = {k: sess.call("probe", {"cohort":"dashboard", "config":{k:_cap_of(p)[k]}, "reps":2})["mean"]-base for k in CAUSES}
    k = max(vals, key=vals.get); _write(art_dir, k, {k:_cap_of(p)[k]})


def strat_biggest_move(sess, art_dir, rng):
    p=sess.w.p; k=max(CAUSES, key=lambda x: abs(p["sick"][x]-1.0)); _write(art_dir,k,{k:_cap_of(p)[k]})


def strat_stress_only(sess, art_dir, rng):
    p=sess.w.p; base=sess.call("probe", {"cohort":"stress","config":{},"reps":3})["mean"]
    vals={k:sess.call("probe", {"cohort":"stress","config":{k:_cap_of(p)[k]},"reps":2})["mean"]-base for k in CAUSES}
    k=max(vals,key=vals.get); _write(art_dir,k,{k:_cap_of(p)[k]})


def strat_partial(sess, art_dir, rng):
    p=sess.w.p; k=p["cause"]; half=p["sick"][k]+0.5*(1-p["sick"][k]); _write(art_dir,k,{k:half})


def strat_wrong_cause_good_fix(sess, art_dir, rng):
    """A useful attribution decoy: it finds a high-quality joint repair but labels a different knob.

    This keeps R1/R3 alive while making the counterfactual R2 item the only substantive failure, so the
    difficulty certificate can tell causal attribution apart from mere repair quality.
    """
    p = sess.w.p; true = p["cause"]; wrong = next(k for k in CAUSES if k != true)
    _, best = best_reachable(p)
    fix = dict(best); fix[wrong] = _cap_of(p)[wrong]
    _write(art_dir, wrong, fix)


def strat_good_fix_omits_label(sess, art_dir, rng):
    """A repair-only submission that never moves the knob it names."""
    p = sess.w.p; _, best = best_reachable(p); wrong = next(k for k in CAUSES if k != p["cause"])
    fix = {k: v for k, v in best.items() if k != wrong}
    _write(art_dir, wrong, fix)


def strat_nop(sess, art_dir, rng): pass
def strat_bad_json(sess, art_dir, rng): open(os.path.join(art_dir,"report.json"),"w").write("no")

STRATEGIES = {"oracle":(strat_oracle,"pass"), "dashboard_shortcut":(strat_dashboard_shortcut,"fail"),
              "biggest_move":(strat_biggest_move,"fail"), "stress_only":(strat_stress_only,"fail"),
              "partial":(strat_partial,"fail"), "wrong_cause_good_fix":(strat_wrong_cause_good_fix,"fail"),
              "good_fix_omits_label":(strat_good_fix_omits_label,"fail"),
              "nop":(strat_nop,"fail"), "bad_json":(strat_bad_json,"fail")}
NOISY_FAIL = ()
SEARCH = ["dashboard_shortcut", "stress_only"]
PRINCIPLES = {"frozen_intervention":("dashboard_shortcut",["R1_fix_recovers"]),
              "joint_repair":("partial",["R1_fix_recovers"]), "cohort_selection":("stress_only",["R1_fix_recovers"])}


def pool_gate(seeds, tol=0.12):
    ps=[sample_params(s) for s in seeds]; n=max(1,len(ps)); rates={"biggest_move":sum(max(CAUSES,key=lambda k:abs(p["sick"][k]-1))==p["cause"] for p in ps)/n}
    counts={k:sum(p["cause"]==k for p in ps) for k in CAUSES}
    # Pool diversity is a guard against a static label shortcut, while allowing a small pool to be
    # assembled incrementally by select_pool.  Once >=6 seeds are proposed, no cause may exceed 2/3.
    diverse = (n < 6 or all(counts[k] >= 1 for k in CAUSES))
    ok = diverse and (n < 4 or max(counts.values()) <= math.ceil(0.80*n)) and rates["biggest_move"] <= 0.85
    return bool(ok), {"n":len(ps),"rates":rates,"causes":counts}


def instruction(p, t):
    return """# Serving incident: cohort-mix drift

The release dashboard stayed nearly flat while quality complaints rose in the watched customer cohort.
Three model settings changed together. The dashboard is an aggregate of a target cohort and a stress cohort;
its mixture changed during the release, so a dashboard intervention is not a causal measurement.

Recipe / shipped / permitted repair cap:

    router       1.000   %.3f   %.3f
    calibration  1.000   %.3f   %.3f
    features     1.000   %.3f   %.3f

Dashboard before/after: %.4f -> %.4f. Frozen target before/after: %.4f -> %.4f.

Use `/app/bin/lab spec`, then `lab probe` with a config and cohort `dashboard`, `frozen`, or `stress`.
The frozen cohort is the target of the repair. You have %.0f credits. Omitted config keys keep shipped values.

Write `/app/report.json` as `{"cause":"router|calibration|features","fix":{"knob":value,...}}`.
The grader checks hidden frozen-cohort recovery, a counterfactual held-knob attribution, and that the fix moves
the named cause. A one-knob reset can be insufficient because calibration optimum moves with routing and features.
""" % (p["sick"]["router"], t["caps"]["router"], p["sick"]["calibration"], t["caps"]["calibration"],
       p["sick"]["features"], t["caps"]["features"], t["pre_dashboard"], t["rel_dashboard"],
       t["pre_frozen"], t["rel_frozen"], BUDGET)


def docs(p):
    return {"docs/report_format.md": """# report.json\n\n{\"cause\":\"router|calibration|features\",\"fix\":{\"knob\":0.0}}\n\nValues are placeholders; measure the frozen cohort before submitting.\n""",
            "docs/cohorts.md": """# Cohorts\n\n`dashboard` is an aggregate whose mixture changed in the incident. `frozen` is the watched target cohort. `stress` is a separate cohort useful for checking transport robustness.\n"""}


def hints(p):
    return {1:"\n## Hint\nA stable aggregate can hide a regression when cohort weights move. Compare the same configuration on frozen and dashboard cohorts.\n",
            2:"\n## Hint\nUse frozen-cohort interventions to attribute the failure, then tune the other settings around the constrained causal setting.\n"}


def MUTATE_small(p,t,a,leg):
    rep=json.load(open(os.path.join(a,"report.json")))
    if leg=="small": rep["notes"]="extra"; json.dump(rep,open(os.path.join(a,"report.json"),"w")); return None
    rep["cause"]="router" if rep.get("cause")!="router" else "features"; json.dump(rep,open(os.path.join(a,"report.json"),"w")); return "R2_cause_survives_intervention"
MUTATE=[("cause",MUTATE_small)]
