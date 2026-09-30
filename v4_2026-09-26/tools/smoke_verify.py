"""Unit smoke test for scalelab.verify against a live EvalLab session.

Run:  cd v4 && python3 tools/smoke_verify.py
Checks, in order:
  1  consistency() accepts the true world on rows the session actually produced
  2  consistency() rejects a perturbed world (a moved ability), and reports which row broke
  3  consistency() rejects a world that contradicts a *structural* (sd == 0) quantity
  4  grade_cex() passes a genuine witness and fails a consistent-but-non-flipping one
  5  eval_expr / eval_rule accept the declarative DSL and reject everything else
  6  grade_prereg() discriminates a good plan from a naive one across two worlds
  7  grade_audit() scores the three parts independently
"""
import copy, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalelab import verify as V
from scalelab.lab import Session, LabError
from scalelab.labs import evallab as EL

FAIL = []


def ck(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + str(extra)) if extra else ""))
    if not cond:
        FAIL.append(name)


# --------------------------------------------------------------------------- a world
PARAMS = dict(
    n_items=600, n_slices=3, bank_seed=7,
    a_mu=1.30, a_sd=0.28, b_mu=0.10, b_sd=0.90, slice_b=[-0.35, 0.0, 0.40],
    dup_frac=0.12, dup_lam=5.0,
    fmt_ext={"mc_letter": {"base": 0.01, "by_slice": [0.01, 0.01, 0.02]},
             "free_exact": {"base": 0.09, "by_slice": [0.06, 0.09, 0.15]}},
    fmt_kappa={"mc_letter": 1.0, "free_exact": 1.0},
    models={"alpha": dict(theta=0.55, kappa=0.5, length=320.0, cost=1.0, family="A"),
            "bravo": dict(theta=0.35, kappa=0.5, length=640.0, cost=1.0, family="B")},
    judge=dict(cq=2.1, dpos=0.45, phi=0.30, psi=0.0, sigl=150.0, cost=0.5),
    sig_call=0.10,
)
SPEC = {"lab": "evallab",
        "knobs": {"svc": {"type": "choice", "values": ["score", "judge", "corpus"], "default": "score"},
                  "model": {"type": "choice", "values": ["alpha", "bravo"], "default": "alpha"},
                  "a": {"type": "choice", "values": ["alpha", "bravo"], "default": "alpha"},
                  "b": {"type": "choice", "values": ["alpha", "bravo"], "default": "bravo"},
                  "fmt": {"type": "choice", "values": list(EL.FMTS), "default": "mc_letter"},
                  "split": {"type": "choice", "values": ["all", "s0", "s1", "s2"], "default": "all"},
                  "mode": {"type": "choice", "values": ["fast", "scan"], "default": "fast"},
                  "order": {"type": "choice", "values": ["both", "ab", "ba"], "default": "both"},
                  "n": {"type": "float", "min": 1, "max": 600, "int": True, "default": 200},
                  "reps": {"type": "float", "min": 1, "max": 5, "int": True, "default": 1},
                  "detail": {"type": "float", "min": 0, "max": 1, "int": True, "default": 0}},
        "caps": {"run_cost": 4000.0, "total_cost": 60000.0, "max_runs": 60}}


def sess_for(params, salt):
    return Session(params, SPEC, salt)


print("1-3  consistency() on rows a live session produced")
s = sess_for(PARAMS, "salt-A")
rows = [s.run({"svc": "score", "model": "alpha", "fmt": "mc_letter", "n": 300, "detail": 1, "seed": 1}),
        s.run({"svc": "score", "model": "alpha", "fmt": "free_exact", "n": 300, "detail": 1, "seed": 2}),
        s.run({"svc": "score", "model": "bravo", "fmt": "mc_letter", "n": 300, "reps": 3, "detail": 1, "seed": 3}),
        s.run({"svc": "judge", "a": "alpha", "b": "bravo", "n": 150, "order": "both", "seed": 4}),
        s.run({"svc": "corpus", "mode": "fast", "n": 40, "seed": 5})]
p_true = EL.full(PARAMS)
ok, rep = V.consistency(EL, p_true, rows)
ck("true world is consistent", ok, rep)
print("       %s" % json.dumps({k: rep[k] for k in ("n_obs", "sum_z2", "chi2_bar", "max_abs_z")}))

p_bad = copy.deepcopy(PARAMS); p_bad["models"]["alpha"]["theta"] = 1.45
ok2, rep2 = V.consistency(EL, EL.full(p_bad), rows)
ck("far-off ability is rejected", not ok2, rep2.get("reason", ""))

p_str = copy.deepcopy(PARAMS)
p_str["fmt_ext"]["free_exact"] = {"base": 0.30, "by_slice": [0.30, 0.30, 0.30]}
ok3, rep3 = V.consistency(EL, EL.full(p_str), rows)
ck("structural mismatch (extraction count) is rejected", not ok3, rep3.get("reason", ""))
ck("structural rejection names the row", "row" in rep3, rep3)

print("1b   no false rejection of the true world (20 salts)")
nbad = []
for i in range(20):
    si = sess_for(PARAMS, "cal-%d" % i)
    ri = [si.run({"svc": "score", "model": "alpha", "fmt": "mc_letter", "n": 300, "detail": 1, "seed": 1}),
          si.run({"svc": "score", "model": "alpha", "fmt": "free_exact", "n": 300, "detail": 1, "seed": 2}),
          si.run({"svc": "score", "model": "bravo", "fmt": "mc_letter", "n": 300, "reps": 3, "detail": 1, "seed": 3}),
          si.run({"svc": "judge", "a": "alpha", "b": "bravo", "n": 150, "order": "both", "seed": 4}),
          si.run({"svc": "corpus", "mode": "fast", "n": 40, "seed": 5})]
    o, r = V.consistency(EL, p_true, ri)
    if not o:
        nbad.append((i, r.get("reason"), r.get("max_abs_z")))
ck("the true world is never rejected", not nbad, nbad[:3])

print("1c   witness slack: how far one ability can move and still pass")
edge = None
for d in [0.05 * k for k in range(1, 21)]:
    q = copy.deepcopy(PARAMS); q["models"]["alpha"]["theta"] = 0.55 + d
    o, _ = V.consistency(EL, EL.full(q), rows)
    if not o:
        edge = d; break
print("       alpha's ability is pinned to about +/-%s around the truth by these 5 rows" % edge)
ck("the disclosed rows do constrain the ability", edge is not None and edge <= 0.6, edge)

print("4    grade_cex")
SCHEMA = {"params": {"dpos": {"path": ["judge", "dpos"], "lo": -1.5, "hi": 1.5},
                     "phi": {"path": ["judge", "phi"], "lo": 0.0, "hi": 1.2},
                     "th_a": {"path": ["models", "alpha", "theta"], "lo": -1.0, "hi": 1.5},
                     "th_b": {"path": ["models", "bravo", "theta"], "lo": -1.0, "hi": 1.5}}}
# claim: "alpha is genuinely better than bravo" -- true in the real world, refutable because the
# observed arena margin can be explained by a length preference instead.
claim = lambda p: EL.full(p)["models"]["alpha"]["theta"] > EL.full(p)["models"]["bravo"]["theta"]
item = {"id": "q1", "kind": "cex", "claim": "alpha has higher ability than bravo",
        "schema": SCHEMA, "entailed": False}
# a witness far outside the identified set: flips the claim but cannot reproduce the score rows
r_bad = V.grade_cex(item, {"verdict": "refutable",
                           "witness": {"dpos": 0.45, "phi": 0.30, "th_a": -0.60, "th_b": 1.40}},
                    EL, PARAMS, rows, claim)
ck("witness outside the identified set fails", not r_bad["pass"], r_bad.get("reason"))
# the true world as a witness: consistent, but the claim still holds -> must fail
r_tw = V.grade_cex(item, {"verdict": "refutable",
                          "witness": {"dpos": 0.45, "phi": 0.30, "th_a": 0.55, "th_b": 0.35}},
                   EL, PARAMS, rows, claim)
ck("consistent non-flipping witness fails", not r_tw["pass"], r_tw.get("reason"))
ck("  (and it was accepted as consistent)", r_tw.get("consistency", {}).get("ok") is True,
   r_tw.get("consistency"))
r_ent = V.grade_cex(item, {"verdict": "entailed"}, EL, PARAMS, rows, claim)
ck("'entailed' fails on a refutable item", not r_ent["pass"])
r_mal = V.grade_cex(item, {"verdict": "refutable", "witness": {"dpos": 0.4}}, EL, PARAMS, rows, claim)
ck("witness missing parameters is rejected", not r_mal["pass"] and "missing" in r_mal["reason"],
   r_mal.get("reason"))
r_out = V.grade_cex(item, {"verdict": "refutable",
                           "witness": {"dpos": 9.0, "phi": 0.3, "th_a": 0.5, "th_b": 0.3}},
                    EL, PARAMS, rows, claim)
ck("out-of-range witness is rejected", not r_out["pass"] and "outside" in r_out["reason"],
   r_out.get("reason"))

print("5    the plan DSL")
env = {"r1.acc": 0.62, "r2.acc": 0.55, "r1.n_scored": 300.0, "hi": 2.0}
ck("arithmetic over run fields", abs(V.eval_expr("r1.acc - r2.acc", env) - 0.07) < 1e-12)
ck("whitelisted functions", abs(V.eval_expr("abs(r2.acc - r1.acc) * sqrt(4)", env) - 0.14) < 1e-12)
for bad in ("__import__('os').system('id')", "r1.acc.__class__", "[x for x in (1,2)]",
            "open('/etc/passwd')", "r1.acc if hi else 0", "eval('1')", "r9.acc", "lambda: 1",
            "r1.acc and 1", "globals()"):
    try:
        V.eval_expr(bad, env); ck("rejects %r" % bad, False, "it evaluated")
    except V.RuleError:
        ck("rejects %r" % bad, True)
    except Exception as e:
        ck("rejects %r" % bad, False, "wrong exception %r" % (e,))
lab, val = V.eval_rule({"expr": "r1.acc - r2.acc", "cuts": [[">", 0.03, "A"], ["<", -0.03, "B"],
                                                            ["else", "undetermined"]]}, env)
ck("rule returns the first matching cut", lab == "A" and abs(val - 0.07) < 1e-12, (lab, val))
for badrule in ({"expr": "r1.acc", "cuts": [[">", 0.1, "A"]]},
                {"expr": "r1.acc", "cuts": [["else", "A"], [">", 0.1, "B"]]},
                {"expr": "r1.acc", "cuts": [["==", 0.1, "A"], ["else", "B"]]},
                {"expr": "r1.acc"}):
    try:
        V.eval_rule(badrule, env); ck("rejects rule %r" % (badrule,), False, "it evaluated")
    except V.RuleError:
        ck("rejects rule %r" % (badrule,), True)

print("6    grade_prereg across two worlds")
# Two worlds consistent with a disclosed *pooled* arena margin: in W_ability alpha is genuinely
# better; in W_style the two are equal and the judge's margin is a length preference alone.  A plan
# that only runs the pooled arena cannot tell them apart; a plan that scores both models on the item
# bank can -- provided it buys enough repetitions to beat the between-call jitter.
W = []
for nm, th_b, phi, label in (("W_ability", 0.05, 0.30, "ability"), ("W_style", 0.55, 0.90, "style")):
    q = copy.deepcopy(PARAMS); q["models"]["bravo"]["theta"] = th_b; q["judge"]["phi"] = phi
    W.append({"name": nm, "params": q, "label": label, "salts": ["s1", "s2"]})


def factory_for(w, salt):
    return lambda: sess_for(w["params"], "%s-%s" % (w["name"], salt))


def cost_fn(req):
    q = Session(PARAMS, SPEC, "cost")
    cfg, seed, extra = q.validate(req)
    return q.cost(cfg, extra)


pitem = {"id": "q2", "kind": "prereg", "max_runs": 4, "budget": 8000.0,
         "key": {"labels": ["ability", "style"]}}
good = {"runs": [{"label": "a", "svc": "score", "model": "alpha", "fmt": "mc_letter", "n": 600,
                  "reps": 5, "seed": 11},
                 {"label": "b", "svc": "score", "model": "bravo", "fmt": "mc_letter", "n": 600,
                  "reps": 5, "seed": 12}],
        "rule": {"expr": "a.acc - b.acc", "cuts": [[">", 0.05, "ability"], ["else", "style"]]}}
naive = {"runs": [{"label": "j", "svc": "judge", "a": "alpha", "b": "bravo", "n": 300, "seed": 13}],
         "rule": {"expr": "j.win_rate_a - 0.5", "cuts": [[">", 0.02, "ability"], ["else", "style"]]}}
gi = dict(pitem); gi["_cost_fn"] = cost_fn
rg = V.grade_prereg(gi, good, W, factory_for)
ck("discriminating plan passes", rg["pass"], rg.get("reason"))
print("       good:  " + json.dumps([{k: b.get(k) for k in ("world", "got", "stat")} for b in rg["by_world"]]))
rn = V.grade_prereg(dict(gi), naive, W, factory_for)
ck("naive pooled-arena plan fails", not rn["pass"], rn.get("reason"))
print("       naive: " + json.dumps([{k: b.get(k) for k in ("world", "expected", "got", "stat")}
                                     for b in rn["by_world"]]))
under = {"runs": [{"label": "a", "svc": "score", "model": "alpha", "n": 40, "seed": 11},
                  {"label": "b", "svc": "score", "model": "bravo", "n": 40, "seed": 12}],
         "rule": good["rule"]}
ru = V.grade_prereg(dict(gi), under, W, factory_for)
print("       under-powered (n=40): pass=%s  %s" % (ru["pass"], [b.get("stat") for b in ru["by_world"]]))
over = {"runs": [dict(r) for r in good["runs"]] * 3, "rule": good["rule"]}
ro = V.grade_prereg(dict(gi), over, W, factory_for)
ck("over-long plan is rejected on shape", not ro["pass"], ro.get("reason"))
rich = {"runs": [{"label": "a", "svc": "score", "model": "alpha", "n": 600, "reps": 5, "seed": 1},
                 {"label": "b", "svc": "score", "model": "bravo", "n": 600, "reps": 5, "seed": 2},
                 {"label": "c", "svc": "score", "model": "bravo", "n": 600, "reps": 5, "seed": 3}],
        "rule": good["rule"]}
rr = V.grade_prereg(dict(gi), rich, W, factory_for)
ck("over-budget plan is rejected on cost", not rr["pass"], rr.get("reason"))
poor = {"runs": [{"label": "a", "svc": "score", "model": "alpha", "n": 600, "seed": 1}],
        "rule": {"expr": "a.acc", "cuts": [[">", 0.0, "ability"], ["else", "style"]]}}
rp = V.grade_prereg(dict(gi), poor, W, factory_for)
ck("constant-answer plan fails", not rp["pass"], rp.get("reason"))

print("7    grade_audit")
aitem = {"id": "q3", "kind": "audit", "defect": "D2_denominator_scored_only",
         "sites": [["analysis/analyze.py", 61, 68]], "lo": 0.512, "hi": 0.538, "max_width": 0.02}
ck("all three right", V.grade_audit(aitem, {"defect": "D2_denominator_scored_only",
                                            "site": "/app/analysis/analyze.py:64",
                                            "corrected": 0.525})["pass"])
ck("wrong defect id fails", not V.grade_audit(aitem, {"defect": "D1_selection_max_over_seeds",
                                                      "site": "analysis/analyze.py:64",
                                                      "corrected": 0.525})["pass"])
ck("wrong line fails", not V.grade_audit(aitem, {"defect": "D2_denominator_scored_only",
                                                 "site": "analysis/analyze.py:12",
                                                 "corrected": 0.525})["pass"])
ck("wrong number fails", not V.grade_audit(aitem, {"defect": "D2_denominator_scored_only",
                                                   "site": "analysis/analyze.py:64",
                                                   "corrected": 0.61})["pass"])
ck("too-wide interval fails", not V.grade_audit(aitem, {"defect": "D2_denominator_scored_only",
                                                        "site": "analysis/analyze.py:64",
                                                        "corrected": {"lo": 0.3, "hi": 0.7}})["pass"])
ck("tight overlapping interval passes", V.grade_audit(aitem, {"defect": "D2_denominator_scored_only",
                                                              "site": "analysis/analyze.py:68",
                                                              "corrected": {"lo": 0.520, "hi": 0.530}})["pass"])

print()
print("FAILED: %d  %s" % (len(FAIL), FAIL) if FAIL else "all checks passed")
sys.exit(1 if FAIL else 0)
