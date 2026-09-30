"""B-CONTROL-MIN: the b_control instances with the world model taken OUT of the task text.

Why this file exists (v8 frontier round 1, 2026-09-29): claude-fable-5-1 passed b_control 2/2 in ~20 lab
calls by executing, step for step, the protocol the task text described.  b_control's instruction printed
`reported = quality(weights) + offset(harness configuration)`, its docs said "two checkpoints scored under the
same harness configuration carry the same offset", and rubric item R3 asked for "changes that move the
reported number without moving the model".  The v8 gates certified that the *naive* protocols fail; they
never certified that the right protocol is non-obvious - and with the decomposition written down it is a
textbook exercise.  Correctness gates and difficulty are different claims (HLE / SWE-V lesson), and so are
"shortcut-resistant" and "hard".

This variant is the controlled experiment for one lever - DISCLOSURE LEVEL - on the SAME instances:
  * same sample_params, same hidden mechanics, same truth for R1/R2, same instance_gate and pool;
  * the text gives operational facts only: what each op does, what it costs, its noise, and what the default
    of each argument is.  No decomposition, no stated consequence, no rubric item whose definition is itself
    the insight (R3 is dropped for that reason - its definition *was* the hint);
  * the target is made unambiguous WITHOUT naming a harness configuration: it is defined on a third, frozen
    instrument, the external suite (`ext`), which is what customers see.  `ext` is too noisy/expensive to
    produce the required interval by itself (checked in `instance_gate`, condition X2), so the agent still has
    to use the cheap in-house harness - and has to decide, on its own, how to configure it;
  * `score` without `harness=` now uses the checkpoint's OWN manifest (a retrained build is evaluated with
    the harness built from the same manifest), which is what a real build system does.  So the coupled
    leave-one-out design is the DEFAULT PATH, not a design the agent has to go out of its way to write.

DISCOVERABILITY CERTIFICATE (new in v8, required whenever the text stops disclosing the model).  Hiding the
model is only fair if the agent can find it out.  Two standard, cheap diagnostics on agent-visible data:
  D1 cross-instrument check - the in-house default drop (previous -> release) vs the external drop on the same
     two archived checkpoints: they differ by sum(r) (|sum r| >= 3 points), rejected at >= 3 sigma with 4 ext
     units per checkpoint (condition X1);
  D2 same-weights check - one archived checkpoint scored under two harness configurations moves by |r_h| for
     the loud change, rejected at >= 3 sigma at n = 2 (condition X1b).
Either one falsifies "the in-house number measures the model".  Neither is suggested by the text.
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for
from . import b_control as B
from .b_control import (CHANGES, BUDGET, TRAIN_COST, SCORE_UNIT, MAX_N, WIDTH_CAP, EPS_E, DELTA_R, MARGIN,
                        sample_params, quality, offset, truth, _parse_set, _n, _ckpt_id, _cost_train,
                        _run_train, _write, _train, _half)

EXT_COST = 4.0            # per unit: third-party run, real money
EXT_SIG = 0.30            # per-unit noise of the external suite (it is a smaller item set)
EXT_MAX_N = 8


def ext_const(p):
    """External-suite items differ from the in-house items, so the two report different absolute levels.  The
    constant is hidden and irrelevant to every graded quantity (all are differences), but it stops an agent
    from reading `offset` off the absolute gap between the two instruments on one checkpoint."""
    g = np.random.default_rng(57700 + int(round(p["q0"] * 1000)) % 100000)
    return round(float(g.uniform(-4.0, -1.0)), 3)


# ------------------------------------------------------------------ ops
def _ckpt_rev(ck):
    if not isinstance(ck, str) or not ck:
        raise LabError("ckpt is required: 'release', 'previous', or an id returned by `train`")
    if ck == "release":
        return []
    if ck == "previous":
        return list(CHANGES)
    if ck.startswith("ck_rev_"):
        tail = ck[len("ck_rev_"):]
        rev = [] if tail == "none" else [t[:1] + "-" + t[1:] for t in tail.split("_")]
        if any(h not in CHANGES for h in rev) or _ckpt_id(sorted(rev)) != ck:
            raise LabError("no such checkpoint %r" % ck)
        return sorted(rev)
    raise LabError("no such checkpoint %r (archived: 'release', 'previous'; or train one)" % ck)


def _harness_arg(a, rev):
    """Omitted -> the checkpoint's own manifest (same reversion set as its training).  Given (even as an
    empty list) -> exactly that.  `None` and a missing key both mean omitted."""
    if a.get("harness", None) is None:
        return list(rev), True
    return _parse_set(a, "harness"), False


def _cost_score(w, a):
    n = _n(a)
    rev = _ckpt_rev(a.get("ckpt"))
    _harness_arg(a, rev)
    return n * SCORE_UNIT


def _run_score(w, a, ctx):
    p = w.p
    n = _n(a)
    ck = a.get("ckpt")
    rev = _ckpt_rev(ck)
    har, dflt = _harness_arg(a, rev)
    true = quality(p, set(rev)) + offset(p, set(har))
    g = rng_for(w.salt, "score", ck, ",".join(har), ctx["i"])
    vals = [true + float(g.normal(0, p["sig"])) for _ in range(n)]
    return {"ckpt": ck, "harness_reverted": har, "harness_source": "checkpoint manifest" if dflt else "explicit",
            "n": n, "score": round(float(np.mean(vals)), 4), "per_unit": [round(v, 4) for v in vals]}


def _cost_ext(w, a):
    _ckpt_rev(a.get("ckpt"))
    return int(num(a.get("n", 1), "n", 1, EXT_MAX_N, integer=True)) * EXT_COST


def _run_ext(w, a, ctx):
    p = w.p
    n = int(num(a.get("n", 1), "n", 1, EXT_MAX_N, integer=True))
    ck = a.get("ckpt")
    rev = _ckpt_rev(ck)
    true = quality(p, set(rev)) + ext_const(p)
    g = rng_for(w.salt, "ext", ck, ctx["i"])
    vals = [true + float(g.normal(0, EXT_SIG)) for _ in range(n)]
    return {"ckpt": ck, "n": n, "score": round(float(np.mean(vals)), 4), "per_unit": [round(v, 4) for v in vals]}


class World(_W):
    NAME = "b_control_min"
    ARTIFACTS = ["report.json"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "credits"
    OPS = {"train": (_cost_train, _run_train,
                     "retrain with a subset of the release reverted; returns a scoreable checkpoint id"),
           "score": (_cost_score, _run_score, "run a checkpoint through the in-house eval harness"),
           "ext": (_cost_ext, _run_ext, "run a checkpoint through the external suite")}

    def public_spec(self):
        return {"ops": {
            "train": {"args": {"revert": "list of change ids to roll back in the training pipeline"},
                      "cost": "%g credits" % TRAIN_COST,
                      "returns": "a checkpoint id; deterministic given the reversion set"},
            "score": {"args": {"ckpt": "'release', 'previous', or an id from `train`",
                               "harness": "optional: list of change ids rolled back in the in-house harness.  "
                                          "If omitted, the harness is built from the checkpoint's own manifest "
                                          "(the same changes rolled back as in its training)",
                               "n": "eval units, 1..%d" % MAX_N},
                      "cost": "n x %g credits" % SCORE_UNIT,
                      "noise": "+-%.3f points per unit" % self.p["sig"],
                      "returns": "mean in-house score over n units, and the per-unit values"},
            "ext": {"args": {"ckpt": "'release', 'previous', or an id from `train`",
                             "n": "eval units, 1..%d" % EXT_MAX_N},
                    "cost": "n x %g credits" % EXT_COST,
                    "noise": "+-%.2f points per unit" % EXT_SIG,
                    "returns": "mean external-suite score over n units, and the per-unit values"}},
            "changes_in_release": CHANGES,
            "archived_checkpoints": ["release", "previous"],
            "deliverable": "/app/report.json (see /app/docs/report_format.md)"}

    def grade(self, art_dir, ledger=None):
        p, t = self.p, truth(self.p)
        items = {}
        try:
            rep = json.load(open(os.path.join(art_dir, "report.json")))
        except Exception as ex:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {"R0_report": {"ok": False,
                                            "detail": "report.json missing or unreadable: %s" % type(ex).__name__}}}
        why, ok0 = [], True
        cul = rep.get("culprit")
        if cul not in CHANGES:
            ok0 = False; why.append("culprit must be one of %s, got %r" % (", ".join(CHANGES), cul))
        iv = rep.get("effect")
        if not (isinstance(iv, dict) and "lo" in iv and "hi" in iv):
            ok0 = False; why.append("effect must be an object with numeric lo/hi")
        items["R0_report"] = (ok0, "; ".join(why) or "report.json is well-formed")
        if not ok0:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()}}
        gain = {h: -p["e"][h] for h in CHANGES}          # external-suite gain = quality gain (ext has no offset)
        items["R1_culprit"] = (
            cul == t["culprit"],
            "reverting `%s` raises the external-suite score by %+.3f; the best single reversion is `%s` at "
            "%+.3f.  Full table (true external gain from reverting each alone): %s"
            % (cul, gain[cul], t["culprit"], gain[t["culprit"]],
               ", ".join("%s %+.3f" % (h, gain[h]) for h in CHANGES)))
        try:
            lo, hi = float(iv["lo"]), float(iv["hi"])
            bad = not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi)
        except Exception:
            lo = hi = float("nan"); bad = True
        if bad:
            items["R2_effect"] = (False, "malformed interval")
        else:
            cov, wide = lo <= gain[cul] <= hi, (hi - lo) > WIDTH_CAP + 1e-9   # float slack: 0.450 written = cap
            items["R2_effect"] = (
                bool(cov and not wide),
                "the true external-suite gain from reverting YOUR named change `%s` is %+.3f; you reported "
                "[%+.3f, %+.3f] (%s, width %.3f, cap %.2f).  The in-house default (own-manifest) leave-one-out "
                "design would have estimated %+.3f for it." % (cul, gain[cul], lo, hi,
                                                              "covers" if cov else "misses", hi - lo, WIDTH_CAP,
                                                              t["coupled_estimate"][cul]))
        diag = {"named": cul, "true_culprit": t["culprit"], "coupled_culprit": t["coupled_culprit"],
                "rescore_culprit": t["rescore_culprit"],
                "named_the_coupled_answer": cul == t["coupled_culprit"],
                "named_the_rescore_answer": cul == t["rescore_culprit"],
                "effect_width": None if bad else round(hi - lo, 4),
                "effect_centre_minus_coupled": None if bad else round((lo + hi) / 2 - t["coupled_estimate"][cul], 4)}
        if ledger is not None:
            tr = [r for r in ledger if r["op"] == "train"]
            sc = [r for r in ledger if r["op"] == "score"]
            ex = [r for r in ledger if r["op"] == "ext"]
            explicit = [r for r in sc if r["args"].get("harness", None) is not None]
            hs = set(tuple(sorted(r["args"].get("harness") or [])) for r in explicit)
            # the process facts that separate the protocols: did the agent ever set the harness explicitly,
            # and did it ever compare a retrained checkpoint with another checkpoint under one configuration?
            fixed_pairs = 0
            by_cfg = {}
            for r in explicit:
                by_cfg.setdefault(tuple(sorted(r["args"].get("harness") or [])), set()).add(r["args"].get("ckpt"))
            for cfg, cks in by_cfg.items():
                if len(cks) >= 2 and any(str(c).startswith("ck_rev_") for c in cks):
                    fixed_pairs += 1
            diag.update({"n_train": len(tr), "n_score": len(sc), "n_ext": len(ex),
                         "n_score_explicit_harness": len(explicit), "distinct_explicit_configs": len(hs),
                         "configs_with_retrained_vs_other_ckpt": fixed_pairs,
                         "spent": round(sum(r["cost"] for r in ledger), 2)})
        n_ok = sum(1 for v in items.values() if v[0])
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(n_ok / 3.0, 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": t}


# ------------------------------------------------------------------ strategies
def _score(sess, ck, harness=None, n=4):
    a = {"ckpt": ck, "n": int(n)}
    if harness is not None:
        a["harness"] = list(harness)
    return sess.call("score", a)["score"]


def _ext(sess, ck, n):
    return sess.call("ext", {"ckpt": ck, "n": int(n)})["score"]


def _write2(art_dir, culprit, lo, hi):
    json.dump({"culprit": culprit, "effect": {"lo": round(lo, 4), "hi": round(hi, 4)}},
              open(os.path.join(art_dir, "report.json"), "w"))


def strat_oracle(sess, art_dir, rng):
    """Hold the in-house harness at ONE explicit configuration for every checkpoint compared.

        base      : release under FIX, 8 units                                   ->  8
        quality   : retrain each single reversion, score under FIX, 8 units    -> 5 x (8 + 8) = 80
        refine    : winner and release again under FIX, 8 units each           -> 16
                                                                                 ----- 104 / 120
    Any fixed configuration works; the offset cancels in every difference, and the external suite differs
    from any fixed in-house configuration by a constant, so the differences are the external differences."""
    p = sess.w.p
    FIX = []
    base = _score(sess, "release", FIX, 8)
    q = {}
    for h in CHANGES:
        ck = _train(sess, [h])
        q[h] = _score(sess, ck, FIX, 8) - base
    cul = max(q, key=lambda h: q[h])
    a = _score(sess, _ckpt_id([cul]), FIX, 8)
    b = _score(sess, "release", FIX, 8)
    # pool the two independent contrasts: (a1 - b1) and (a2 - b2), each with 8+8 units
    eff = 0.5 * ((q[cul]) + (a - b))
    half = 2.8 * p["sig"] * math.sqrt(2.0 / 16)
    _write2(art_dir, cul, eff - half, eff + half)


def strat_coupled_loo(sess, art_dir, rng):
    """P3 ABLATION: the default path.  Retrain without change i and score it the way the build system scores
    any build - with its own manifest's harness.  Reproducible, tight, and estimating -(e_i + r_i)."""
    p = sess.w.p
    base = _score(sess, "release", None, 6)
    d = {}
    for h in CHANGES:
        ck = _train(sess, [h])
        d[h] = _score(sess, ck, None, 6) - base
    cul = max(d, key=lambda h: d[h])
    half = _half(p["sig"], 6, 6)
    _write2(art_dir, cul, d[cul] - half, d[cul] + half)


def strat_rescore_only(sess, art_dir, rng):
    """P4 ABLATION: never retrain; roll each change back in the harness only, on the archived release."""
    p = sess.w.p
    base = _score(sess, "release", [], 8)
    d = {h: _score(sess, "release", [h], 8) - base for h in CHANGES}
    cul = max(d, key=lambda h: d[h])
    half = _half(p["sig"], 8, 8)
    _write2(art_dir, cul, d[cul] - half, d[cul] + half)


def strat_ext_culprit_coupled_effect(sess, art_dir, rng):
    """P3b ABLATION, realistic form: use the trustworthy external suite to NAME the culprit (it is
    unbiased, just noisy), then get the tight interval the cheap way - own-manifest in-house scoring.
    R1 may well be right; R2 is centred on -(e+r), which the gate keeps >= 1.5 widths from -e."""
    p = sess.w.p
    base_e = _ext(sess, "release", 3)
    ge = {}
    for h in CHANGES:
        ck = _train(sess, [h])
        ge[h] = _ext(sess, ck, 2) - base_e
    cul = max(ge, key=lambda h: ge[h])
    a = _score(sess, _ckpt_id([cul]), None, 8)
    b = _score(sess, "release", None, 8)
    half = _half(p["sig"], 8, 8)
    _write2(art_dir, cul, (a - b) - half, (a - b) + half)


def strat_ext_only(sess, art_dir, rng):
    """Everything on the external suite, honest interval: unbiased and far too wide at this budget."""
    base_e = _ext(sess, "release", 4)
    ge = {}
    for h in CHANGES:
        ck = _train(sess, [h])
        ge[h] = _ext(sess, ck, 2) - base_e
    cul = max(ge, key=lambda h: ge[h])
    half = 2.8 * EXT_SIG * math.sqrt(1.0 / 2 + 1.0 / 4)
    _write2(art_dir, cul, ge[cul] - half, ge[cul] + half)


def strat_right_culprit_coupled_effect(sess, art_dir, rng):
    t = truth(sess.w.p)
    v = t["coupled_estimate"][t["culprit"]]
    _write2(art_dir, t["culprit"], v - 0.1, v + 0.1)


def strat_wide(sess, art_dir, rng):
    t = truth(sess.w.p)
    g = t["culprit_gain"]
    _write2(art_dir, t["culprit"], g - 1.2, g + 1.2)


def strat_nop(sess, art_dir, rng):
    pass


def strat_bad_json(sess, art_dir, rng):
    open(os.path.join(art_dir, "report.json"), "w").write("{nope")


STRATEGIES = {"oracle": (strat_oracle, "pass"),
              "coupled_loo": (strat_coupled_loo, "fail"),
              "rescore_only": (strat_rescore_only, "fail"),
              "ext_culprit_coupled_effect": (strat_ext_culprit_coupled_effect, "fail"),
              "ext_only": (strat_ext_only, "fail"),
              "right_culprit_coupled_effect": (strat_right_culprit_coupled_effect, "fail"),
              "wide": (strat_wide, "fail"),
              "nop": (strat_nop, "fail"),
              "bad_json": (strat_bad_json, "fail")}
NOISY_FAIL = ()
SEARCH = ["coupled_loo", "rescore_only", "ext_only"]
PRINCIPLES = {"P3_hold_the_ruler_fixed":          ("coupled_loo", ["R1_culprit"]),
              "P3b_effect_needs_a_fixed_ruler":   ("ext_culprit_coupled_effect", ["R2_effect"]),
              "P4_model_side_needs_retraining":   ("rescore_only", ["R1_culprit"]),
              "P6_cheap_instrument_is_needed":    ("ext_only", ["R2_effect"])}


# ------------------------------------------------------------------ grader falsification
def _mut_culprit(p, t, art_dir, leg):
    rep = B._load(art_dir)
    if leg == "small":
        rep["notes"] = "extra key; nothing else in the file is read"
        rep["effect"] = {"lo": float(rep["effect"]["lo"]), "hi": float(rep["effect"]["hi"])}
    else:
        other = [h for h in CHANGES if h != t["culprit"]]
        rep["culprit"] = other[0]
        g = t["revert_gain"][other[0]]
        rep["effect"] = {"lo": g - 0.4 * WIDTH_CAP, "hi": g + 0.4 * WIDTH_CAP}
    B._save(art_dir, rep)
    return None if leg == "small" else "R1_culprit"


MUTATE = [("culprit_label", _mut_culprit),
          ("effect_coverage", B._mut_effect_coverage),
          ("effect_width", B._mut_effect_width)]


# ------------------------------------------------------------------ instance screen
def _conditions_min(p):
    t = truth(p)
    r = p["r"]
    loud = t["coupled_culprit"]
    sr = sum(r.values())
    # X1 / D1: previous vs release, in-house default (own manifests) vs external, 4 ext units and 4 in-house
    # units per checkpoint.  The two drops differ by sum(r).
    sd1 = math.sqrt(2 * p["sig"] ** 2 / 4 + 2 * EXT_SIG ** 2 / 4)
    z1 = abs(sr) / sd1
    # X1b / D2: the archived release under the default harness vs with the coupled winner rolled back in the
    # harness, 2 units each.
    z1b = abs(r[loud]) / math.sqrt(2 * p["sig"] ** 2 / 2)
    # X2: the external suite alone cannot deliver R2 at this budget: the narrowest honest 2.8-sigma interval it
    # can buy on ONE contrast, after the five retrains, exceeds the cap.
    units = int((BUDGET - len(CHANGES) * TRAIN_COST) // EXT_COST)
    best_half = 2.8 * EXT_SIG * math.sqrt(2.0 / (units / 2.0))
    x2 = 2 * best_half > WIDTH_CAP
    # X3: the oracle's interval fits and its ranking survives its own noise.
    x3 = 2 * 2.8 * p["sig"] * math.sqrt(2.0 / 16) <= WIDTH_CAP and \
        t["margin"] >= 3.0 * p["sig"] * math.sqrt(2.0 / 8) * math.sqrt(2)
    return [z1 >= 3.0, z1b >= 3.0, x2, x3], {"disc_z_cross_instrument": round(z1, 2),
                                            "disc_z_same_weights": round(z1b, 2),
                                            "ext_best_width": round(2 * best_half, 3), "sum_r": round(sr, 3)}


def instance_gate(p):
    ok, info = B.instance_gate(p)
    c2, info2 = _conditions_min(p)
    info = dict(info)
    info.update(info2)
    info["c_min"] = [bool(x) for x in c2]
    return bool(ok and all(c2)), info


def pool_gate(seeds):
    ps = [s for s in seeds if instance_gate(sample_params(s))[0]]
    n = len(ps)
    if n < 4:
        return True, {"n": n}
    culs = {}
    for s in ps:
        c = truth(sample_params(s))["culprit"]
        culs[c] = culs.get(c, 0) + 1
    mc = max(culs.values())
    return bool(mc <= math.ceil(n / 4.0)), {"n": n, "modal_culprit": mc, "cap_culprit": math.ceil(n / 4.0),
                                            "culprits": culs}


# ------------------------------------------------------------------ task text: operational facts only
def instruction(p, t):
    ids = ", ".join("`%s`" % h for h in CHANGES)
    prev_dash = t["q_previous"] + p["r0"]                                  # previous, own manifest (all reverted)
    rel_dash = t["q_release"] + p["r0"] + sum(p["r"].values())           # release, own manifest (none reverted)
    return """# Release regression: which change should we roll back?

Last week's runtime release bundled five changes, %s, shipped from one manifest.  Since then the in-house
eval dashboard has dropped from **%.2f** (checkpoint `previous`) to **%.2f** (checkpoint `release`).

Leadership will roll back exactly **one** change.  They want the one whose rollback buys back the most model
quality **as customers see it - on the external suite** - and a number for how much it buys.

You have **%g credits**.

* `lab train revert=<ids>` - retrain with those changes rolled back in the training pipeline.  Returns a
  checkpoint id.  **%g credits.**  Deterministic given the reversion set.
* `lab score ckpt=<id> n=<1..%d> [harness=<ids>]` - the in-house eval harness.  **n credits**,
  **+-%.3f points** of noise per unit.  `harness` lists changes rolled back in the harness; if you omit it,
  the harness is built from the checkpoint's own manifest, i.e. with the same changes rolled back as in its
  training (`release`: none, `previous`: all five).  That is how the dashboard numbers above were produced.
* `lab ext ckpt=<id> n=<1..%d>` - the external suite, run by a third party.  **n x %g credits**,
  **+-%.2f points** of noise per unit.

`ckpt` may be `release`, `previous`, or any id `train` returned.  Both evals report accuracy in percent on
held-out problems from the same task distribution (different item sets).  Noise is independent and mean
zero; there is no other randomness in this task.

## Deliverable: `/app/report.json`

```json
{"culprit": "%s",
 "effect": {"lo": 0.00, "hi": 0.00}}
```

* `culprit` - the single change whose rollback raises the **external-suite** score the most.
* `effect` - an interval for that gain, in external-suite points: the external score of a checkpoint retrained
  with exactly the change you named rolled back, minus that of `release`.  Width may not exceed **%.2f points**.
""" % (ids, prev_dash, rel_dash, BUDGET, TRAIN_COST, MAX_N, p["sig"], EXT_MAX_N, EXT_COST, EXT_SIG,
       CHANGES[0], WIDTH_CAP)


def docs(p):
    return {"docs/report_format.md": """# /app/report.json

```json
{"culprit": "H-2",
 "effect": {"lo": 1.31, "hi": 1.58}}
```

* `culprit` (string) - one change id.
* `effect.lo`, `effect.hi` (numbers) - external-suite points gained by rolling back the change you named in
  `culprit`.  Graded against that change, not against the correct one, so a wrong culprit costs you one item
  and not two.  `hi - lo` must be <= %.2f.

Nothing else in the file is read.  There is no partial credit inside an item.
""" % WIDTH_CAP}


def hints(p):
    return {1: """
Hint 1: the in-house number for a rebuilt checkpoint can differ from the release's for more than one reason.
""", 2: """
Hint 2: compare checkpoints under one explicitly fixed harness configuration.
"""}
