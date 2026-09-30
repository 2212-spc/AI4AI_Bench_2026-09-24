"""K8 postmortem: diagnose why a shipped fine-tune regressed online, and fix it.

Mechanism (hidden; transplanted from the distillation-scaling / data-contamination literature - capacity gap
between teacher and student, eval-set leakage inflating offline numbers, mixture shift starving a slice - with
counterfactual constants):
  * the model's true quality on the watched slice is
        true = ceiling(capacity) * coverage(mixture_share) * stability(lr_scale)
  * the OFFLINE number is `true` inflated by whatever fraction of the eval slice leaked into training:
        offline = true + leak * (1 - true)
  * the ONLINE number is `true` (plus serving noise).
Exactly one knob was moved by the release that regressed; the other three are at their recipe values.  Because
leakage raises offline while capacity/mixture/lr lower online, several single-knob stories reproduce the same
release report.  They are separated only by *intervention*: re-run with one knob reset and see what moves.
The agent submits a diagnosis (which knob) plus a fix (a config), and is graded on both - the fix must recover
quality on hidden re-runs, and the declared cause must survive the grader's counterfactual check.
"""
import json
import re, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

KNOBS = ["capacity", "leak", "mixture", "lr_scale"]
CAUSES = ["capacity", "mixture", "lr_scale"]    # leak is a confounder, never the answer (see sample_params)
RECIPE = {"capacity": 1.0, "leak": 0.0, "mixture": 1.0, "lr_scale": 1.0}
RERUN_COST = 1.0
CANARY_COST = 4.0                    # measuring ONLINE quality needs live traffic; offline re-runs are cheap
BUDGET = 20.0                        # a blind 3-knob canary ablation (3 x 4 = 12, +2 for a baseline) does not
                                     # leave enough to confirm anything, and 2 reps of it is impossible
TAU_REL = 0.90                       # the fix must recover this fraction of the healthy-minus-regressed gap
HEADROOM = 0.30                      # how far back toward the recipe value the causal knob may be moved
N_TEST = 3                           # hidden re-runs the fix is scored on


def sample_params(seed):
    g = np.random.default_rng(8000 + seed)
    # `leak` is NOT a candidate cause: leakage inflates the offline number without touching true quality, so
    # it cannot by itself produce an online regression.  It is the CONFOUNDER - present in most instances, it
    # is what makes the release look clean offline and hides which real knob moved.  Diagnosing "the leak" is
    # the single most attractive wrong answer, which is exactly why it must not be scoreable as the cause.
    sick = dict(RECIPE)
    # ALL THREE knobs move by a comparable FRACTION of their recipe value, and the sensitivity curves
    # (ceil_*/cov_*/lr_*) are what decide which move actually hurt.
    #
    # Publishing the shipped values is unavoidable - a postmortem that hid the release config would be
    # absurd - so the magnitudes themselves must not identify the cause.  They did: with the cause moved
    # ~45% and the decoys ~3%, the biggest mover WAS the answer, readable without a single lab call.
    # The mechanism is monotone in every knob (measured: no knob has a flat region where a large move is
    # quality-neutral), so "big move, no damage" cannot be arranged by choosing values alone.  Instead the
    # three moves are drawn from ONE shared fractional deviation and the per-instance sensitivity
    # constants decide the outcome - the gate then keeps only the seeds where exactly one of them matters.
    # Attribution now requires intervention, which is the capability this task exists to measure.
    # All three knobs move by nearly the SAME relative amount (one shared magnitude, +-8% jitter), so the
    # printed deviations are exchangeable and cannot rank the causes.  Drawing the three independently
    # from a wide band was tried and rejected: it let "blame the biggest mover" reach 91/120 (76%) against
    # a 33% chance baseline.  Which knob actually matters is settled by the per-instance sensitivity
    # constants - above all `cov_a` - not by how far each one was moved.  `strat_blame_biggest_move`
    # holds this property in the gate.
    # Equal moves, UNEQUAL HIDDEN SENSITIVITY.  This is the whole design of the instance.
    #
    # With all three knobs moved by the same relative amount AND all three sensitivity constants drawn
    # from narrow bands, no knob dominated: the median dominance margin was 0.084, and the gate accepted
    # 2/150 seeds.  Making the causal MOVE bigger would restore dominance, but that is precisely the leak
    # that started this (blame-the-biggest-mover hit 76%).  So the asymmetry is moved into the part of the
    # world the agent cannot read: one knob per instance is drawn sensitive, the other two insensitive.
    # The release report then shows three equal-looking changes, and which one mattered is discoverable
    # only by intervening on the simulator.  `sens_knob` is a sampling device, not the answer - the cause
    # is still DERIVED from the resulting quality surface, and the gate still verifies dominance.
    sens_knob = CAUSES[int(g.integers(len(CAUSES)))]

    def _sens(knob, ins_lo, ins_hi, sen_lo, sen_hi):
        """Draw a sensitivity constant: the (sen_lo, sen_hi) band if this knob is the sensitive one for
        this instance, else the insensitive (ins_lo, ins_hi) band.  For lr_beta the sensitive band is the
        NUMERICALLY LOWER one (less compensable = more sensitive), which is why the bands are passed in
        that order there; the helper does not assume sen > ins."""
        return g.uniform(sen_lo, sen_hi) if knob == sens_knob else g.uniform(ins_lo, ins_hi)

    # The three knobs do not have the same natural scale - a 40% lr change is drastic where a 40% capacity
    # change is merely bad - so equal RELATIVE moves are not equal-looking moves.  Each knob therefore has
    # a FIXED scale factor, constant across every instance.  That is the property that matters: a ratio
    # which never varies carries no information about which knob is the cause, whereas anything drawn
    # per-instance in correlation with the cause is exactly the leak this is replacing.  The instance
    # picks one shared severity; the fixed ratios turn it into three deviations.
    KSCALE = {"capacity": 1.0, "mixture": 1.0, "lr_scale": 0.85}
    base_dev = float(g.uniform(0.30, 0.46))
    dev = {k: base_dev * KSCALE[k] * float(g.uniform(0.92, 1.08)) for k in CAUSES}
    for k in CAUSES:
        sick[k] = (1.0 + dev[k]) if k == "lr_scale" else (1.0 - dev[k])
    # The release must move MORE THAN ONE knob, or the diagnosis is readable off the instruction.
    #
    # Measured 2026-09-28, and this is the defect the k8 doc leak was only a symptom of.  When the release
    # moved exactly one knob, `instance_truth` set release_knob = cause on 20/20 seeds and the instruction
    # printed it in bold in its second paragraph.  A strategy that does ZERO diagnosis - read the knob off
    # the instruction, reset it to its cap, grid-search the free knobs - passed 5/6 (seeds 1/6/8 x 2 salts).
    # The task was scoring the re-tuning search, not the attribution it claims to test.
    #
    # So the release also perturbs the decoy knobs.  The decoys are drawn in the region where they are
    # nearly quality-neutral (verified by instance_gate: resetting any non-cause knob must recover <= 15%),
    # which is what makes them decoys rather than co-causes: they are visibly CHANGED but not responsible.
    # Now the release report shows three moved knobs and only intervention separates them.

    # leak is always present: without it the offline number drops too and the release report is not
    # confusing at all (measured: leak-free seeds show the regression plainly offline, no diagnosis needed).
    # leak is chosen so that the OFFLINE number lands back near the healthy one: that is what made the
    # release look shippable.  Drawing it blind mostly produces instances where the drop is visible offline
    # too, and those need no diagnosis at all.
    sick["leak"] = 0.0
    # How far back the release lets you move the capped knob is a property of the incident (a serving
    # budget, a data licence, a stability workaround), not a universal constant - so it is drawn per
    # instance rather than fixed module-wide.  It also has to be: with a single global HEADROOM=0.30 the
    # `naive_reset` condition rejected EVERY mixture instance (16/16), because mixture's coverage curve is
    # gentle enough that moving to the cap already recovered ~0.9 of what was reachable and left nothing to
    # re-tune.  Tuning one global number to admit mixture would have loosened capacity and lr_scale too;
    # drawing per instance lets the gate keep the same bar for all three.  Frozen into world.json at build.
    pp = {"cause": None, "headroom": round(float(g.uniform(0.12, 0.34)), 4),
          "sick": {k: round(v, 4) for k, v in sick.items()},
            "ceil_a": round(float(g.uniform(0.86, 0.93)), 4),
            "ceil_b": round(float(_sens("capacity", 0.30, 0.44, 0.56, 0.78)), 4),
            # ceil_k is the curvature of the capacity ceiling: a LOW value spreads the capacity damage
            # over a wide range and makes it large (measured on one instance: ceil_k 5.0 -> stuck-cost
            # 0.022, ceil_k 1.5 -> 0.109), so the sensitive band is again the lower one.  ceil_b alone
            # was not enough - pushing it to 0.95 only reached 0.071 and dragged lr_scale to 0.058 with
            # it, because a capacity-starved model also wants a different lr.  cov_g turned out to be
            # inert for this purpose (identical stuck-costs across 0.2-1.4) and is left alone.
            "ceil_k": round(float(_sens("capacity", 3.1, 4.4, 1.4, 2.4)), 4),
            "cov_a": round(float(_sens("mixture", 0.10, 0.20, 0.30, 0.52)), 4), "cov_p": round(float(g.uniform(1.25, 1.55)), 4),
            "cov_g": round(float(g.uniform(0.55, 0.85)), 4),
            "lr_base": round(float(g.uniform(0.95, 1.05)), 4),
            # lr_beta is how strongly the best lr tracks capacity, i.e. how COMPENSABLE a stuck lr is.
            # It, not lr_c, is what makes lr_scale the cause: raising lr_c saturates (the stability term
            # is 1/(1+lr_c*d^2), so stuck-cost peaked at 0.079 and then fell), while lowering lr_beta from
            # 0.50 to 0.10 took the same instance from 0.076 to 0.345.  A high lr_beta lets capacity be
            # re-tuned to make the shipped lr right again, so lr can never be the dominant cause.
            "lr_beta": round(float(_sens("lr_scale", 0.62, 0.80, 0.10, 0.30)), 4),
            "lr_c": round(float(g.uniform(1.6, 2.4)), 4),
            "sig": round(float(g.uniform(0.006, 0.011)), 5),
            "cost": {"capacity": 3.0, "leak": 1.0, "mixture": 1.5, "lr_scale": 1.0}}
    # The cause is DERIVED, not drawn.  It used to be sampled first and the knob values chosen to match,
    # which is what forced the causal move to be the large one.  Now all three moves are comparable and
    # the cause is simply whichever reset recovers the most - a fact about this instance's sensitivity
    # constants, not about how its numbers were printed.  `instance_gate` then keeps only the seeds where
    # that knob dominates the other two by a wide margin, so "the cause" is always well defined.
    _heal = true_quality(pp, RECIPE)
    _cost = {k: _heal - best_with_held(pp, k, cap=False) for k in CAUSES}
    pp["cause"] = max(_cost, key=_cost.get)
    cause = pp["cause"]
    q_sick = true_quality(pp, pp["sick"]); q_heal = true_quality(pp, RECIPE)
    # offline = q_sick + leak*(1-q_sick); solve for the leak that lands it within a hair of q_heal
    target = q_heal - float(g.uniform(0.004, 0.022))
    pp["sick"]["leak"] = round(float(np.clip((target - q_sick) / max(1e-6, 1 - q_sick), 0.0, 0.85)), 4)
    return pp


def _ceiling(p, c):
    return p["ceil_a"] * (1 - p["ceil_b"] * math.exp(-p["ceil_k"] * c))


def _coverage(p, m, c):
    """A slice that lost training share hurts less when the model has capacity to spare: the damage from a
    skewed mixture is divided by capacity.  (Transplanted, counterfactual constants.)"""
    return 1 - p["cov_a"] * (1 - min(m, 1.0)) ** p["cov_p"] / max(0.3, c) ** p["cov_g"]


def _stability(p, lr, c):
    """There is an interior best learning rate and it MOVES WITH CAPACITY - a smaller model wants a larger
    lr.  This is what makes a one-knob fix insufficient: cap the capacity and the recipe lr is no longer
    the right lr."""
    opt = p["lr_base"] * max(0.3, c) ** (-p["lr_beta"])
    return 1.0 / (1.0 + p["lr_c"] * ((lr - opt) / opt) ** 2)


def true_quality(p, cfg):
    c = cfg["capacity"]
    return _ceiling(p, c) * _coverage(p, cfg["mixture"], c) * _stability(p, cfg["lr_scale"], c)


_GRID = {"capacity": np.linspace(0.35, 1.0, 27), "mixture": np.linspace(0.45, 1.0, 23),
         "lr_scale": np.linspace(0.6, 2.0, 29)}


def best_with_held(p, held, cfg0=None, cap=True):
    """Best quality reachable while `held` stays at its shipped value and the other two are re-tuned.

    This is the quantity that defines the cause, and it has to be a CONSTRAINED optimum rather than a
    single-knob reset.  The knobs are coupled on purpose - the best lr moves with capacity - so resetting
    one knob alone can easily make things *worse* (measured: resetting capacity alone scored -0.47 on the
    normalised scale, because the shipped lr was then wrong for it).  Under coupling, "which knob caused
    the regression" only means something as "which knob, by being stuck where the release left it, costs
    the most once everything else has been re-tuned around it".  That is also exactly what R2 checks.
    """
    base = dict(cfg0 or p["sick"])
    free = [k for k in CAUSES if k != held]
    best = -1.0
    for v0 in _GRID[free[0]]:
        for v1 in _GRID[free[1]]:
            c = dict(base)
            c[free[0]] = _clip_constraint(p, free[0], float(v0)) if cap else float(v0)
            c[free[1]] = _clip_constraint(p, free[1], float(v1)) if cap else float(v1)
            q = true_quality(p, c)
            if q > best:
                best = q
    return best


def observe(p, cfg, salt, tag, rep=0):
    """One re-run: the offline eval number (leak-inflated) and the online number (the truth), both noisy."""
    t = true_quality(p, cfg)
    off = t + cfg["leak"] * (1 - t)
    g = rng_for(salt, "obs", tag, rep, *[round(cfg[k], 6) for k in KNOBS])
    return {"offline_slice": round(float(np.clip(off + g.normal(0, p["sig"]), 0, 1)), 4),
            "online_slice": round(float(np.clip(t + g.normal(0, p["sig"]), 0, 1)), 4),
            "offline_other": round(float(np.clip(0.79 + g.normal(0, p["sig"]), 0, 1)), 4)}


def best_reachable(p):
    """Best quality reachable under the release constraints: the causal knob goes to its cap, and the free
    knobs are re-tuned around it.  Computed by grid + local refine on the noiseless surface."""
    best = (-1.0, None)
    caps = [_clip_constraint(p, "capacity", v) for v in np.linspace(0.3, 1.0, 36)]
    mixes = [_clip_constraint(p, "mixture", v) for v in np.linspace(0.4, 1.0, 31)]
    lrs = [_clip_constraint(p, "lr_scale", v) for v in np.linspace(0.5, 2.2, 69)]
    for c in caps:
        for m in mixes:
            for lr in lrs:
                cfg = {"capacity": c, "mixture": m, "lr_scale": lr, "leak": p["sick"]["leak"]}
                q = true_quality(p, cfg)
                if q > best[0]:
                    best = (q, dict(cfg))
    return best


def truth(p):
    healthy = true_quality(p, RECIPE)
    q_best, cfg_best = best_reachable(p)
    sick = true_quality(p, p["sick"])
    return {"q_healthy": round(healthy, 5), "q_sick": round(sick, 5), "cause": p["cause"],
            "gap": round(healthy - sick, 5), "sick_value": p["sick"][p["cause"]],
            "q_best": round(q_best, 5), "cfg_best": {k: round(v, 4) for k, v in cfg_best.items()},
            "gap_reachable": round(q_best - sick, 5)}


# ----------------------------------------------------------------------------------------------- lab ops
def cap_of(p):
    """The release moved ONE knob for a real reason (serving cost, a data licence, a stability workaround),
    and that reason has not gone away: the fix may move it only part of the way back.  Only that knob is
    capped - the knobs the release never touched are free.  So naming the right knob is necessary but not
    sufficient: the remaining loss has to be bought back with the free knobs."""
    # Every knob the release moved is capped, not just the causal one.  While only the cause was capped,
    # the set of capped knobs WAS the answer: an agent could have read it off `lab spec`.  Same headroom
    # for all three, so the caps carry no information about which knob is responsible.
    h = p.get("headroom", HEADROOM)
    return {k: p["sick"][k] + h * (RECIPE[k] - p["sick"][k]) for k in CAUSES}


def _clip_constraint(p, k, v):
    cap = cap_of(p).get(k)
    if cap is None:
        return v
    return min(v, cap) if RECIPE[k] > p["sick"][k] else max(v, cap)


def _parse_cfg(a, p, field="config"):
    c = a.get(field)
    if not isinstance(c, dict):
        raise LabError("%s must be an object with keys %s" % (field, KNOBS))
    out = {}
    for k in KNOBS:
        v = c.get(k, p["sick"][k])            # omitted knobs keep their RELEASE value, not the recipe value
        # capacity and mixture share cannot exceed the recipe: you cannot buy more serving budget or more
        # of a licensed corpus than the project already had.  lr is free to move either way.
        lo, hi = {"capacity": (0.2, 1.0), "mixture": (0.2, 1.0), "lr_scale": (0.2, 2.2), "leak": (0.0, 1.0)}[k]
        out[k] = _clip_constraint(p, k, num(v, k, lo, hi))
    return out


def _cost_rerun(w, a):
    """An offline re-run is cheap; changing an expensive knob costs retraining."""
    c = _parse_cfg(a, w.p)
    reps = num(a.get("reps", 1), "reps", 1, 3, integer=True)
    per = 1.0 + sum(w.p["cost"][k] - 1.0 for k in ("capacity", "mixture") if abs(c[k] - w.p["sick"][k]) > 1e-9) * 0.5
    return float(RERUN_COST * reps * max(1.0, per))


def _run_rerun(w, a, ctx):
    """Offline evaluation only - and the offline slice is leak-inflated, which is the whole trap."""
    c = _parse_cfg(a, w.p)
    reps = int(a.get("reps", 1))
    obs = [observe(w.p, c, w.salt, "R%d" % ctx["i"], r) for r in range(reps)]
    return {"config": {k: round(c[k], 4) for k in KNOBS},
            "runs": [{"offline_slice": o["offline_slice"], "offline_other": o["offline_other"]} for o in obs],
            "mean": {k: round(float(np.mean([o[k] for o in obs])), 4) for k in ("offline_slice", "offline_other")}}


def _cost_canary(w, a):
    c = _parse_cfg(a, w.p)
    reps = num(a.get("reps", 1), "reps", 1, 3, integer=True)
    per = 1.0 + sum(w.p["cost"][k] - 1.0 for k in ("capacity", "mixture") if abs(c[k] - w.p["sick"][k]) > 1e-9) * 0.5
    return float(CANARY_COST * reps * max(1.0, per))


def _run_canary(w, a, ctx):
    """Serve a config to live traffic and measure the number users actually experience."""
    c = _parse_cfg(a, w.p)
    reps = int(a.get("reps", 1))
    obs = [observe(w.p, c, w.salt, "C%d" % ctx["i"], r) for r in range(reps)]
    return {"config": {k: round(c[k], 4) for k in KNOBS},
            "runs": [{"online_slice": o["online_slice"]} for o in obs],
            "mean_online_slice": round(float(np.mean([o["online_slice"] for o in obs])), 4)}


def _cost_decon(w, a):
    return float(RERUN_COST)


def _run_decon(w, a, ctx):
    """Re-score an existing config on a decontaminated eval slice: removes the leak inflation only."""
    c = _parse_cfg(a, w.p)
    t = true_quality(w.p, c)
    g = rng_for(w.salt, "dec", ctx["i"], *[round(c[k], 6) for k in KNOBS])
    return {"config": {k: round(c[k], 4) for k in KNOBS},
            "offline_slice_decontaminated": round(float(np.clip(t + g.normal(0, w.p["sig"]), 0, 1)), 4),
            "note": "same checkpoint, eval slice with train-overlapping items removed"}


class World(_W):
    NAME = "k8_post"
    ARTIFACTS = ["report.json"]
    # R3 compares the fix to the cause THIS SAME artifact declares, so it is self-consistency, not a claim
    # about the instance: it transfers 8/8 by construction and must be exempt from the transfer gate.
    SELF_CONSISTENT = ("R3_fix_acts_on_cause",)
    BUDGET_UNIT = "re-runs"
    OPS = {"rerun": (_cost_rerun, _run_rerun, "re-run the fine-tune offline with knobs you choose"),
           "canary": (_cost_canary, _run_canary, "serve a config to live traffic and measure online quality"),
           "decontaminate": (_cost_decon, _run_decon, "re-score a config on a decontaminated eval slice")}

    def public_spec(self):
        return {"ops": {
            "rerun": {"args": {"config": "object with any of %s; omitted knobs keep their RELEASE value" % KNOBS,
                               "reps": "independent repeats, 1..3 (default 1)"},
                      "cost": "1 re-run per rep, x1.5-2 when capacity or mixture is changed (retraining)",
                      "returns": "per repeat: offline_slice, offline_other; plus their means"},
            "canary": {"args": {"config": "object, same shape as rerun", "reps": "1..3 (default 1)"},
                       "cost": "%g re-runs per rep, x1.5-2 when capacity or mixture is changed" % CANARY_COST,
                       "returns": "the ONLINE slice number of that config, averaged over the repeats"},
            "decontaminate": {"args": {"config": "object, same shape as rerun"}, "cost": "1 re-run",
                              "returns": "the offline slice number of that config on a decontaminated eval slice"}},
            "deliverable": "/app/report.json (see /app/docs/report_format.md)"}

    def grade(self, art_dir, ledger=None):
        t = truth(self.p)
        rep, err = None, None
        try:
            rep = json.load(open(os.path.join(art_dir, "report.json")))
        except Exception as e:
            err = "report.json missing or unreadable: %s" % type(e).__name__
        items = {}
        if err:
            items["R0_report"] = (False, err)
            return {"pass": False, "score": 0.0, "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()},
                    "diag": {}, "ref": t}
        cause = rep.get("cause")
        fix = rep.get("fix")
        ok0, why = True, []
        if cause not in CAUSES:
            ok0 = False; why.append("cause must be one of %s, got %r" % (CAUSES, cause))
        if not isinstance(fix, dict):
            ok0 = False; why.append("fix must be an object of knob -> value")
        else:
            for k in fix:
                if k not in KNOBS:
                    ok0 = False; why.append("unknown knob %r in fix" % k)
        items["R0_report"] = (ok0, "; ".join(why) or "report.json declares a cause from %s and a fix config" % CAUSES)
        if not ok0:
            return {"pass": False, "score": 0.0, "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()},
                    "diag": {}, "ref": t}

        # R1: the fix must actually recover online quality on hidden re-runs
        cfg = dict(self.p["sick"])
        cfg.update({k: _clip_constraint(self.p, k, float(v)) for k, v in fix.items()})
        q = float(np.mean([observe(self.p, cfg, self.salt, "G%d" % i)["online_slice"] for i in range(N_TEST)]))
        rec = (q - t["q_sick"]) / max(1e-9, t["gap_reachable"])
        items["R1_fix_recovers"] = (rec >= TAU_REL,
                                    "your fix reaches online %.4f (release %.4f, best reachable under the release "
                                    "constraints %.4f) -> recovers %.1f%% of what is reachable, need %.0f%%"
                                    % (q, t["q_sick"], t["q_best"], 100 * rec, 100 * TAU_REL))

        # R2: the declared cause must survive a counterfactual check.
        #
        # The check is "how much does leaving THIS knob where the release put it cost, once the other two
        # are re-tuned freely" - and that must be the largest such cost of the three.  It cannot be the
        # older "resetting only this knob recovers": the knobs are coupled (the best lr tracks capacity),
        # so on the current instances resetting the true cause alone often makes quality WORSE, and that
        # version rejected even the oracle 0/4.  The caps are ignored here on purpose: the question is
        # which knob broke it, not how far back it may be put.
        heal = t["q_healthy"]
        stuck = {k: (heal - best_with_held(self.p, k, cap=False)) / max(1e-9, t["gap"]) for k in CAUSES}
        runner = max(v for k, v in stuck.items() if k != cause)
        cause_ok = bool(stuck[cause] >= max(stuck.values()) - 1e-9 and stuck[cause] - runner >= 0.15)
        items["R2_cause_survives_intervention"] = (
            cause_ok, "holding `%s` at its release value costs %.1f%% of the regression once the other knobs "
                      "are re-tuned; the knobs rank %s - the cause must rank first by >= 15 points"
                      % (cause, 100 * stuck[cause],
                         ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in sorted(stuck.items(), key=lambda x: -x[1]))))

        # R3: the fix must be minimal - changing knobs that were never broken is not a diagnosis
        # the fix has to actually act on the knob it blames; compensating with the free knobs is fine and in
        # fact necessary, but a "fix" that never touches the declared cause is not a fix for that cause.
        moved = abs(float(fix.get(cause, self.p["sick"][cause])) - self.p["sick"][cause]) > 1e-6
        items["R3_fix_acts_on_cause"] = (moved, "fix moves `%s`, the knob you diagnosed" % cause if moved
                                         else "fix never changes `%s`, the knob you declared as the cause" % cause)
        diag = {"declared_cause": cause, "true_cause": t["cause"], "fix": fix, "q_fixed": round(q, 4),
                "recovered": round(rec, 4)}
        if ledger is not None:
            diag["n_rerun"] = sum(1 for r in ledger if r["op"] == "rerun")
            diag["n_canary"] = sum(1 for r in ledger if r["op"] == "canary")
            diag["n_decon"] = sum(1 for r in ledger if r["op"] == "decontaminate")
            diag["spent"] = round(sum(r["cost"] for r in ledger), 2)
        # bool() is not decoration: `rec >= TAU_REL` is a numpy scalar, and whatever json encoder is in
        # play wrote it out as 1.0/0.0 rather than true/false, so every downstream reader of grade.json
        # had to special-case `v["ok"] is True or v["ok"] == 1.0`.  Coerce at the source instead.
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(rec, 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": t}


# ----------------------------------------------------------------------------------------------- strategies
def _write(art_dir, cause, fix):
    json.dump({"cause": cause, "fix": fix}, open(os.path.join(art_dir, "report.json"), "w"))


def strat_oracle(sess, art_dir, rng):
    """Existence proof: rule the knobs out one at a time by intervention.

    The release report cannot distinguish them, so reset each candidate knob on its own and watch the ONLINE
    number; the one whose reset recovers quality is the cause.  Decontamination is used only to confirm that
    the clean-looking offline number was inflated - it never identifies the cause by itself."""
    # offline re-runs are cheap, so use them to narrow the field first: on a decontaminated eval the
    # leak inflation is gone, and a knob that is not the cause moves the clean number by almost nothing.
    # (This baseline used to be measured twice, the first result discarded - 1 re-run of pure waste out
    # of 20.  With the search now starting from the caps it needs that call back.)
    d0 = sess.call("decontaminate", {"config": {}})["offline_slice_decontaminated"]
    cand = []
    for k in CAUSES:
        d = sess.call("decontaminate", {"config": {k: RECIPE[k]}})["offline_slice_decontaminated"]
        cand.append((d - d0, k))
    cand.sort(reverse=True)
    k = cand[0][1]
    # then spend the expensive online measurement only on confirming the single survivor
    # the release report already gives the online number of the shipped config, so only the candidate fix
    # needs a (costly) canary.
    on_fix = sess.call("canary", {"config": {k: RECIPE[k]}, "reps": 1})["mean_online_slice"]
    if on_fix <= truth(sess.w.p)["q_sick"]:     # the cheap screen was wrong; fall back to the runner-up
        k = cand[1][1]
    # naming the knob is not the fix: the cap means the free knobs have to be re-tuned around it, and the
    # offline (decontaminated) surface is cheap enough to search for that.
    #
    # The free knobs have to be searched in BOTH directions, and the direction depends on which knob was
    # capped.  Measured: when `capacity` is the cause, the cap holds capacity low and the fix is to raise
    # `lr_scale` to the smaller model's optimum.  When `lr_scale` is the cause the cap holds lr HIGH, and the
    # only way to make a high lr right is to SHRINK capacity - the same coupling read backwards.  A grid that
    # only looks at capacity in [0.85, 1.0] misses that entirely and leaves s ~ 0.70-0.88 on those instances.
    # Start from ALL THREE knobs at their caps, not just the causal one.  The release moved every knob,
    # so `{k: RECIPE[k]}` leaves the other two sitting at their shipped (bad) values while asking the
    # lab to score a value the caps will silently clip anyway - the search then begins outside the
    # feasible region and coordinate descent never recovers.  Measured on the capacity-cause instances,
    # which is where it bit: seeds 78 and 208 gave oracle 0/4 with R1 stuck at 0.81-0.90 against a 0.90
    # bar, while the very same grid can reach rec 0.93-1.00 from a capped start.  An oracle has to
    # demonstrate the task is solvable WITHIN its constraints, so it must optimise inside them.
    caps = cap_of(sess.w.p)
    cfg = {kk: caps[kk] for kk in CAUSES}
    best = (sess.call("decontaminate", {"config": cfg})["offline_slice_decontaminated"], dict(cfg))
    grid = {"lr_scale": (0.8, 1.0, 1.2, 1.4, 1.5), "capacity": (0.55, 0.7, 0.85, 1.0), "mixture": (0.85, 1.0)}
    free = [x for x in ("capacity", "lr_scale", "mixture") if x != k]
    for _sweep in range(2):                         # coordinate descent: one pass sets the coupling, the
        moved = False                               # second re-optimises the partner against it
        for fk in free:
            for v in grid[fk]:
                if sess.left() < 2 * RERUN_COST:
                    break
                c2 = dict(best[1])
                if abs(c2.get(fk, p_default(sess, fk)) - v) < 1e-9:
                    continue
                c2[fk] = v
                q = sess.call("decontaminate", {"config": c2})["offline_slice_decontaminated"]
                if q > best[0] + 1e-4:
                    best = (q, c2); moved = True
        if not moved:
            break
    _write(art_dir, k, best[1])


def p_default(sess, knob):
    """Knobs absent from a config keep their RELEASE value - that is what the lab does, so the search has to
    reason about the same baseline."""
    return float(sess.w.p["sick"][knob])


def strat_cap_everything(sess, art_dir, rng):
    """Move every knob to its cap and guess a cause.  No diagnosis, no lab calls, no search.

    This is the cheapest thing the constraints allow, and it is why R1 alone cannot carry the task:
    measured over the gated pool, all-knobs-at-caps satisfies R1 on 122/153 seeds (80%).  The caps are
    published in the instruction, so this costs nothing at all.  R2 is what stands between it and a pass,
    and this strategy exists to keep that true - if a change ever makes R2 satisfiable without an
    attribution, the gate fails here rather than in a results table.
    """
    caps = cap_of(sess.w.p)
    _write(art_dir, CAUSES[int(rng.integers(len(CAUSES)))], {k: caps[k] for k in CAUSES})


def strat_blame_leak(sess, art_dir, rng):
    """The most attractive wrong answer: decontaminate, see the offline number was inflated, call it the cause."""
    sess.call("decontaminate", {"config": {}})
    _write(art_dir, "leak", {"leak": 0.0})


def strat_fix_all(sess, art_dir, rng):
    """Reset every knob: quality recovers, but nothing was diagnosed.

    The declared cause is drawn from the salt rather than hardcoded.  It used to be the literal string
    "capacity", which made this strategy accidentally CORRECT on every capacity-cause instance - it then
    passed the gate and was reported as a leak in those seeds (25, 110, 208) when the real problem was
    that the decoy was cheating in the task's favour.  A decoy must be wrong for the reason it is meant
    to be wrong: here, that a blanket reset is not a diagnosis.  Picking blind makes it wrong 2/3 of the
    time on the cause and, when it does guess right, R2's ranking check still has to be satisfied by a
    fix that reset everything - which is the property actually under test.
    """
    k = CAUSES[int(rng.integers(len(CAUSES)))]
    _write(art_dir, k, dict(RECIPE))


def strat_biggest_offline(sess, art_dir, rng):
    """Pick whichever single-knob reset moves the OFFLINE number most - leak-inflated, so it misleads."""
    best = None
    for k in CAUSES:
        r = sess.call("rerun", {"config": {k: RECIPE[k]}, "reps": 1})["mean"]["offline_slice"]
        if best is None or r > best[1]:
            best = (k, r)
    _write(art_dir, best[0], {best[0]: RECIPE[best[0]]})


def _fixed_cause(c):
    def f(sess, art_dir, rng, _c=c):
        _write(art_dir, _c, {_c: RECIPE[_c]})
    return f


def strat_partial_fix(sess, art_dir, rng):
    """Right cause, half-hearted fix: moves the knob only part of the way back.

    Identification is done the CHEAP way here, exactly as the oracle does it, and that is the point of the
    rewrite.  The first version of this decoy identified the cause by brute force - one canary per candidate
    at 4 re-runs each - and ran out of budget before it could write the artifact, so it failed every salt on
    `R0_report` with a LabError instead of on the thing it exists to test.  A decoy that dies before it plays
    its strategy is a gate that proves nothing: `partial_fix` was nominally passing the gate (expect=fail,
    0/2) while never once submitting a partial fix.  Found 2026-09-28 by reading the gate log rather than its
    verdict.  Now it reaches the artifact and fails on `R1_fix_recovers`, which is the claim being made -
    that moving the blamed knob halfway back does not recover 90% of what is reachable.
    """
    d0 = sess.call("decontaminate", {"config": {}})["offline_slice_decontaminated"]
    cand = []
    for k in CAUSES:
        d = sess.call("decontaminate", {"config": {k: RECIPE[k]}})["offline_slice_decontaminated"]
        cand.append((d - d0, k))
    cand.sort(reverse=True)
    k = cand[0][1]
    half = sess.w.p["sick"][k] + 0.5 * (RECIPE[k] - sess.w.p["sick"][k])
    _write(art_dir, k, {k: half})


def strat_nop(sess, art_dir, rng):
    pass


def strat_bad_json(sess, art_dir, rng):
    open(os.path.join(art_dir, "report.json"), "w").write("the cause was capacity")


def strat_copy_doc(sess, art_dir, rng):
    """Copy the format example out of docs/ and submit it, spending nothing.

    This strategy exists because the gate was structurally blind to the defect it catches.  Every other
    `fail` strategy here plays a bad REASONING process; none of them played the DOCUMENTATION.  So when
    `docs/report_format.md` shipped a concrete worked example whose cause happened to be the true cause of
    seed 1, the gate passed the instance 10/10 and four of five models went on to submit the example's
    lr_scale verbatim.  A material a task hands the agent is part of the task's attack surface, and the
    gate only tests what a strategy plays.  If the doc is ever made copyable again, this fails loudly.
    """
    doc = docs(sess.w.p)["docs/report_format.md"]
    cause = next((c for c in CAUSES if '"%s"' % c in doc), None)
    fix = {k: float(v) for k, v in re.findall(r'"(capacity|lr_scale|mixture)":\s*([0-9.]+)', doc)}
    if cause is None or not fix:            # placeholders, as intended -> nothing to copy
        _write(art_dir, "<one of>", {})     # submit the placeholder itself: must fail R0
        return
    _write(art_dir, cause, fix)


def strat_read_instruction(sess, art_dir, rng):
    """Skip the diagnosis: blame whichever knob the release report says moved furthest, then re-tune.

    The counterpart of `copy_doc` for the INSTRUCTION.  While the release moved exactly one knob, this
    strategy passed 5/6 (seeds 1/6/8 x 2 salts) without performing any attribution at all, because
    `instance_truth` set release_knob = cause on 20/20 seeds.  The release now moves all three knobs, so
    the report no longer identifies the cause and this must fail.  It is kept permanently so that any
    future change which re-couples the two is caught by the gate rather than by a reader of the results.
    """
    p = sess.w.p
    mv = instance_truth({"params": p, "salt": sess.w.salt, "truth": {}})["release_moved"]
    k = max(mv, key=lambda kk: abs(mv[kk][1] - mv[kk][0]) / max(1e-9, abs(mv[kk][0])))
    cfg = {k: RECIPE[k]}
    best = (sess.call("decontaminate", {"config": cfg})["offline_slice_decontaminated"], dict(cfg))
    grid = {"lr_scale": (0.8, 1.0, 1.1, 1.2, 1.3, 1.4), "capacity": (0.55, 0.7, 0.85, 1.0), "mixture": (0.85, 1.0)}
    for fk in [x for x in ("capacity", "lr_scale", "mixture") if x != k]:
        for v in grid[fk]:
            if sess.left() < 2 * RERUN_COST:
                break
            c2 = dict(best[1]); c2[fk] = v
            q = sess.call("decontaminate", {"config": c2})["offline_slice_decontaminated"]
            if q > best[0] + 1e-4:
                best = (q, c2)
    _write(art_dir, k, best[1])


STRATEGIES = {"oracle": (strat_oracle, "pass"), "blame_leak": (strat_blame_leak, "fail"),
              "copy_doc": (strat_copy_doc, "fail"), "read_instruction": (strat_read_instruction, "fail"),
              "cap_everything": (strat_cap_everything, "fail"),
              "fix_all": (strat_fix_all, "fail"), "biggest_offline": (strat_biggest_offline, "fail"),
              "partial_fix": (strat_partial_fix, "fail"), "nop": (strat_nop, "fail"),
              "bad_json": (strat_bad_json, "fail")}
for _c in CAUSES:
    globals()["strat_guess_" + _c] = _fixed_cause(_c)
    STRATEGIES["guess_" + _c] = (globals()["strat_guess_" + _c], "fail")
# These decoys guess the cause at random, so on 1/3 of salts they are accidentally right and pass.  That
# is not a leak - it is the guess being lucky - so they are judged over many salts against the 1/3 rate
# rather than required to fail every time.  `fix_all` and `cap_everything` used to hardcode "capacity",
# which hid this by making them deterministic and silently wrong on 2/3 of instances.
NOISY_FAIL = ("biggest_offline", "fix_all", "cap_everything")


def _rank_by_move(p):
    """The three causal knobs ordered by relative deviation from the recipe, largest first."""
    return sorted(CAUSES, key=lambda k: -abs(p["sick"][k] - RECIPE[k]) / abs(RECIPE[k]))


def balanced_pool(seeds, per_cause=None):
    """Pick a cause-balanced subset of gated seeds, so `always_<c>` cannot beat chance.

    The generator does not produce the three causes equally often (over 400 seeds the gated pool was
    mixture 77 / lr_scale 58 / capacity 18), and rebalancing it by widening the sensitivity bands would
    trade one leak for another.  Balance is a property of WHAT IS SHIPPED, so it is imposed at selection
    time: take the same number of each cause, in seed order, deterministically.
    """
    by = {c: [] for c in CAUSES}
    for sd in seeds:
        by[sample_params(sd)["cause"]].append(sd)
    k = per_cause or min(len(v) for v in by.values())
    return sorted(sd for c in CAUSES for sd in by[c][:k])


def pool_gate(seeds, tol=0.12):
    """Pool-level screen: no cheap heuristic may beat chance across the instances actually shipped.

    A per-instance exclusion cannot do this job, and trying was instructive.  Rejecting every seed whose
    cause was the BIGGEST mover took that heuristic to 0/42 - and handed 64% to "smallest mover", because
    with three knobs, forbidding one rank concentrates the answer in the other two.  Pinning the middle
    rank instead gave "middle mover" 100%.  Excluding a correlation creates its complement: the property
    that has to hold is that the cause's rank is UNIFORM over the pool, which is a fact about the pool and
    can only be checked there.  Balance over causes is the same kind of property, and is checked here too.

    Returns (ok, info).  `info["worst"]` is the best-scoring cheap heuristic and its rate; chance is 1/3.
    """
    ps = [sample_params(sd) for sd in seeds]
    n = len(ps)
    heur = {"biggest_move": lambda p: _rank_by_move(p)[0],
            "middle_move": lambda p: _rank_by_move(p)[1],
            "smallest_move": lambda p: _rank_by_move(p)[2],
            "narrowest_headroom": lambda p: min(CAUSES, key=lambda k: abs(cap_of(p)[k] - p["sick"][k])),
            "widest_headroom": lambda p: max(CAUSES, key=lambda k: abs(cap_of(p)[k] - p["sick"][k]))}
    heur.update({"always_" + c: (lambda c: (lambda p: c))(c) for c in CAUSES})
    rates = {nm: sum(f(p) == p["cause"] for p in ps) / n for nm, f in heur.items()}
    worst = max(rates.items(), key=lambda x: x[1])
    return bool(worst[1] <= 1.0 / len(CAUSES) + tol), {"n": n, "worst": worst, "rates": rates,
                                                       "causes": {c: sum(p["cause"] == c for p in ps) for c in CAUSES}}


def instance_gate(p):
    """Cheap noiseless screen: the release must look clean offline but be clearly broken online, the true
    cause must be the only knob whose reset recovers, and the regression must be far bigger than the noise."""
    t = truth(p)
    o_sick = true_quality(p, p["sick"]) + p["sick"]["leak"] * (1 - true_quality(p, p["sick"]))
    o_heal = true_quality(p, RECIPE)
    # Identifiability is measured with the SAME definition that derives the cause: how much quality is
    # lost by leaving knob k stuck at its shipped value while the other two are re-tuned freely.  The old
    # single-knob-reset version disagreed with that definition once the knobs were coupled (it scored
    # resetting the true cause at -0.47 on some instances, because the shipped lr was wrong for the
    # restored capacity) and rejected every seed.  Normalised so 1.0 = "this knob accounts for the whole
    # regression".
    heal = true_quality(p, RECIPE)
    rec = {k: (heal - best_with_held(p, k, cap=False)) / max(1e-9, t["gap"]) for k in CAUSES}
    others = [v for k, v in rec.items() if k != p["cause"]]
    info = {"gap": t["gap"], "offline_drop": round(o_heal - o_sick, 4), "recover_by_knob": {k: round(v, 3) for k, v in rec.items()},
            "cause": p["cause"], "snr": round(t["gap"] / p["sig"], 1)}
    naive = dict(p["sick"]); naive[p["cause"]] = _clip_constraint(p, p["cause"], RECIPE[p["cause"]])
    naive_rec = (true_quality(p, naive) - t["q_sick"]) / max(1e-9, t["gap_reachable"])
    info["naive_reset"] = round(naive_rec, 3)
    # `rec[cause] >= 0.98` was the right bar only while the release moved a single knob: with the decoys
    # also perturbed, resetting the cause alone cannot recover 100% by construction, and that threshold
    # would reject every instance (measured: 0/40 seeds).  The property that actually has to hold is
    # IDENTIFIABILITY BY INTERVENTION - the cause must dominate every decoy by a wide margin - so it is
    # restated as a dominance margin rather than weakened to a smaller number.  The decoy draw is tightened
    # toward the recipe instead, which is what keeps `rec[cause]` high honestly.
    ok = (t["gap"] >= 0.07                         # the online regression is real
          and abs(o_heal - o_sick) <= 0.035        # ... but offline it looks fine: that is the whole puzzle
          and rec[p["cause"]] >= 0.55              # the cause accounts for most of the regression
          and rec[p["cause"]] - max(others) >= 0.25  # ... and dominates every decoy by a clear margin
          and max(others) <= 0.45                  # no decoy is a co-cause
          and t["gap"] / p["sig"] >= 12            # the effect is far above the observation noise
          and naive_rec <= TAU_REL - 0.12          # and knowing the knob is not enough: the free knobs must
                                                   # have to be re-tuned around the capped one
          )
    return bool(ok), info


# ----------------------------------------------------------------------------------------------- agent facing
def instance_truth(cfg):
    """Release-report numbers, cached at build time so the instruction can quote what the team saw."""
    p = cfg["params"]; salt = cfg["salt"]
    rel = observe(p, p["sick"], salt, "REL")
    heal = observe(p, RECIPE, salt, "PRE")
    # `release_knob` used to be p["cause"], which put the answer in the instruction's second paragraph.
    # The release now reports ALL THREE knobs it moved, in a fixed alphabetical order that carries no
    # signal, and the caps for all three - so the report says what was changed (public fact) without
    # saying which change was responsible (the thing to be diagnosed).  The knob-level caps are published
    # for every knob because the fix has to respect them regardless of which one turns out to be the cause.
    caps = {k: round(float(p["sick"][k] + p.get("headroom", HEADROOM) * (RECIPE[k] - p["sick"][k])), 4)
            for k in CAUSES}
    return {"rel_offline": rel["offline_slice"], "rel_online": rel["online_slice"],
            "pre_offline": heal["offline_slice"], "pre_online": heal["online_slice"],
            "rel_other": rel["offline_other"],
            "release_moved": {k: [RECIPE[k], p["sick"][k]] for k in CAUSES},
            "caps": caps, "cap": caps[p["cause"]]}


def instruction(p, t):
    mv = t.get("release_moved") or {k: [RECIPE[k], p["sick"][k]] for k in CAUSES}
    cp = t.get("caps") or cap_of(p)
    return """# Postmortem: the fine-tune that looked fine offline

Release `ft-2026.09` shipped last week and online quality on the **watched slice** dropped. The offline
eval suite did not catch it - that is why it shipped.

What the release report says:

| | before (`ft-2026.08`) | after (`ft-2026.09`) |
|---|---|---|
| offline, watched slice | %.4f | %.4f |
| offline, everything else | ~0.79 | %.4f |
| online, watched slice | %.4f | **%.4f** |

Four knobs describe a fine-tune in this project. This release changed **three** of them, each for its own
reason, and none of those reasons has gone away. Here is the recipe, what shipped, and how far back each
one may be moved:

    knob       recipe   shipped   may be restored only as far as
    capacity   1.0      %.3f     %.3f     student capacity headroom on the watched slice
    mixture    1.0      %.3f     %.3f     the watched slice's share of the fine-tuning mixture
    lr_scale   1.0      %.3f     %.3f     fine-tuning learning rate vs the recipe default
    leak       0.0      ?         -        fraction of the eval slice that overlaps the training set

**One of those three changes caused the online regression. The release report cannot tell you which** -
all three shipped together, and the offline number came out clean. `leak` is not a knob anyone sets: it is
a property of the eval slice, it is *measured*, this quarter's overlap audit is still open, and the `0.0`
above is the recipe's assumption rather than a measurement of what shipped.

`capacity` and `mixture` cannot go above their recipe values - there is no more serving budget and no more
licensed data.

## The lab

Run `/app/bin/lab spec` first. Budget: **%.0f re-runs**.

* `lab rerun` re-runs the fine-tune with knobs you choose and gives you the **offline** numbers (1 re-run,
  more if it has to retrain);
* `lab canary` serves a config to live traffic and gives you the **online** number - this is the number that
  regressed, and it costs **%.0f re-runs** a go;
* `lab decontaminate` re-scores a config on an eval slice with the train-overlapping items removed (1 re-run).

Knobs you leave out of a config keep their **release** value, not their recipe value.

## Deliverable: `/app/report.json`

    {"cause": "<one of %s>",
     "fix":   {"<knob>": <value>, ...}}

## How it is graded (mechanically, against the simulator; only the file counts)

1. **R0** - `report.json` parses, names a cause from %s and gives a fix.
2. **R1** - running your fix recovers **>= %.0f%%** of what is reachable online under the release
   constraint, measured on %d hidden re-runs.
3. **R2** - your declared cause survives the grader's counterfactual: resetting *only* that knob must
   recover the regression, and holding *only* that knob at its release value must reproduce it.
4. **R3** - your fix actually moves the knob you blamed.

All four must hold. Note what R1 and R3 do together: naming the right knob is not a fix, and a fix that
recovers quality by accident does not tell you the cause.

Write `/app/report.json` and stop.
""" % (t.get("pre_offline", 0.0), t.get("rel_offline", 0.0), t.get("rel_other", 0.0),
       t.get("pre_online", 0.0), t.get("rel_online", 0.0),
       mv["capacity"][1], cp["capacity"], mv["mixture"][1], cp["mixture"],
       mv["lr_scale"][1], cp["lr_scale"],
       BUDGET, CANARY_COST, CAUSES, CAUSES, 100 * TAU_REL, N_TEST)


def docs(p):
    # The format example must not be a usable ANSWER.  Measured 2026-09-28: this example used
    # cause "capacity" with lr_scale 1.2, and on k8a1 the true cause IS capacity - so four of five
    # models (gemflash, gpt6, sonnet5, gpt55) submitted the byte-identical fix {capacity: 0.773,
    # lr_scale: 1.2}, with 1.2 copied straight from here and 0.773 read off the instruction.  gpt55
    # scored 1.008 on R1 while declaring the WRONG cause, which is how little the fix had to be earned.
    # Only fable deviated (1.15).  Placeholders are now obviously-invalid sentinels: a model that
    # copies them fails R0 loudly instead of passing quietly.
    return {"docs/report_format.md": """# report.json

    {
      "cause": "<one of %s>",
      "fix": {"<knob>": 0.0,          // the knob you blamed, moved as far as the release note allows
              "<other knob>": 0.0}    // and whatever else you had to re-tune around it
    }

The values above are placeholders, not a worked example - fill in what you measured.

Knobs you leave out keep their **release** value. The grader clamps `fix` to the same release constraint the
lab enforces, so asking for a value beyond the cap silently gets you the cap.
""" % CAUSES,
            "docs/knobs.md": """# The four knobs

| knob | recipe | meaning |
|---|---|---|
| `capacity` | 1.0 | student capacity headroom on the watched slice. Below 1.0 the student cannot absorb what the teacher knows about that slice. |
| `leak` | 0.0 | fraction of the eval slice that overlaps the fine-tuning set. Raises the **offline** number. Has no effect on what users get. |
| `mixture` | 1.0 | the watched slice's share of the fine-tuning mixture, relative to the recipe. |
| `lr_scale` | 1.0 | fine-tuning learning rate relative to the recipe default. |

Notes from the training team, collected during the incident review:

* "offline eval and online serving use the same prompts but different sampling; they normally agree to
  within a few thousandths"
* "we have never re-tuned the learning rate after changing model size - the recipe value has always worked"
* "the data team added a new source this quarter; the overlap audit is still open"
* "the watched slice is small, so its share of the mixture moves easily when anything else is added"
""",
            "docs/incident.md": """# ft-2026.09 incident timeline

* day 0 - release ships; offline suite green, no slice below threshold
* day 2 - support escalates quality complaints concentrated in the watched slice
* day 3 - online dashboard confirms a drop; offline suite re-run on the shipped checkpoint, still green
* day 4 - rollback rejected: the knob the release changed cannot be fully reverted
* day 4 - three hypotheses on the table, no way to choose between them from the numbers we have:
  the student is too small for this slice / the mixture starved the slice / the recipe lr is wrong now
* day 5 - you are asked to find which one it is and ship a fix
"""}


def hints(p):
    return {1: "\n## Hint\nThe offline suite and the online number disagree. Before you trust either, ask what "
               "each one measures - and which of the four knobs can move one without moving the other.\n",
            2: "\n## Hint\nLeakage inflates the offline number and leaves online untouched, so it explains why the "
               "release looked clean - it cannot explain the regression itself. Separate the remaining candidates "
               "by intervention, not by staring at the numbers: reset one knob at a time and see what moves. The "
               "online measurement is expensive, so narrow the field on the decontaminated offline number first. "
               "And once you know the knob, remember the recipe values of the *other* knobs were tuned for the "
               "recipe value of this one.\n"}
