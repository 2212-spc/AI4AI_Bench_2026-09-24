"""B-CONTROL: attribute a release regression when one of the changes moved the measuring device.

Difficulty mechanism (this family's *signature*):
  P3  HOLD THE RULER FIXED.  A reported eval number decomposes as `measured = quality(model) +
      offset(harness)`.  Five changes shipped together; some touch code shared by the training pipeline
      and the eval harness, so reverting a change can move `offset` as well as `quality`.  The natural
      experiment - "revert change i and re-measure" - reverts it on BOTH sides at once, so it estimates
      `-(e_i + r_i)` and not `-e_i`.  One of the five moves the ruler hard and the model not at all; under
      the coupled design it is the biggest apparent win in the table, and the table is internally
      consistent, reproducible, and wrong.
  P4  A MODEL-SIDE CHANGE NEEDS A NEW MODEL.  Re-scoring an archived checkpoint under a reverted harness
      is cheap (no training) and returns a number that looks like a reversion result.  It is not one: the
      weights did not change, so that route measures the ruler only.  The cheap op is the tempting op.

What makes the wrong answers survive the agent's own validation: both wrong protocols produce a complete
leave-one-out table with tight, reproducible intervals.  Nothing inside either protocol is inconsistent -
the estimand is wrong, not the arithmetic - so re-running the measurements confirms them.

Ground truth is exact and closed-form: training is deterministic given the reversion set, quality and
harness offset are additive in the per-change effects, and `score` adds mean-zero measurement noise only.
So `-e_i` for every change, the true culprit, and the harness-only set are computed, not sampled.

Distinct from A-STACK: nothing here is non-additive and there is no proxy workload.  The whole difficulty
is that the outcome variable is contaminated by the treatment, which no amount of repetition fixes.
"""
import itertools, json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

CHANGES = ["H-1", "H-2", "H-3", "H-4", "H-5"]
BUDGET = 120.0
TRAIN_COST = 8.0          # a reversion has to be retrained; this is the expensive op
SCORE_UNIT = 1.0          # one eval unit = 1 credit; noise falls as 1/sqrt(n)
MAX_N = 8
WIDTH_CAP = 0.45          # absolute score points allowed for the effect interval
EPS_E = 0.30              # |true quality effect| <= EPS_E  =>  "does not move the model"
DELTA_R = 0.90            # |true harness offset|  >= DELTA_R =>  "moves the reported number"
MARGIN = 0.60             # required separation between the true culprit and the runner-up

_ROLES = [
    # (e-range, e-sign, r-range, r-sign)   e = effect on model quality when the change is APPLIED
    #                                      r = effect on the harness offset when APPLIED
    ("culprit",      (1.85, 2.70), -1, (0.42, 0.54), +1),   # the real regression - and it MASKS itself:
    #     its harness patch pushed the reported number UP, so the coupled leave-one-out design shows a
    #     smaller win for reverting it than the truth.  Without this the coupled estimate would be exactly
    #     unbiased for the one change that matters and the R2 decoy would pass.
    ("loud_ruler",   (0.00, 0.05), +1, (2.40, 3.60), -1),   # huge apparent win, zero model effect
    ("small_mover",  (0.78, 1.10), -1, (0.15, 0.30), -1),   # model-side AND a little ruler: in neither set
    ("quiet_ruler",  (0.00, 0.05), -1, (1.48, 1.85), -1),   # second harness-only member
    ("helpful",      (0.78, 1.30), +1, (0.00, 0.05), +1),   # reverting it would HURT
]


def sample_params(seed):
    """Roles are fixed in kind and permuted across the change IDs, so no ID carries an answer between
    instances; the magnitudes are redrawn.  Signs on the two near-zero roles are randomised so that
    `harness_only` cannot be read off the sign of anything."""
    g = np.random.default_rng(57000 + seed)
    perm = [int(x) for x in g.permutation(len(_ROLES))]
    e, r, role = {}, {}, {}
    for i, h in enumerate(CHANGES):
        name, (elo, ehi), es, (rlo, rhi), rs = _ROLES[perm[i]]
        sgn_e = es if ehi > 0.10 else int(g.choice([-1, 1]))
        sgn_r = rs if rhi > 0.10 else int(g.choice([-1, 1]))
        e[h] = round(sgn_e * float(g.uniform(elo, ehi)), 4)
        r[h] = round(sgn_r * float(g.uniform(rlo, rhi)), 4)
        role[h] = name
    return {"e": e, "r": r, "role": role,
            "q0": round(float(g.uniform(54.0, 63.0)), 3),     # quality of the pre-release checkpoint
            "r0": round(float(g.uniform(-1.5, 1.5)), 3),      # harness offset with every change reverted
            "sig": round(float(g.uniform(0.055, 0.085)), 4)}    # score noise at n = 1


# ------------------------------------------------------------------ hidden mechanics
def quality(p, reverted):
    """Quality of the checkpoint trained with `reverted` rolled back.  Deterministic in the reversion set:
    training uses a fixed seed, which is stated in the task text so that repeats are never the answer."""
    return p["q0"] + sum(p["e"][h] for h in CHANGES if h not in reverted)


def offset(p, harness):
    return p["r0"] + sum(p["r"][h] for h in CHANGES if h not in harness)


def truth(p):
    rev = {h: -p["e"][h] for h in CHANGES}                       # gain in QUALITY from reverting h alone
    naive = {h: -(p["e"][h] + p["r"][h]) for h in CHANGES}       # what the coupled design estimates
    cheap = {h: -p["r"][h] for h in CHANGES}                     # what re-scoring an archive estimates
    order = sorted(CHANGES, key=lambda h: -rev[h])
    culprit = order[0]
    harness_only = sorted(h for h in CHANGES
                          if abs(p["e"][h]) <= EPS_E and abs(p["r"][h]) >= DELTA_R)
    return {"revert_gain": {h: round(rev[h], 4) for h in CHANGES},
            "coupled_estimate": {h: round(naive[h], 4) for h in CHANGES},
            "rescore_estimate": {h: round(cheap[h], 4) for h in CHANGES},
            "culprit": culprit,
            "culprit_gain": round(rev[culprit], 4),
            "runner_up": order[1],
            "margin": round(rev[culprit] - rev[order[1]], 4),
            "harness_only": harness_only,
            "coupled_culprit": max(CHANGES, key=lambda h: naive[h]),
            "rescore_culprit": max(CHANGES, key=lambda h: cheap[h]),
            "total_regression": round(sum(p["e"][h] for h in CHANGES), 4),
            "q_release": round(quality(p, set()), 4),
            "q_previous": round(p["q0"], 4),
            "roles": p["role"]}


# ------------------------------------------------------------------ ops
def _parse_set(a, key):
    v = a.get(key, [])
    if v is None:
        v = []
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",") if x.strip()]
    if not isinstance(v, list):
        raise LabError("%s must be a list of change ids" % key)
    out = []
    for x in v:
        if x not in CHANGES:
            raise LabError("unknown change id %r; the release contains %s" % (x, ", ".join(CHANGES)))
        if x in out:
            raise LabError("duplicate change id %r" % x)
        out.append(x)
    return sorted(out)


def _n(a):
    return int(num(a.get("n", 1), "n", 1, MAX_N, integer=True))


def _ckpt_id(rev):
    return "ck_rev_" + ("none" if not rev else "_".join(h.replace("-", "") for h in rev))


def _cost_train(w, a):
    _parse_set(a, "revert")
    return TRAIN_COST


def _run_train(w, a, ctx):
    rev = _parse_set(a, "revert")
    return {"ckpt": _ckpt_id(rev), "reverted_in_training": rev,
            "note": "training is deterministic given the reversion set; this checkpoint is now scoreable"}


def _cost_score(w, a):
    n = _n(a)
    _parse_set(a, "harness")
    return n * SCORE_UNIT


def _run_score(w, a, ctx):
    p = w.p
    n = _n(a)
    har = _parse_set(a, "harness")
    ck = a.get("ckpt")
    if not isinstance(ck, str) or not ck:
        raise LabError("ckpt is required: 'release', 'previous', or an id returned by `train`")
    if ck == "release":
        rev = []
    elif ck == "previous":
        rev = list(CHANGES)
    elif ck.startswith("ck_rev_"):
        tail = ck[len("ck_rev_"):]
        rev = [] if tail == "none" else [t[:1] + "-" + t[1:] for t in tail.split("_")]
        if any(h not in CHANGES for h in rev) or _ckpt_id(sorted(rev)) != ck:
            raise LabError("no such checkpoint %r" % ck)
        rev = sorted(rev)
    else:
        raise LabError("no such checkpoint %r (archived: 'release', 'previous'; or train one)" % ck)
    true = quality(p, set(rev)) + offset(p, set(har))
    g = rng_for(w.salt, "score", ck, ",".join(har), ctx["i"])
    vals = [true + float(g.normal(0, p["sig"])) for _ in range(n)]
    return {"ckpt": ck, "harness_reverted": har, "n": n,
            "score": round(float(np.mean(vals)), 4),
            "per_unit": [round(v, 4) for v in vals]}


class World(_W):
    NAME = "b_control"
    ARTIFACTS = ["report.json"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "credits"
    OPS = {"train": (_cost_train, _run_train,
                     "retrain with a subset of the release reverted; returns a scoreable checkpoint id"),
           "score": (_cost_score, _run_score,
                     "run an archived or freshly trained checkpoint through the eval harness")}

    def public_spec(self):
        return {"ops": {
            "train": {"args": {"revert": "list of change ids to roll back in the TRAINING pipeline"},
                      "cost": "%g credits" % TRAIN_COST,
                      "returns": "a checkpoint id; deterministic given the reversion set"},
            "score": {"args": {"ckpt": "'release', 'previous', or an id from `train`",
                               "harness": "list of change ids to roll back in the EVAL HARNESS",
                               "n": "eval units, 1..%d" % MAX_N},
                      "cost": "n x %g credits" % SCORE_UNIT,
                      "returns": "mean reported score over n units, and the per-unit values"}},
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
        ho = rep.get("harness_only")
        if not isinstance(ho, list) or any(h not in CHANGES for h in ho) or len(set(ho)) != len(ho):
            ok0 = False; why.append("harness_only must be a list of distinct change ids (may be empty)")
        items["R0_report"] = (ok0, "; ".join(why) or "report.json is well-formed")
        if not ok0:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()}}

        gain = {h: -p["e"][h] for h in CHANGES}
        items["R1_culprit"] = (
            cul == t["culprit"],
            "reverting `%s` raises true quality by %+.3f; the best single reversion is `%s` at %+.3f.  "
            "Full table (true quality gain from reverting each alone): %s"
            % (cul, gain[cul], t["culprit"], gain[t["culprit"]],
               ", ".join("%s %+.3f" % (h, gain[h]) for h in CHANGES)))

        # R2 is scored against the change the agent itself named, exactly as A-STACK scores the interval
        # against the set the agent itself shipped: naming the wrong culprit must not also break the
        # interval item, or one mistake would be punished twice and the items would stop being separable.
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
                "the true quality gain from reverting YOUR named change `%s` is %+.3f; you reported "
                "[%+.3f, %+.3f] (%s, width %.3f, cap %.2f).  The coupled leave-one-out design would have "
                "estimated %+.3f for it." % (cul, gain[cul], lo, hi, "covers" if cov else "misses",
                                             hi - lo, WIDTH_CAP, t["coupled_estimate"][cul]))

        got, want = sorted(ho), t["harness_only"]
        items["R3_harness_only"] = (
            got == want,
            "changes with |quality effect| <= %.2f AND |reported-number offset| >= %.2f: %s.  You listed: %s.  "
            "Per change (quality effect, harness offset): %s"
            % (EPS_E, DELTA_R, want or "(none)", got or "(none)",
               ", ".join("%s (%+.3f, %+.3f)" % (h, p["e"][h], p["r"][h]) for h in CHANGES)))

        diag = {"named": cul, "true_culprit": t["culprit"], "coupled_culprit": t["coupled_culprit"],
                "rescore_culprit": t["rescore_culprit"],
                "named_the_coupled_answer": cul == t["coupled_culprit"],
                "named_the_rescore_answer": cul == t["rescore_culprit"]}
        if ledger is not None:
            tr = [r for r in ledger if r["op"] == "train"]
            sc = [r for r in ledger if r["op"] == "score"]
            hs = set()
            for r in sc:
                hs.add(tuple(sorted(r["args"].get("harness") or [])))
            diag.update({"n_train": len(tr), "n_score": len(sc), "distinct_harness_configs": len(hs),
                         "spent": round(sum(r["cost"] for r in ledger), 2),
                         # the single most diagnostic process fact: did the agent ever compare two
                         # checkpoints under one fixed harness configuration?
                         "scored_under_one_config": sorted(hs)[:6]})
        n_ok = sum(1 for v in items.values() if v[0])
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(n_ok / 4.0, 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": t}


# ------------------------------------------------------------------ strategies
def _write(art_dir, culprit, lo, hi, harness_only):
    json.dump({"culprit": culprit, "effect": {"lo": round(lo, 4), "hi": round(hi, 4)},
               "harness_only": list(harness_only)},
              open(os.path.join(art_dir, "report.json"), "w"))


def _score(sess, ck, harness, n=4):
    return sess.call("score", {"ckpt": ck, "harness": list(harness), "n": int(n)})["score"]


def _train(sess, rev):
    return sess.call("train", {"revert": list(rev)})["ckpt"]


def _half(sig, n1, n2, k=2.8):
    return k * sig * math.sqrt(1.0 / n1 + 1.0 / n2)


def strat_oracle(sess, art_dir, rng):
    """Generic protocol, no instance knowledge.

        ruler arm   : score the ARCHIVED release under each single-change harness, 4 units   -> 5x4 = 20
                      plus the same archive under the untouched harness, 6 units              ->        6
        quality arm : retrain each single reversion and score every one of them under ONE
                      fixed harness configuration, 6 units                                   -> 5x8 + 5x6 = 70
        refine      : re-score the winner and the release at 8 units under that same config   ->       16
                                                                                                  ----- 112 / 120

    The unit counts are not free parameters: R3 asks for a classification against a stated threshold, so
    the standard error of each per-change quality estimate has to be small compared with that threshold.
    At 6 units a side the standard error is ~0.58*sigma, and the gate requires every true effect to sit at
    least 2.5x the threshold away from it or at most half of it, so the classification is ~3 sigma safe.

    The whole content of the protocol is the phrase "under ONE fixed harness configuration": the offset
    then cancels in every difference, so the quality arm needs no knowledge of the ruler at all.  The ruler
    arm is needed only for R3 - classifying which changes move the reported number without moving the model.
    """
    p = sess.w.p
    FIX = []                                   # any single configuration works, as long as it never moves
    base_rel = _score(sess, "release", FIX, 6)
    r_hat = {}
    for h in CHANGES:                          # archived weights, so this isolates the harness offset
        r_hat[h] = base_rel - _score(sess, "release", [h], 4)
    q_hat = {}
    for h in CHANGES:
        if sess.left() < TRAIN_COST + 6 * SCORE_UNIT + 16 * SCORE_UNIT:
            break
        ck = _train(sess, [h])
        q_hat[h] = _score(sess, ck, FIX, 6) - base_rel
    cul = max(q_hat, key=lambda h: q_hat[h]) if q_hat else CHANGES[0]
    n = 8 if sess.left() >= 16 * SCORE_UNIT else 4
    a = _score(sess, _ckpt_id([cul]), FIX, n)
    b = _score(sess, "release", FIX, n)
    eff = a - b
    half = _half(p["sig"], n, n)
    ho = sorted(h for h in CHANGES if abs(q_hat.get(h, 0.0)) <= EPS_E and abs(r_hat[h]) >= DELTA_R)
    _write(art_dir, cul, eff - half, eff + half, ho)


def strat_coupled_loo(sess, art_dir, rng):
    """P3 ABLATION: revert change i everywhere - training AND harness - and re-measure.  This is the design
    a careful engineer writes down first; every number in it is reproducible."""
    p = sess.w.p
    base = _score(sess, "release", [], 6)
    d = {}
    for h in CHANGES:
        ck = _train(sess, [h])
        d[h] = _score(sess, ck, [h], 6) - base          # = -(e_h + r_h)
    cul = max(d, key=lambda h: d[h])
    half = _half(p["sig"], 6, 6)
    _write(art_dir, cul, d[cul] - half, d[cul] + half, [])


def strat_rescore_only(sess, art_dir, rng):
    """P4 ABLATION: never retrain.  Roll each change back in the harness, re-score the archived release,
    and read the difference as that change's effect.  Cheap, tight, and measuring the ruler."""
    p = sess.w.p
    base = _score(sess, "release", [], 8)
    d = {h: _score(sess, "release", [h], 8) - base for h in CHANGES}
    cul = max(d, key=lambda h: d[h])
    half = _half(p["sig"], 8, 8)
    _write(art_dir, cul, d[cul] - half, d[cul] + half, [])


def strat_right_culprit_coupled_effect(sess, art_dir, rng):
    """Isolates R2: the true culprit, the true harness-only set, but the effect size taken from the coupled
    measurement instead of a fixed-harness contrast."""
    p = sess.w.p
    t = truth(p)
    v = t["coupled_estimate"][t["culprit"]]
    _write(art_dir, t["culprit"], v - 0.1, v + 0.1, t["harness_only"])


def strat_harness_only_empty(sess, art_dir, rng):
    t = truth(sess.w.p)
    g = t["culprit_gain"]
    _write(art_dir, t["culprit"], g - 0.1, g + 0.1, [])


def strat_harness_only_all_movers(sess, art_dir, rng):
    """Isolates R3 from the other side: lists every change whose reported number moves at all, ignoring the
    stated threshold and ignoring whether the model moved too."""
    p = sess.w.p
    t = truth(p)
    g = t["culprit_gain"]
    _write(art_dir, t["culprit"], g - 0.1, g + 0.1,
           sorted(h for h in CHANGES if abs(p["r"][h]) >= 0.10))


def strat_wide(sess, art_dir, rng):
    t = truth(sess.w.p)
    g = t["culprit_gain"]
    _write(art_dir, t["culprit"], g - 1.2, g + 1.2, t["harness_only"])


def strat_nop(sess, art_dir, rng):
    pass


def strat_bad_json(sess, art_dir, rng):
    open(os.path.join(art_dir, "report.json"), "w").write("{nope")


STRATEGIES = {"oracle": (strat_oracle, "pass"),
              "coupled_loo": (strat_coupled_loo, "fail"),
              "rescore_only": (strat_rescore_only, "fail"),
              "right_culprit_coupled_effect": (strat_right_culprit_coupled_effect, "fail"),
              "harness_only_empty": (strat_harness_only_empty, "fail"),
              "harness_only_all_movers": (strat_harness_only_all_movers, "fail"),
              "wide": (strat_wide, "fail"),
              "nop": (strat_nop, "fail"),
              "bad_json": (strat_bad_json, "fail")}
NOISY_FAIL = ()

# SEARCH: the exhaustive sweeps of the answer space, done with the most obvious measurement.  Both of these
# ARE exhaustive - they try all five single reversions, which is the entire answer space for R1 - and both
# fit the budget comfortably.  They still fail, because the design of each arm estimates the wrong quantity.
# That is the whole reason this family survives the search gate while a_stack did not: here the cheap
# measurement is BIASED, so sweeping it harder does not help.
SEARCH = ["coupled_loo", "rescore_only"]

# Each principle names the strategy that ablates ONLY it and the rubric item that ablation must break.
# v8gates.p_ablation enforces both halves; declaring the item is what turns "the decoy failed" into
# "the decoy failed because it lacks this principle" (v7 gate 9).  Between them the four ablations cover
# every graded item, which is also why no item can be inert.
PRINCIPLES = {"P3_hold_the_ruler_fixed":           ("coupled_loo", ["R1_culprit"]),
              "P3b_effect_needs_a_fixed_ruler":    ("right_culprit_coupled_effect", ["R2_effect"]),
              "P4_model_side_needs_retraining":    ("rescore_only", ["R1_culprit"]),
              "P5_harness_only_is_two_sided":      ("harness_only_all_movers", ["R3_harness_only"])}


# ------------------------------------------------------------------ grader falsification
# Each entry perturbs a PASSING oracle report.  `small` must stay a pass (the grader is not hair-trigger -
# the v6 bugs were graders that rejected correct answers); `large` must fail, on the named item.
def _load(art_dir):
    return json.load(open(os.path.join(art_dir, "report.json")))


def _save(art_dir, rep):
    json.dump(rep, open(os.path.join(art_dir, "report.json"), "w"))


def _mut_culprit(p, t, art_dir, leg):
    rep = _load(art_dir)
    if leg == "small":                      # semantically neutral: reorder the set, add an ignored key
        rep["harness_only"] = list(reversed(rep.get("harness_only", [])))
        rep["notes"] = "reordered; nothing else in the file is read"
    else:
        other = [h for h in CHANGES if h != t["culprit"]]
        rep["culprit"] = other[0]
        g = t["revert_gain"][other[0]]      # keep R2 self-consistent so ONLY R1 can break
        rep["effect"] = {"lo": g - 0.4 * WIDTH_CAP, "hi": g + 0.4 * WIDTH_CAP}
    _save(art_dir, rep)
    return None if leg == "small" else "R1_culprit"


def _mut_effect_coverage(p, t, art_dir, leg):
    rep = _load(art_dir)
    g = t["revert_gain"][rep["culprit"]]
    d = 0.4 * WIDTH_CAP if leg == "small" else 0.4 * WIDTH_CAP + 3.0 * WIDTH_CAP
    rep["effect"] = {"lo": g + d - 0.4 * WIDTH_CAP, "hi": g + d + 0.4 * WIDTH_CAP}
    _save(art_dir, rep)
    return None if leg == "small" else "R2_effect"


def _mut_effect_width(p, t, art_dir, leg):
    rep = _load(art_dir)
    g = t["revert_gain"][rep["culprit"]]
    w = 0.90 * WIDTH_CAP if leg == "small" else 1.20 * WIDTH_CAP   # centred on the truth either way:
    rep["effect"] = {"lo": g - w / 2, "hi": g + w / 2}             # only the width cap can decide this
    _save(art_dir, rep)
    return None if leg == "small" else "R2_effect"


def _mut_harness_only(p, t, art_dir, leg):
    rep = _load(art_dir)
    if leg == "small":
        rep["harness_only"] = list(reversed(sorted(rep["harness_only"])))   # order must be ignored
    else:
        rep["harness_only"] = sorted(rep["harness_only"])[1:]               # drop one true member
    _save(art_dir, rep)
    return None if leg == "small" else "R3_harness_only"


MUTATE = [("culprit_label", _mut_culprit),
          ("effect_coverage", _mut_effect_coverage),
          ("effect_width", _mut_effect_width),
          ("harness_only_set", _mut_harness_only)]


def instance_gate(p):
    t = truth(p)
    e, r = p["e"], p["r"]
    gain = t["revert_gain"]
    # Decidability of R3: no change may sit near either threshold, or the classification would be a
    # coin-flip on measurement noise rather than on understanding.  (v7 gate: the tolerance has to be
    # measured off the mechanism, not asserted.)
    decidable = all((abs(e[h]) <= 0.5 * EPS_E or abs(e[h]) >= 2.5 * EPS_E) and
                    (abs(r[h]) <= 0.6 * DELTA_R or abs(r[h]) >= 1.6 * DELTA_R) for h in CHANGES)
    small_mover = any(0.12 <= abs(r[h]) < DELTA_R for h in CHANGES)
    c = [t["margin"] >= MARGIN,                                     # 1 the culprit is decidable
         t["culprit_gain"] >= 1.20,                                 # 2 the fix is worth shipping
         t["total_regression"] < -0.30,                             # 3 there really was a regression
         t["coupled_culprit"] != t["culprit"],                      # 4 P3 bites on R1
         gain[t["coupled_culprit"]] <= t["culprit_gain"] - MARGIN,  # 5 ... and not by a hair
         abs(r[t["coupled_culprit"]]) >= 1.5 * WIDTH_CAP,           # 6 P3 also bites on R2
         t["rescore_culprit"] != t["culprit"],                      # 7 P4 bites on R1
         len(t["harness_only"]) >= 1,                               # 8 R3 is not vacuous
         len(t["harness_only"]) < len(CHANGES),                     # 9 ... and not everything
         small_mover,                                               # 10 "anything that moves" != R3 answer
         abs(r[t["culprit"]]) > 0.5 * WIDTH_CAP,                    # 11 no cap-compliant interval centred on
         #      the coupled estimate of the culprit can cover the truth, so widening is not an escape
         t["culprit_gain"] - abs(r[t["culprit"]]) > 0.0,            # 12 the culprit is still a real win
         decidable,                                                 # 13 no change sits on a threshold
         EPS_E - 0.5 * EPS_E >= 3.0 * p["sig"] * math.sqrt(1.0 / 6 + 1.0 / 6),   # 14 R3's quality threshold
         DELTA_R - 0.6 * DELTA_R >= 3.0 * p["sig"] * math.sqrt(1.0 / 4 + 1.0 / 6)]  # 15 ... and its ruler
         #      threshold are both >= 3 sigma away from the nearest true value AT THE UNIT COUNTS THE
         #      ORACLE CAN AFFORD.  Without these two the item would be decided by measurement noise.
    info = dict(t)
    info.update({"c": [bool(x) for x in c], "decidable": bool(decidable),
                 "small_mover": bool(small_mover)})
    return bool(all(c)), info


def pool_gate(seeds):
    """Pool-level guessing gate.  A-STACK can use the v7 form (`no single fixed artifact satisfies R1 on
    two instances`) because its answer is a subset out of ~30.  Here R1 is one of five ids, so that form is
    unsatisfiable for any pool larger than five and would be a fake gate.  The honest version for a
    discrete-answer family is a MODAL-ANSWER gate: no single id may be the culprit in more than a quarter
    of the pool, and no single harness-only set in more than a third, so a fixed guess earns no more than
    chance."""
    ps = [sample_params(s) for s in seeds]
    ps = [p for p in ps if instance_gate(p)[0]]
    n = len(ps)
    if n < 4:
        return True, {"n": n}
    culs, hos = {}, {}
    for p in ps:
        t = truth(p)
        culs[t["culprit"]] = culs.get(t["culprit"], 0) + 1
        k = ",".join(t["harness_only"])
        hos[k] = hos.get(k, 0) + 1
    mc, mh = max(culs.values()), max(hos.values())
    ok = mc <= math.ceil(n / 4.0) and mh <= math.ceil(n / 3.0)
    return bool(ok), {"n": n, "modal_culprit": mc, "cap_culprit": math.ceil(n / 4.0),
                      "modal_harness_only": mh, "cap_harness_only": math.ceil(n / 3.0),
                      "culprits": culs, "harness_only_sets": hos}


def instruction(p, t):
    ids = ", ".join("`%s`" % h for h in CHANGES)
    return """# Release regression: find the change that actually cost you quality

Last week's runtime release bundled five changes, %s.  They landed together because they share code: some
of them touch modules used by **both** the training pipeline and the evaluation harness.

The reported eval score dropped from **%.2f** (checkpoint `previous`) to **%.2f** (checkpoint `release`).
Leadership wants one change reverted - the one that will buy back the most real quality - and they want a
number for how much it buys.

You have **%g credits**.

* `lab train revert=<ids>` - retrain with those changes rolled back **in the training pipeline**.
  Returns a checkpoint id.  **%g credits.**  Training is deterministic given the reversion set: the same
  reversion always produces the same weights, so repeating a training run tells you nothing new.
* `lab score ckpt=<id> harness=<ids> n=<1..%d> - run a checkpoint through the eval harness with those
  changes rolled back **in the harness**.  **n credits.**  `ckpt` may be `release`, `previous`, or any id
  `train` returned.  The two arguments are independent: you choose the weights and you choose the harness.

Each eval unit carries **+-%.2f points** of independent measurement noise (mean zero).  There is no other
randomness anywhere in this task.

## What the score is made of

The number the harness reports is

    reported = quality(weights) + offset(harness configuration)

`quality` is a property of the weights alone; `offset` is a property of the harness configuration alone.
Both are additive in the five changes.  Neither term is observable on its own - the harness only ever
reports their sum.

## Deliverable: `/app/report.json`

```json
{"culprit": "%s",
 "effect": {"lo": 0.00, "hi": 0.00},
 "harness_only": ["%s"]}
```

* `culprit` - the single change whose reversion raises **quality** the most.
* `effect` - an interval for the gain in **quality** from reverting exactly the change you named, in score
  points.  Width may not exceed **%.2f points**.
* `harness_only` - every change whose |quality effect| is at most **%.2f points** while its
  |offset effect| is at least **%.2f points**: changes that move the reported number without moving the
  model.  The list may be empty.  No change in this release sits near either threshold.
""" % (ids, t["q_previous"] + p["r0"] + 0.0, t["q_release"] + p["r0"] + sum(p["r"].values()),
       BUDGET, TRAIN_COST, MAX_N, p["sig"], CHANGES[0], CHANGES[1], WIDTH_CAP, EPS_E, DELTA_R)


def docs(p):
    return {"docs/report_format.md": """# /app/report.json

```json
{"culprit": "H-2",
 "effect": {"lo": 1.31, "hi": 1.58},
 "harness_only": ["H-4"]}
```

* `culprit` (string) - one change id.
* `effect.lo`, `effect.hi` (numbers) - score points of **quality** gained by reverting the change you
  named in `culprit`.  Graded against that change, not against the correct one, so a wrong culprit costs
  you one item and not two.  `hi - lo` must be <= %.2f.
* `harness_only` (list of strings, possibly empty) - exact set match; order is ignored.

Nothing else in the file is read.  There is no partial credit inside an item.
""" % WIDTH_CAP,
            "docs/pipeline.md": """# How a release is built and measured

The training pipeline and the evaluation harness are configured from the same manifest.  A change entry in
the manifest can carry a training-side patch, a harness-side patch, or both; the manifest does not say
which, and the vendor does not publish per-change numbers.

Two properties of the setup are worth writing down before spending credits:

* **Weights are a function of the training-side configuration only.**  `train revert=<X>` fixes the
  weights; nothing you do at scoring time changes them.  Conversely, re-scoring an archived checkpoint
  under a different harness configuration cannot change that checkpoint's quality.
* **`offset` is a function of the harness configuration only.**  Two different checkpoints scored under
  the same harness configuration carry the same offset.

Archived from the previous cycle and free to re-score as often as you like (they are already trained):
`previous`, the pre-release checkpoint, and `release`, the current one.
"""}


def hints(p):
    return {1: """
Hint 1: the number the harness reports for a reverted build differs from the number it reports for the
release for two reasons at once.  Only one of them is the thing you were asked about.
""", 2: """
Hint 2: `offset` depends only on the harness configuration, so it cancels out of any comparison made
under a single fixed harness configuration - and an archived checkpoint's quality is fixed, so scoring one
under two configurations measures nothing but the offset.
"""}
