"""A-SPAN: optimise for a workload you are not allowed to measure.

Difficulty mechanism (this family's *signature*):
  Q1  THE TARGET IS OUTSIDE THE MEASURABLE SET.  The deliverable is about production traffic.  The canary
      admits at most ONE queued change per rollout window, so no candidate *set* can ever be benchmarked on
      the target - not once, not at any price.  Exhaustive search over the answer space is not expensive
      here, it is structurally impossible.  (This family exists because its predecessor `a_stack` was
      killed by exactly that search: see the header of a_stack.py.)
  Q2  THE CHEAP MEASUREMENT IS BIASED, AND SO IS EVERY OBVIOUS REPAIR.  Five trace replays are cheap and
      admit any set, but each replays one request family in isolation, so ranking on any one of them - or
      on their unweighted average, or by how many of them prefer a set, or by worst case over all mixes -
      picks the wrong set.  Sweeping the whole answer space on the cheap instrument is affordable and still
      wrong, which is what separates this family from a_stack: there the naive measurement of a candidate
      was unbiased, so sweeping it won.
  Q3  THE BRIDGE IS LINEAR, THE COMPOSITION IS NOT.  Mean latency is linear in the TRAFFIC MIX (production
      is a mixture of the same five families the replays isolate), so the target is exactly reconstructible
      from the five replays once the mix is known - and the mix is not instrumented, so it has to be
      estimated from the handful of single-change canary points that ARE allowed.  Meanwhile latency is NOT
      additive in the changes: two changes that relieve the same stage do not stack, so the single-change
      canary points cannot be composed into a set's effect directly.  An agent that tries to build the
      answer out of the canary points alone meets the non-additivity; one that tries to read it off the
      replays alone meets the mix bias.  Only the combination works.

Ground truth is exact and closed-form: latency is a sum over stages, every change multiplies the stages it
touches and adds a fixed fraction elsewhere, and a workload is a vector of stage weights.  So the optimum,
every set's true production speedup and the in-context ranking inside a set are computed, not sampled.

Distinct from B-CONTROL: nothing here is confounded, every measurement is unbiased for the thing it
measures.  What is missing is a measurement of the target at all.
"""
import itertools, json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

NSTAGE = 5
CHANGES = ["C-114", "C-127", "C-203", "C-241", "C-318", "C-352"]
TRACES = ["R1", "R2", "R3", "R4", "R5"]
BUDGET = 320.0
TRACE_COST = 1.0            # per repetition, any set
CANARY_COST = 4.0           # per repetition, at most CANARY_MAX_OPTS changes applied
CANARY_MAX_OPTS = 1
MAX_REPS = 6
TAU_GAIN = 0.90             # fraction of the achievable production gain the shipped set must capture
WIDTH_CAP = 0.06            # relative width allowed for the predicted-speedup interval
WEAK_EPS = 0.015            # slack on the in-context weakest-member ranking, in speedup-gain units
CANARY_REPS_REF = 3         # repetitions the reference protocol spends per canary point
SWEEP_REPS = 1              # ... per (set, family) cell of the replay sweep
REFINE_REPS = 3             # ... per family when re-measuring the shipped set and the stock stack
N_CANARY_DESIGN = 7         # stock + six single changes: the entire measurable surface of the target


# ------------------------------------------------------------------ instance
def sample_params(seed):
    """Structure is redrawn per seed; nothing about a change id or a trace id carries over between instances.

    Construction, and why it is this way (the first two drafts of this function both failed the screen at a
    yield of 0/1199, which is what the screen is for):

    * The five request families are drawn SPIKY - each one dwells mostly in a different stage of the stack -
      and production is their mixture under a traffic share `pi` that is required to be spread over at least
      three families.  Production therefore has two or three comparable heavy stages and looks like NONE of
      the five families.  The backward construction tried first (draw production skewed, tilt the families
      off it) made production's heaviest stage dominate every workload, so the family spiked at that stage
      agreed with production's optimum 95% of the time and every decoy was right for the wrong reason.
    * Mean latency is linear in the stage profile, so `L(S, prod) = sum_k pi_k L(S, T_k)` holds identically
      for every set S.  Condition 5 of the screen re-verifies that over all 64 subsets rather than trusting
      the algebra.
    * The six changes follow a fixed role template over the stages RANKED BY PRODUCTION WEIGHT: two divide
      the same top stage (substitutes - their effects do not stack), one divides the second stage but taxes
      the top one (worthless alone, best buy once the top stage has been relieved - synergy), the rest spread
      out.  Roles are shuffled onto change ids so no id means anything across instances.  Both mechanisms are
      re-checked numerically below; the template only makes them likely.
    """
    g = np.random.default_rng(71000 + seed)
    spike = [int(x) for x in g.permutation(NSTAGE)]
    b = {}
    for k, T in enumerate(TRACES):
        sh = np.exp(g.normal(0.0, 1.1, NSTAGE))
        sh[spike[k]] *= float(g.uniform(4.0, 10.0))
        sh = sh / sh.sum() * float(g.uniform(600, 1400))
        b[T] = [float(x) for x in sh]
    # A provisional mix, used only to rank the stages so the role template can be laid out.  The mix that
    # ships is chosen at the end of this function.
    pi0 = np.ones(len(TRACES)) / len(TRACES)
    h = list(np.argsort(-np.array([sum(pi0[k] * b[T][r] for k, T in enumerate(TRACES))
                                   for r in range(NSTAGE)])))

    def mk(div, tax=None):
        m, v = [1.0] * NSTAGE, [0.0] * NSTAGE
        for r, val in div:
            m[h[r]] = float(val)
        if tax:
            v[h[tax[0]]] = float(tax[1])
        return [round(x, 5) for x in m], [round(x, 5) for x in v]

    # Every role interacts with at least one other role, which is what makes a change's value depend on what
    # ships alongside it.  Weak interactions were the second yield killer: with a small tax the ordering of
    # the members of the optimal set by SOLO effect equalled their ordering by IN-CONTEXT contribution, so the
    # decoy that reads R3 off solo effects was accidentally right and the instance had to be thrown away.
    roles = [mk([(0, g.uniform(0.30, 0.45))]),                                    # heavy hitter on h0
             mk([(0, g.uniform(0.35, 0.50)), (2, g.uniform(0.75, 0.92))]),        # substitute on h0
             mk([(1, g.uniform(0.25, 0.40))], (0, g.uniform(0.18, 0.35))),        # synergy: taxes h0 hard
             mk([(2, g.uniform(0.35, 0.55))], (1, g.uniform(0.06, 0.16))),        # taxes h1
             mk([(1, g.uniform(0.50, 0.70)), (3, g.uniform(0.60, 0.85))]),        # substitute on h1
             mk([(3, g.uniform(0.30, 0.50))], (2, g.uniform(0.06, 0.16)))]        # taxes h2
    # Review cost is correlated with how big the change is, which is both realistic and load-bearing: the two
    # strong h0 changes are the expensive ones, so a greedy that ranks by solo effect and ignores overlap
    # spends its whole budget on two changes that substitute for each other.
    rc = [int(g.integers(4, 6)), int(g.integers(4, 6)), int(g.integers(3, 5)),
          int(g.integers(3, 5)), int(g.integers(4, 6)), int(g.integers(3, 5))]
    perm = [int(x) for x in g.permutation(len(CHANGES))]
    mult, over, cost = {}, {}, {}
    for j, c in enumerate(CHANGES):
        mult[c], over[c] = roles[perm[j]]
        cost[c] = rc[perm[j]]
    base = {"b": b, "mult": mult, "over": over, "cost": cost, "pi": [float(x) for x in pi0],
            "sig": round(float(g.uniform(0.0015, 0.003)), 5)}
    # Pick the review-day cap that puts the number of feasible sets nearest the middle of the admissible band
    # (the band is a budget constraint: the reference protocol replays every feasible set on all five
    # families).  Deterministic given the draw, so nothing here depends on the search below.
    caps = sorted(range(11, 17), key=lambda k: abs(int(_tables(dict(base, cap=k))["feas"].sum()) - 26))
    base["cap"] = caps[0]
    tb = _tables(base)
    # Search the hidden traffic mix.  The five family shapes, the six changes and the review costs fix which
    # sets are candidates; `pi` alone fixes which one wins, and none of the shortcuts' own answers depend on
    # it.  So the mix is the free variable that turns "hope every shortcut is wrong" into "choose a mix for
    # which every shortcut is wrong".  Difficulty conditions only - solvability and the correctness invariant
    # are re-checked independently by `instance_gate`, which is the authority.
    best, best_n = None, -1
    for _ in range(2500):
        pi = g.dirichlet(np.ones(len(TRACES)) * 1.4)
        if pi.max() > 0.40 or int((pi >= 0.12).sum()) < 3:
            continue
        cand = dict(base, pi=[float(x) for x in pi])
        c, _info = _conditions(cand, tb=tb)
        if all(c):
            return cand
        if sum(c) > best_n:
            best, best_n = cand, sum(c)
    return best or dict(base, pi=[float(x) for x in g.dirichlet(np.ones(len(TRACES)) * 1.4)])


# ------------------------------------------------------------------ hidden mechanics
def factors(p, S):
    """Per-stage factor of an option set: the product of its members' multipliers on that stage, inflated by
    their overheads on that stage.

        f_r(S) = (prod_{o in S} m[o][r]) * (1 + sum_{o in S} v[o][r])

    Both interactions live here.  Two members that divide the same stage MULTIPLY, so the second one's
    marginal saving is a fraction of the first's - substitution.  An overhead is a percentage of the stage's
    REMAINING time, so a change that taxes a stage is cheap once another change has divided that stage -
    synergy.  The first draft of this file had the overhead additive on the stock stage time, which made it
    context-free and quietly removed the synergy mechanism the whole family was built around.
    """
    f = []
    for r in range(NSTAGE):
        prod, add = 1.0, 0.0
        for o in S:
            prod *= p["mult"][o][r]
            add += p["over"][o][r]
        f.append(prod * (1.0 + add))
    return f


def mix_prod(p):
    return [sum(p["pi"][k] * p["b"][T][r] for k, T in enumerate(TRACES)) for r in range(NSTAGE)]


def latency_mix(p, S, bvec):
    f = factors(p, S)
    return sum(bvec[r] * f[r] for r in range(NSTAGE))


def latency(p, S, wl):
    return latency_mix(p, S, mix_prod(p) if wl == "prod" else p["b"][wl])


def speedup(p, S, wl):
    return latency(p, [], wl) / max(1e-9, latency(p, S, wl))


def feasible_sets(p):
    out = []
    for k in range(len(CHANGES) + 1):
        for S in itertools.combinations(CHANGES, k):
            if sum(p["cost"][o] for o in S) <= p["cap"]:
                out.append(list(S))
    return out


def best_set(p, wl):
    F = [S for S in feasible_sets(p) if S]
    S = min(F, key=lambda X: latency(p, X, wl))
    return S, speedup(p, S, wl)


def weakest_member(p, S, wl):
    """The member whose removal costs the least speedup gain - in context, inside the set the agent ships."""
    if not S:
        return None, {}
    g0 = speedup(p, S, wl) - 1.0
    rem = {o: g0 - (speedup(p, [x for x in S if x != o], wl) - 1.0) for o in S}
    return min(rem, key=lambda k: rem[k]), rem


def canary_design(p):
    """The entire measurable surface of the target workload: the stock stack and each change alone."""
    return [[]] + [[c] for c in CHANGES]


def truth(p):
    S, sp = best_set(p, "prod")
    w, rem = weakest_member(p, S, "prod")
    rv = sorted(rem.values())
    F = [X for X in feasible_sets(p) if X]
    g = sp - 1.0
    ratio = {tuple(X): (speedup(p, X, "prod") - 1.0) / max(1e-9, g) for X in F}
    qual = [list(k) for k, v in ratio.items() if v >= TAU_GAIN]
    rs = sorted(ratio.values(), reverse=True)
    return {"best_set": S, "best_speedup": round(sp, 5), "n_feasible": len(F),
            "n_qualifying": len(qual), "ratio_gap_at_tau": round(rs[len(qual) - 1] - rs[len(qual)], 4)
            if 0 < len(qual) < len(rs) else 9.9,
            "weakest": w, "weak_margin": round((rv[1] - rv[0]) if len(rv) > 1 else 9.9, 4),
            "pi": p["pi"], "max_pi": round(max(p["pi"]), 4),
            "base_prod": round(latency(p, [], "prod"), 3)}


# ------------------------------------------------------------------ ops
def _parse_opts(a, key="opts"):
    S = a.get(key, [])
    if S is None:
        S = []
    if isinstance(S, str):
        S = [x.strip() for x in S.split(",") if x.strip()]
    if not isinstance(S, list):
        raise LabError("%s must be a list of change ids (or a comma-separated string)" % key)
    S = [str(x) for x in S]
    for o in S:
        if o not in CHANGES:
            raise LabError("unknown change id %r; the queue is %s" % (o, ", ".join(CHANGES)))
    if len(set(S)) != len(S):
        raise LabError("%s contains duplicates" % key)
    return sorted(set(S))


def _reps(a):
    return int(num(a.get("reps", 1), "reps", 1, MAX_REPS, integer=True))


def _cost_replay(w, a):
    _parse_opts(a)
    if a.get("trace") not in TRACES:
        raise LabError("trace must be one of %s" % ", ".join(TRACES))
    return _reps(a) * TRACE_COST


def _run_replay(w, a, ctx):
    S, T, reps = _parse_opts(a), a["trace"], _reps(a)
    L = latency(w.p, S, T)
    g = rng_for(w.salt, "replay", T, ",".join(S), ctx["i"])
    vals = [L * (1.0 + float(g.normal(0, w.p["sig"]))) for _ in range(reps)]
    return {"trace": T, "opts": S, "reps": reps, "latency_ms": round(float(np.mean(vals)), 4),
            "per_rep_ms": [round(v, 4) for v in vals],
            "review_cost_of_set": sum(w.p["cost"][o] for o in S)}


def _cost_canary(w, a):
    S = _parse_opts(a)
    if len(S) > CANARY_MAX_OPTS:
        raise LabError("the canary rollout gate admits at most %d change(s) per window; you asked for %d (%s).  "
                       "Sets larger than that cannot be measured on production traffic at any price."
                       % (CANARY_MAX_OPTS, len(S), ", ".join(S)))
    return _reps(a) * CANARY_COST


def _run_canary(w, a, ctx):
    S, reps = _parse_opts(a), _reps(a)
    L = latency(w.p, S, "prod")
    g = rng_for(w.salt, "canary", ",".join(S), ctx["i"])
    vals = [L * (1.0 + float(g.normal(0, w.p["sig"]))) for _ in range(reps)]
    return {"opts": S, "reps": reps, "latency_ms": round(float(np.mean(vals)), 4),
            "per_rep_ms": [round(v, 4) for v in vals],
            "review_cost_of_set": sum(w.p["cost"][o] for o in S)}


class World(_W):
    NAME = "a_span"
    ARTIFACTS = ["plan.json"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "lab credits"
    OPS = {"replay": (_cost_replay, _run_replay,
                      "replay one request family through the stack with a set of changes applied"),
           "canary": (_cost_canary, _run_canary,
                      "serve live production traffic with at most %d change(s) applied" % CANARY_MAX_OPTS)}

    def public_spec(self):
        return {"ops": {
            "replay": {"args": {"opts": "list of queued change ids (empty = stock stack)",
                                "trace": "one of %s" % ", ".join(TRACES),
                                "reps": "independent repeats, 1..%d" % MAX_REPS},
                       "cost": "reps x %g credits" % TRACE_COST,
                       "returns": "mean latency in ms over the repeats, and the per-repeat values"},
            "canary": {"args": {"opts": "at most %d change id(s)" % CANARY_MAX_OPTS,
                                "reps": "independent repeats, 1..%d" % MAX_REPS},
                       "cost": "reps x %g credits" % CANARY_COST,
                       "returns": "mean latency in ms on live production traffic"}},
            "queued_changes": {c: "%d review-days" % self.p["cost"][c] for c in CHANGES},
            "review_day_cap": "%d review-days" % self.p["cap"],
            "measurement_noise": "+-%.2f%% per measurement, independent, mean zero" % (100 * self.p["sig"]),
            "deliverable": "/app/plan.json (see /app/docs/plan_format.md)"}

    def grade(self, art_dir, ledger=None):
        p, t = self.p, truth(self.p)
        items = {}
        try:
            rep = json.load(open(os.path.join(art_dir, "plan.json")))
        except Exception as e:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {"R0_plan": {"ok": False,
                                          "detail": "plan.json missing or unreadable: %s" % type(e).__name__}}}
        why, ok0, S = [], True, rep.get("opts")
        if not isinstance(S, list) or any(o not in CHANGES for o in S) or len(set(S)) != len(S) or not S:
            ok0 = False
            why.append("opts must be a non-empty list of distinct queued change ids, got %r" % (S,))
            S = []
        else:
            S = sorted(set(S))
            if sum(p["cost"][o] for o in S) > p["cap"]:
                ok0 = False
                why.append("review cost %d exceeds the cap %d"
                           % (sum(p["cost"][o] for o in S), p["cap"]))
        iv = rep.get("speedup")
        if not (isinstance(iv, dict) and "lo" in iv and "hi" in iv):
            ok0 = False
            why.append("speedup must be an object with numeric lo/hi")
        if rep.get("weakest") is not None and rep.get("weakest") not in CHANGES:
            ok0 = False
            why.append("weakest must be a change id")
        items["R0_plan"] = (ok0, "; ".join(why) or "plan.json is well-formed and within the review-day cap")
        if not ok0:
            return {"pass": False, "score": 0.0, "diag": {"declared": S}, "ref": t,
                    "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()}}

        sp = speedup(p, S, "prod")
        ratio = (sp - 1.0) / max(1e-9, t["best_speedup"] - 1.0)
        items["R1_set_is_near_optimal"] = (
            ratio >= TAU_GAIN,
            "your set reaches %.4fx on production traffic; the best feasible set reaches %.4fx -> you capture "
            "%.1f%% of the achievable gain, need %.0f%%.  Only %d of %d feasible sets clear that bar."
            % (sp, t["best_speedup"], 100 * ratio, 100 * TAU_GAIN, t["n_qualifying"], t["n_feasible"]))

        try:
            lo, hi = float(iv["lo"]), float(iv["hi"])
            bad = not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi)
        except Exception:
            lo = hi = float("nan")
            bad = True
        if bad:
            items["R2_speedup_predicted"] = (False, "malformed interval")
        else:
            wide = (hi - lo) / max(1e-9, sp) > WIDTH_CAP + 1e-9
            cov = lo <= sp <= hi
            items["R2_speedup_predicted"] = (
                bool(cov and not wide),
                "the true production speedup of YOUR set is %.4fx; you reported [%.4f, %.4f] (%s, relative "
                "width %.4f, cap %.3f).  Production traffic is %s over %s."
                % (sp, lo, hi, "covers" if cov else "misses", (hi - lo) / max(1e-9, sp), WIDTH_CAP,
                   ", ".join("%.3f" % x for x in p["pi"]), ", ".join(TRACES)))

        w_true, rem = weakest_member(p, S, "prod")
        wk = rep.get("weakest")
        if wk not in S:
            items["R3_weakest_member"] = (False, "weakest must name one of the changes you shipped")
        else:
            mn = min(rem.values())
            items["R3_weakest_member"] = (
                rem[wk] <= mn + WEAK_EPS,
                "removing `%s` from your own set costs %.4f of production speedup gain; the least-contributing "
                "member is `%s` at %.4f (slack %.3f).  In-context ranking inside your set: %s"
                % (wk, rem[wk], w_true, mn, WEAK_EPS,
                   ", ".join("%s %.4f" % (k, v) for k, v in sorted(rem.items(), key=lambda x: x[1]))))

        diag = {"declared": S, "true_best": t["best_set"], "sp": round(sp, 5), "gain_ratio": round(ratio, 4),
                "pi": p["pi"]}
        for T in TRACES:                       # did the agent ship some single trace's favourite?
            bs, _ = best_set(p, T)
            diag["is_best_on_" + T] = sorted(S) == sorted(bs)
        if ledger is not None:
            rp = [r for r in ledger if r["op"] == "replay"]
            cn = [r for r in ledger if r["op"] == "canary"]
            diag.update({"n_replay": len(rp), "n_canary": len(cn),
                         "distinct_replay_sets": len({tuple(r["args"].get("opts") or []) for r in rp}),
                         "traces_used": sorted({r["args"].get("trace") for r in rp}),
                         "canary_points": sorted({tuple(r["args"].get("opts") or []) for r in cn}),
                         "spent": round(sum(r["cost"] for r in ledger), 2)})
        n_ok = sum(1 for v in items.values() if v[0])
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(n_ok / 4.0, 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": t}


# ------------------------------------------------------------------ strategies
def _write(art_dir, S, lo, hi, weakest):
    json.dump({"opts": list(S), "speedup": {"lo": round(float(lo), 5), "hi": round(float(hi), 5)},
               "weakest": weakest}, open(os.path.join(art_dir, "plan.json"), "w"))


def _replay(sess, S, T, reps=1):
    return sess.call("replay", {"opts": list(S), "trace": T, "reps": int(reps)})["latency_ms"]


def _canary(sess, S, reps=1):
    return sess.call("canary", {"opts": list(S), "reps": int(reps)})["latency_ms"]


def _solve_mix(A, y, scale=None):
    """Least squares for the traffic mix subject to sum(pi) = 1, solved exactly through its KKT system.

    A[i, k] = measured latency of design i on family k; y[i] = measured latency of design i on production.
    No clipping: a small true share estimated slightly negative is still the unbiased linear estimate, and
    clipping-then-renormalising biases every prediction made from it (it was the first draft's estimator).
    `scale` is accepted for backwards compatibility and ignored.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    K = A.shape[1]
    s = float(np.mean(np.abs(y))) or 1.0
    A, y = A / s, y / s
    M = np.zeros((K + 1, K + 1))
    M[:K, :K] = 2.0 * A.T.dot(A)
    M[:K, K] = 1.0
    M[K, :K] = 1.0
    r = np.concatenate([2.0 * A.T.dot(y), [1.0]])
    return np.linalg.solve(M, r)[:K]


def strat_oracle(sess, art_dir, rng):
    """Reference protocol.  No instance knowledge; every number it uses is measured or published.

        replays  : every feasible set on all five families, 1 rep            5 x |F| <= 180 credits
        canary   : the stock stack and each single change, 3 reps              7 x 3 x 4 =  84 credits
        refine   : the chosen set and the stock stack on the families, 3 reps  2 x 5 x 3 =  30 credits

    Four moves: (1) mean latency is linear in the traffic mix, so production latency for ANY set is the same
    fixed combination of that set's five replay latencies; (2) the mix is identified by constrained least
    squares from the single-change canary points, the only production measurements that exist; (3) with the
    mix in hand every feasible set's production latency is a prediction, so the argmin and the in-context
    leave-one-out ranking are both computable; (4) the interval is a parametric bootstrap of the whole
    estimate - design rows, canary points and refined replays redrawn at the PUBLISHED per-measurement noise -
    because the design matrix is itself measured and the fit has almost no residual degrees of freedom.
    """
    p = sess.w.p
    sig = p["sig"]                                   # published in public_spec as measurement_noise
    F = [S for S in feasible_sets(p) if S]
    reserve = (N_CANARY_DESIGN * CANARY_REPS_REF * CANARY_COST + 2 * len(TRACES) * REFINE_REPS * TRACE_COST
               + len(TRACES) * TRACE_COST)
    rl = {}
    for S in F:
        if sess.left() < reserve + len(TRACES) * TRACE_COST:
            break
        rl[tuple(S)] = [_replay(sess, S, T, SWEEP_REPS) for T in TRACES]
    base = [_replay(sess, [], T, SWEEP_REPS) for T in TRACES]
    A, y = [], []
    for d in canary_design(p):
        row = base if not d else (rl.get(tuple(d)) or [_replay(sess, d, T, SWEEP_REPS) for T in TRACES])
        A.append(row)
        y.append(_canary(sess, d, CANARY_REPS_REF))
    A, y = np.array(A), np.array(y)
    pi = _solve_mix(A, y)
    pred = {k: float(np.dot(pi, v)) for k, v in rl.items()}
    b0 = float(np.dot(pi, base))
    win = list(min(pred, key=lambda k: pred[k]))
    # in-context leave-one-out on the PREDICTED target, which is the only place it can be done
    gain = {}
    for o in win:
        sub = tuple(sorted(x for x in win if x != o))
        Lsub = pred.get(sub)
        if Lsub is None:
            Lsub = float(np.dot(pi, [_replay(sess, list(sub), T, SWEEP_REPS) for T in TRACES]))
        gain[o] = (b0 / pred[tuple(win)] - 1.0) - (b0 / Lsub - 1.0)
    wk = min(gain, key=lambda o: gain[o]) if gain else win[0]
    rw = np.array([_replay(sess, win, T, REFINE_REPS) for T in TRACES])
    rb = np.array([_replay(sess, [], T, REFINE_REPS) for T in TRACES])
    sp = float(np.dot(pi, rb) / np.dot(pi, rw))
    g = np.random.default_rng(int(rng.integers(1 << 30)))
    boot = []
    for _ in range(400):
        Ab = A * (1.0 + g.normal(0.0, sig / math.sqrt(SWEEP_REPS), A.shape))
        yb = y * (1.0 + g.normal(0.0, sig / math.sqrt(CANARY_REPS_REF), y.shape))
        pb = _solve_mix(Ab, yb)
        rwb = rw * (1.0 + g.normal(0.0, sig / math.sqrt(REFINE_REPS), rw.shape))
        rbb = rb * (1.0 + g.normal(0.0, sig / math.sqrt(REFINE_REPS), rb.shape))
        boot.append(float(np.dot(pb, rbb) / np.dot(pb, rwb)))
    half = 2.5 * float(np.std(boot))
    _write(art_dir, win, sp - half, sp + half, wk)


def _rank_on_weights(sess, art_dir, wvec, reps=1, honest_interval=True):
    """Shared body for the decoys that differ only in which weighting of the traces they rank on."""
    p = sess.w.p
    F = [S for S in feasible_sets(p) if S]
    rl = {}
    for S in F:
        if sess.left() < 12 * TRACE_COST + 2 * CANARY_COST:
            break
        rl[tuple(S)] = [_replay(sess, S, T, reps) for T in TRACES]
    base = [_replay(sess, [], T, reps) for T in TRACES]
    w = np.array(wvec, dtype=float)
    w = w / w.sum()
    score = {k: float(np.dot(w, v)) for k, v in rl.items()}
    win = list(min(score, key=lambda k: score[k]))
    b0 = float(np.dot(w, base))
    gain = {}
    for o in win:
        sub = tuple(sorted(x for x in win if x != o))
        if sub in score:
            gain[o] = (b0 / score[tuple(win)] - 1.0) - (b0 / score[sub] - 1.0)
    wk = min(gain, key=lambda o: gain[o]) if gain else win[0]
    sp = b0 / score[tuple(win)]
    half = sp * 0.02
    _write(art_dir, win, sp - half, sp + half, wk)


def strat_rank_uniform(sess, art_dir, rng):
    """Q2 ABLATION: sweep the whole answer space on the cheap instrument and average the five replays
    equally.  Affordable, exhaustive, internally consistent - and the wrong weighting."""
    _rank_on_weights(sess, art_dir, [1.0] * len(TRACES))


def strat_rank_nearest_trace(sess, art_dir, rng):
    """Q2 ABLATION, the sharper version: spend a canary call on the stock stack, pick the single trace whose
    stock latency is closest to it, and rank on that trace alone.  This is the heuristic a good engineer
    reaches for - 'find the replay that looks most like production' - and it is a rank-one approximation to
    a five-dimensional mix."""
    p = sess.w.p
    ref = _canary(sess, [], 2)
    base = {T: _replay(sess, [], T, 2) for T in TRACES}
    T = min(base, key=lambda t: abs(base[t] - ref))
    w = [1.0 if t == T else 0.0 for t in TRACES]
    _rank_on_weights(sess, art_dir, w)


def strat_rank_volume(sess, art_dir, rng):
    """Weight the traces by their own stock latency (a plausible proxy for 'how much traffic looks like
    this') rather than by the estimated traffic share."""
    base = [_replay(sess, [], T, 2) for T in TRACES]
    _rank_on_weights(sess, art_dir, base)


def strat_borda(sess, art_dir, rng):
    """Q2 ABLATION by voting: exhaustively rank every feasible set on every trace and ship the set with the
    best average rank.  Mix-free by construction, so it cannot be right for a specific mix."""
    p = sess.w.p
    F = [S for S in feasible_sets(p) if S]
    rl = {}
    for S in F:
        if sess.left() < 12 * TRACE_COST:
            break
        rl[tuple(S)] = [_replay(sess, S, T, 1) for T in TRACES]
    pts = {k: 0.0 for k in rl}
    for i in range(len(TRACES)):
        for rank, k in enumerate(sorted(rl, key=lambda kk: rl[kk][i])):
            pts[k] += rank
    win = list(min(pts, key=lambda k: pts[k]))
    base = [_replay(sess, [], T, 2) for T in TRACES]
    w = np.ones(len(TRACES)) / len(TRACES)
    b0 = float(np.dot(w, base))
    sp = b0 / float(np.dot(w, rl[tuple(win)]))
    gain = {}
    for o in win:
        sub = tuple(sorted(x for x in win if x != o))
        if sub in rl:
            gain[o] = (b0 / float(np.dot(w, rl[tuple(win)])) - 1.0) - (b0 / float(np.dot(w, rl[sub])) - 1.0)
    wk = min(gain, key=lambda o: gain[o]) if gain else win[0]
    _write(art_dir, win, sp * 0.98, sp * 1.02, wk)


def strat_minimax(sess, art_dir, rng):
    """The robust move: ship the set with the best WORST case over the five traces, i.e. over all possible
    mixes.  Requires no mix estimate at all, which is exactly why it cannot hit a specific mix."""
    p = sess.w.p
    F = [S for S in feasible_sets(p) if S]
    rl, base = {}, {}
    for T in TRACES:
        base[T] = _replay(sess, [], T, 1)
    for S in F:
        if sess.left() < 12 * TRACE_COST:
            break
        rl[tuple(S)] = max((_replay(sess, S, T, 1) / base[T]) for T in TRACES)
    win = list(min(rl, key=lambda k: rl[k]))
    w = np.ones(len(TRACES)) / len(TRACES)
    b0 = float(np.mean([base[T] for T in TRACES]))
    L = float(np.mean([_replay(sess, win, T, 2) for T in TRACES]))
    sp = b0 / L
    _write(art_dir, win, sp * 0.98, sp * 1.02, win[0])


def strat_compose_canary(sess, art_dir, rng):
    """Q3 ABLATION: use only the measurements that are actually ON the target.  Take each change's single
    effect from the canary, compose them multiplicatively, and pack greedily under the review-day cap.  Every
    number in it is a genuine production measurement; the composition rule is the error."""
    p = sess.w.p
    b = _canary(sess, [], 3)
    solo = {}
    for c in CHANGES:
        solo[c] = b / _canary(sess, [c], 2)
    cur, tot = [], 0
    for c in sorted(CHANGES, key=lambda c: -solo[c]):
        if solo[c] > 1.0 and tot + p["cost"][c] <= p["cap"]:
            cur.append(c)
            tot += p["cost"][c]
    sp = 1.0
    for c in cur:
        sp *= solo[c]
    wk = min(cur, key=lambda c: solo[c]) if cur else None
    _write(art_dir, cur, sp * 0.99, sp * 1.01, wk)


def strat_right_set_wrong_weakest(sess, art_dir, rng):
    """Isolates R3: the optimal set and an honest interval, but the within-set ranking read off each
    member's solo effect instead of its in-context removal."""
    p = sess.w.p
    S, sp = best_set(p, "prod")
    b = latency(p, [], "prod")
    solo = {o: b - latency(p, [o], "prod") for o in S}
    _write(art_dir, S, sp * 0.995, sp * 1.005, min(solo, key=lambda o: solo[o]))


def strat_wide(sess, art_dir, rng):
    """Correct set, hedged interval: must be rejected by the width cap, never by coverage."""
    p = sess.w.p
    S, sp = best_set(p, "prod")
    w, _ = weakest_member(p, S, "prod")
    _write(art_dir, S, sp * 0.88, sp * 1.12, w)


def strat_nop(sess, art_dir, rng):
    pass


def strat_bad_json(sess, art_dir, rng):
    open(os.path.join(art_dir, "plan.json"), "w").write("{oops")


STRATEGIES = {"oracle": (strat_oracle, "pass"),
              "rank_uniform": (strat_rank_uniform, "fail"),
              "rank_nearest_trace": (strat_rank_nearest_trace, "fail"),
              "rank_volume": (strat_rank_volume, "fail"),
              "borda": (strat_borda, "fail"),
              "minimax": (strat_minimax, "fail"),
              "compose_canary": (strat_compose_canary, "fail"),
              "right_set_wrong_weakest": (strat_right_set_wrong_weakest, "fail"),
              "wide": (strat_wide, "fail"),
              "nop": (strat_nop, "fail"),
              "bad_json": (strat_bad_json, "fail")}
NOISY_FAIL = ()

# The exhaustive sweeps.  All of them enumerate the ENTIRE answer space on the cheap instrument and all of
# them fit the budget - `rank_uniform` and `borda` measure every feasible set on all five traces.  They fail
# because the cheap instrument is biased for the target, which is the property a_stack lacked.
SEARCH = ["rank_uniform", "borda", "minimax"]

PRINCIPLES = {"Q2_mix_must_be_estimated":    ("rank_uniform", ["R1_set_is_near_optimal"]),
              "Q2b_one_trace_is_rank_one":   ("rank_nearest_trace", ["R1_set_is_near_optimal"]),
              "Q3_composition_is_not_additive": ("compose_canary", ["R1_set_is_near_optimal"]),
              "Q4_removal_is_in_context":    ("right_set_wrong_weakest", ["R3_weakest_member"]),
              "Q5_interval_must_be_earned":  ("wide", ["R2_speedup_predicted"])}


# ------------------------------------------------------------------ grader falsification
def _load(art_dir):
    return json.load(open(os.path.join(art_dir, "plan.json")))


def _save(art_dir, rep):
    json.dump(rep, open(os.path.join(art_dir, "plan.json"), "w"))


def _mut_set(p, t, art_dir, leg):
    rep = _load(art_dir)
    if leg == "small":
        rep["opts"] = list(reversed(rep["opts"]))
        rep["notes"] = "reordered; nothing else in the file is read"
    else:
        F = [S for S in feasible_sets(p) if S]
        g = t["best_speedup"] - 1.0
        worst = min(F, key=lambda S: (speedup(p, S, "prod") - 1.0) / g)
        sp = speedup(p, worst, "prod")
        wk, _ = weakest_member(p, worst, "prod")
        rep["opts"], rep["weakest"] = worst, wk          # keep R2/R3 self-consistent so only R1 can break
        rep["speedup"] = {"lo": sp * 0.99, "hi": sp * 1.01}
    _save(art_dir, rep)
    return None if leg == "small" else "R1_set_is_near_optimal"


def _mut_interval_coverage(p, t, art_dir, leg):
    rep = _load(art_dir)
    sp = speedup(p, sorted(rep["opts"]), "prod")
    shift = 0.0 if leg == "small" else 3.0 * WIDTH_CAP * sp
    rep["speedup"] = {"lo": sp + shift - 0.4 * WIDTH_CAP * sp, "hi": sp + shift + 0.4 * WIDTH_CAP * sp}
    _save(art_dir, rep)
    return None if leg == "small" else "R2_speedup_predicted"


def _mut_interval_width(p, t, art_dir, leg):
    rep = _load(art_dir)
    sp = speedup(p, sorted(rep["opts"]), "prod")
    w = (0.9 if leg == "small" else 1.2) * WIDTH_CAP * sp     # centred on the truth either way
    rep["speedup"] = {"lo": sp - w / 2, "hi": sp + w / 2}
    _save(art_dir, rep)
    return None if leg == "small" else "R2_speedup_predicted"


def _mut_weakest(p, t, art_dir, leg):
    rep = _load(art_dir)
    S = sorted(rep["opts"])
    _, rem = weakest_member(p, S, "prod")
    rep["weakest"] = (min(rem, key=lambda o: rem[o]) if leg == "small"
                      else max(rem, key=lambda o: rem[o]))
    _save(art_dir, rep)
    return None if leg == "small" else "R3_weakest_member"


MUTATE = [("set_choice", _mut_set), ("interval_coverage", _mut_interval_coverage),
          ("interval_width", _mut_interval_width), ("weakest_member", _mut_weakest)]


# ------------------------------------------------------------------ instance screen
# The screen is vectorised over all 64 subsets because `sample_params` runs it a few thousand times per seed
# while SEARCHING for a traffic mix that makes every shortcut wrong.  That search is the reason this family
# has a usable yield at all: the five families, the six changes and the review costs fix which sets are
# CANDIDATES, but the traffic mix `pi` alone fixes which of them WINS - and it is hidden, free to choose, and
# affects none of the decoys' own answers.  Screening random mixes gave a yield of 0 in 899 seeds; choosing
# the mix turns the same conditions into a construction.
#
# `_conditions` is the single source of truth for difficulty, used both to propose (inside sample_params) and
# to verify (inside instance_gate).  It is cross-checked against the readable `truth()` on every call, so a
# divergence between the vectorised tables and the code the GRADER uses shows up as a rejected instance
# rather than as a broken task.
_SUBSETS = [tuple(x) for k in range(len(CHANGES) + 1) for x in itertools.combinations(CHANGES, k)]
_INDEX = {s: i for i, s in enumerate(_SUBSETS)}
_EMPTY = _INDEX[()]


def _tables(p):
    Fm = np.array([factors(p, list(S)) for S in _SUBSETS])                  # 64 x NSTAGE
    Bm = np.array([p["b"][T] for T in TRACES])                              # 5 x NSTAGE
    Ltr = Fm.dot(Bm.T)                                                      # 64 x 5, per-family latency
    cost = np.array([sum(p["cost"][o] for o in S) for S in _SUBSETS])
    feas = (cost <= p["cap"])
    feas[_EMPTY] = False
    return {"Ltr": Ltr, "cost": cost, "feas": feas, "Bm": Bm}


def _mix_cov(p, Ltr, pi):
    """Covariance of the reference protocol's traffic-mix estimate, first order, exactly as it is estimated.

    The first version of this screen modelled only the canary noise and ignored the Sum(pi)=1 constraint.
    Measured against 40 real runs of the oracle it UNDER-stated the prediction error by ~25% (1.01% claimed,
    1.25% observed on seed 12), because the design matrix is itself measured: its rows are 1-rep replays,
    an errors-in-variables problem.  The oracle's interval, calibrated on fit residuals with ~2 degrees of
    freedom, was narrower still, and missed on 5/8 salts.  Both are fixed here and in `strat_oracle`:
    the row variance carries the replay noise of A, and the constraint is projected in.
    """
    design = [_EMPTY] + [_INDEX[(c,)] for c in CHANGES]
    A = Ltr[design]
    y = A.dot(pi)
    s2 = p["sig"] ** 2
    var = s2 * (y ** 2 / CANARY_REPS_REF + ((A * pi) ** 2).sum(axis=1) / SWEEP_REPS)
    Cu = np.linalg.inv((A / var[:, None]).T.dot(A))
    u = Cu.sum(axis=1)
    return Cu - np.outer(u, u) / u.sum()


def _conditions(p, tb=None, identity=False):
    tb = tb or _tables(p)
    Ltr, feas = tb["Ltr"], tb["feas"]
    pi = np.array(p["pi"])
    Lpr = Ltr.dot(pi)                                                       # 64, production latency
    fi = np.where(feas)[0]
    base = Lpr[_EMPTY]
    sp = base / Lpr
    star = int(fi[np.argmin(Lpr[fi])])
    S_star = list(_SUBSETS[star])
    gain = sp[star] - 1.0
    ratio = np.where(feas, (sp - 1.0) / max(1e-9, gain), -1.0)
    qual = fi[ratio[fi] >= TAU_GAIN]
    rs = np.sort(ratio[fi])[::-1]
    gap = float(rs[len(qual) - 1] - rs[len(qual)]) if 0 < len(qual) < len(fi) else 9.9

    # --- SOLVABILITY, in the reference protocol's own error model.  Every decision the protocol makes is a
    # comparison of two pi-weighted replay latencies, so it is the variance of a LOG RATIO that matters, and
    # the shared mix error largely cancels in it (the earlier screen added the two variances as if they
    # were independent, which is conservative for rankings but says nothing true about any one of them).
    try:
        C = _mix_cov(p, Ltr, pi)
    except np.linalg.LinAlgError:
        return [False], {"c": [False], "singular": True}
    X = Ltr / Lpr[:, None]
    Q = ((X * pi) ** 2).sum(axis=1) * p["sig"] ** 2

    def vlr(i, J, reps):
        G = X[J] - X[i]
        return np.einsum("ij,jk,ik->i", G, C, G) + (Q[i] + Q[J]) / reps

    others = fi[fi != star]
    z_all = np.abs(np.log(Lpr[others] / Lpr[star])) / np.sqrt(np.maximum(1e-30, vlr(star, others, SWEEP_REPS)))
    nq_mask = ratio[others] < TAU_GAIN
    min_pair_z = float(z_all[nq_mask].min()) if nq_mask.any() else 99.0
    # Every set the protocol could plausibly ship: the optimum, and any QUALIFYING set it cannot tell apart
    # from the optimum at 3 sigma.  R2 and R3 are graded on the set the agent ships, so both must be
    # reachable on each of these, not just on the optimum.
    reach = [star] + [int(j) for j, z, q in zip(others, z_all, nq_mask) if (not q) and z < 3.0]

    def weak_stats(i):
        S = _SUBSETS[i]
        loo = {o: _INDEX[tuple(x for x in S if x != o)] for o in S}
        rem = {o: sp[i] - sp[j] for o, j in loo.items()}
        order = sorted(rem, key=lambda o: rem[o])
        w, s = order[0], order[1]
        margin = rem[s] - rem[w]
        zw = abs(math.log(Lpr[loo[s]] / Lpr[loo[w]])) / math.sqrt(max(1e-30, float(vlr(loo[w], [loo[s]], SWEEP_REPS)[0])))
        rel = math.sqrt(float(vlr(_EMPTY, [i], REFINE_REPS)[0]))
        return w, margin, zw, rel

    ws = {i: weak_stats(i) for i in reach}
    weakest, weak_margin, weak_z, rel_star = ws[star]
    worst_margin = min(v[1] for v in ws.values())
    worst_weak_z = min(v[2] for v in ws.values())
    worst_rel = max(v[3] for v in ws.values())

    # --- every shortcut's answer, in closed form
    tb_best = [int(fi[np.argmin(Ltr[fi, k])]) for k in range(len(TRACES))]
    n_trace_qual = int(sum(1 for i in tb_best if ratio[i] >= TAU_GAIN))
    nearest = int(np.argmin(np.abs(Ltr[_EMPTY] - base)))
    nearest_ratio = float(ratio[tb_best[nearest]])
    trace_sp_bias = float(min(abs((Ltr[_EMPTY, k] / Ltr[tb_best[k], k]) / sp[tb_best[k]] - 1.0)
                              for k in range(len(TRACES))))
    norm = Ltr / Ltr[_EMPTY]                                               # each family's own speedup scale
    big = np.where(feas, 0.0, 9e9)
    uni = int(np.argmin(norm.sum(axis=1) + big))
    mmx = int(np.argmin(norm.max(axis=1) + big))
    pts = np.zeros(len(_SUBSETS))
    for k in range(len(TRACES)):
        order = fi[np.argsort(Ltr[fi, k])]
        for r, i in enumerate(order):
            pts[i] += r
    borda = int(fi[np.argmin(pts[fi])])
    solo = {c: float(sp[_INDEX[(c,)]]) for c in CHANGES}

    def greedy(key):
        cur, tot = [], 0
        for c in sorted(CHANGES, key=key):
            if solo[c] > 1.0 and tot + p["cost"][c] <= p["cap"]:
                cur.append(c)
                tot += p["cost"][c]
        return _INDEX[tuple(sorted(cur))]

    comp = greedy(lambda c: -solo[c])
    eff = greedy(lambda c: -(solo[c] - 1.0) / p["cost"][c])
    naive = 1.0
    for c in S_star:
        naive *= solo[c]
    pred_err = abs(naive - sp[star]) / sp[star]
    solo_gain = {o: base - Lpr[_INDEX[(o,)]] for o in S_star}
    solo_weak = min(solo_gain, key=lambda o: solo_gain[o])
    bp_vec = tb["Bm"].T.dot(pi)
    skew = float(bp_vec.max() / bp_vec.sum())
    cond = float(np.linalg.cond(tb["Bm"] / tb["Bm"].sum(axis=1, keepdims=True)))
    pi_spread = int((pi >= 0.12).sum())

    ident = 0.0
    if identity:
        # CORRECTNESS INVARIANT, not a difficulty condition.  The family rests on production latency being
        # the pi-weighted combination of the five family latencies FOR EVERY SET - the only reason a protocol
        # can reach an unmeasurable target at all.  Checked against the grader's own `latency`, over all 64
        # subsets, so an edit introducing any production-only term empties the pool instead of shipping
        # unsolvable instances.  Also re-checks that these vectorised tables agree with `truth()`, which is
        # what the grader uses.
        for i, S in enumerate(_SUBSETS):
            direct = latency(p, list(S), "prod")
            viamix = sum(p["pi"][k] * latency(p, list(S), T) for k, T in enumerate(TRACES))
            ident = max(ident, abs(direct - viamix) / max(1e-9, direct), abs(direct - Lpr[i]) / max(1e-9, direct))
        tr = truth(p)
        if sorted(tr["best_set"]) != sorted(S_star) or tr["weakest"] != weakest:
            ident = 9.9

    c = [gain >= 0.40,                                              # 1 a real decision exists
         3 <= len(S_star) <= len(CHANGES) - 1,                      # 2 the cap binds; R3 is non-trivial
         1 <= len(qual) <= 4,                                       # 3 no plateau: guessing does not pay
         gap >= 0.04,                                               # 4 the qualifying boundary is decidable
         (not identity) or ident <= 1e-9,                           # 5 INVARIANT: prod is exactly the mix
         nearest_ratio < TAU_GAIN and n_trace_qual <= 1,            # 6 Q2: ranking on one family fails
         trace_sp_bias >= 1.5 * WIDTH_CAP,                          # 7 ... and its interval fails too
         ratio[uni] < TAU_GAIN,                                     # 8 ... nor the equal-weight average's set
         ratio[mmx] < TAU_GAIN,                                     # 9 ... nor the minimax answer
         ratio[borda] < TAU_GAIN,                                   # 10 ... nor the Borda winner
         ratio[comp] < TAU_GAIN,                                    # 11 Q3: greedy on solo canary effects
         ratio[eff] < TAU_GAIN,                                     # 12 ... nor greedy on effect per day
         pred_err >= 1.2 * WIDTH_CAP,                               # 13 ... and composing intervals fails
         solo_weak != weakest,                                      # 14 Q4: solo ranking mis-picks R3
         worst_margin >= 3.0 * WEAK_EPS,                            # 15 R3 decidable beyond grading slack
         worst_weak_z >= 3.0,                                       # 16 ... and beyond the protocol's noise
         worst_rel <= WIDTH_CAP / 6.0,                              # 17 SOLVABLE: a 2.5-sigma interval fits
         min_pair_z >= 3.0,                                         # 18 SOLVABLE: the ranking survives noise
         cond <= 100.0,                                             # 19 the five family shapes are usable
         pi_spread >= 3 and pi.max() <= 0.40,                       # 20 production is a genuine blend
         16 <= int(feas.sum()) <= 36]                               # 21 replay sweep fits the budget
    info = {"c": [bool(x) for x in c], "best_set": S_star, "best_speedup": round(float(sp[star]), 5),
            "n_feasible": int(feas.sum()), "n_qualifying": int(len(qual)), "ratio_gap_at_tau": round(gap, 4),
            "weakest": weakest, "weak_margin": round(weak_margin, 4), "weak_z": round(weak_z, 2),
            "solo_weak": solo_weak, "n_reachable": len(reach), "worst_margin": round(worst_margin, 4),
            "worst_weak_z": round(worst_weak_z, 2), "rel_star": round(rel_star, 5), "worst_rel": round(worst_rel, 5),
            "min_pair_z": round(min_pair_z, 3), "cond": round(cond, 2), "ident": float("%.2g" % ident),
            "skew": round(skew, 4), "pi_spread": pi_spread, "nearest_trace": TRACES[nearest],
            "nearest_ratio": round(nearest_ratio, 4), "n_trace_qual": n_trace_qual,
            "trace_sp_bias": round(trace_sp_bias, 4), "uniform_ratio": round(float(ratio[uni]), 4),
            "minimax_ratio": round(float(ratio[mmx]), 4), "borda_ratio": round(float(ratio[borda]), 4),
            "compose_ratio": round(float(ratio[comp]), 4), "compose_set": list(_SUBSETS[comp]),
            "eff_ratio": round(float(ratio[eff]), 4), "eff_set": list(_SUBSETS[eff]),
            "naive_pred_err": round(float(pred_err), 4), "pi": [round(float(x), 4) for x in pi]}
    return [bool(x) for x in c], info


def instance_gate(p):
    c, info = _conditions(p, identity=True)
    return bool(all(c)), info


def pool_gate(seeds):
    """No fixed artifact may satisfy R1 on two instances of the pool.

    The answer is a subset of six changes under a cost cap, so the v7 form applies here (unlike b_control,
    whose one-of-five answer needs a modal-answer cap instead): replay each admitted instance's optimal set
    on every other instance and require that a set which is optimal for one is never near-optimal for
    another.  `best_fixed` is the largest number of instances any single fixed set satisfies; it must be 1.
    """
    ps = [sample_params(s) for s in seeds]
    ps = [p for p in ps if instance_gate(p)[0]]
    if len(ps) < 2:
        return True, {"n": len(ps)}
    cands = {tuple(truth(p)["best_set"]) for p in ps}
    best, arg = 0, None
    for S in cands:
        n = 0
        for p in ps:
            if sum(p["cost"][o] for o in S) > p["cap"]:
                continue
            t = truth(p)
            if (speedup(p, list(S), "prod") - 1.0) / max(1e-9, t["best_speedup"] - 1.0) >= TAU_GAIN:
                n += 1
        if n > best:
            best, arg = n, S
    return bool(best <= 1), {"n": len(ps), "best_fixed": best, "best_fixed_set": list(arg or ())}


# ------------------------------------------------------------------ task text
def instruction(p, t):
    cat = "\n".join("| `%s` | %d |" % (c, p["cost"][c]) for c in CHANGES)
    return """# Serving-stack rollout plan for production traffic

You own the inference serving stack.  Six changes are queued; each costs review-days out of this quarter's
budget, and you may ship any subset whose total review cost is at most **%d review-days**.

| change | review-days |
|---|---|
%s

Your objective is **mean latency on live production traffic**.

You have **%g lab credits**.

* `lab replay opts=<ids> trace=<%s> reps=<1..%d>` - replay one request family through the stack with those
  changes applied.  **reps credits.**  Any set of changes, any family.
* `lab canary opts=<ids> reps=<1..%d>` - serve live production traffic with those changes applied.
  **reps x %g credits.**  The canary rollout gate admits **at most %d change per window**: `opts` may name
  one change, or none.  A larger set cannot be canaried at any price.

Every measurement carries **+-%.2f%%** independent multiplicative noise (mean zero), and repeats average it
down.  There is no other randomness in this task.

## What production traffic is

Requests are routed to one of five **request families**.  The five trace replays are exactly those families,
one family each, in isolation.  Production traffic is a mixture of the same five families; the routing layer
does not tag requests, so **the share of each family in production traffic is not instrumented** and appears
in no dashboard.

Latency for a given set of changes is the sum over the stack's internal stages of the time spent in each
stage.  A change divides the stages it touches by a fixed factor and adds a small fixed fraction to one
other stage; those factors are properties of the change, not of the workload.  How much of a request's time
lands in each stage is a property of the request family.  Neither the stage breakdown nor the per-change
factors are published, and the vendor's release notes give no per-change numbers.

## Deliverable: `/app/plan.json`

```json
{"opts": ["%s", "%s"],
 "speedup": {"lo": 1.00, "hi": 1.00},
 "weakest": "%s"}
```

* `opts` - the set you will ship.  Graded on the fraction of the achievable production gain it captures;
  you need at least **%.0f%%**.  Total review cost must be within the cap.
* `speedup` - an interval for the production speedup (stock latency / your set's latency) **of the set you
  yourself named**.  Relative width `(hi-lo)/truth` may not exceed **%.0f%%**.
* `weakest` - the member of your own set whose removal would cost the least production speedup, with the
  rest of your set still in place.
""" % (p["cap"], cat, BUDGET, "|".join(TRACES), MAX_REPS, MAX_REPS, CANARY_COST, CANARY_MAX_OPTS,
       100 * p["sig"], CHANGES[0], CHANGES[2], CHANGES[0], 100 * TAU_GAIN, 100 * WIDTH_CAP)


def docs(p):
    return {"docs/plan_format.md": """# /app/plan.json

```json
{"opts": ["C-114", "C-203", "C-241"],
 "speedup": {"lo": 1.61, "hi": 1.68},
 "weakest": "C-241"}
```

* `opts` (list of strings, non-empty, distinct) - the changes you ship.  Total review cost must be <= the
  cap stated in the task (**%d review-days** for this instance).
* `speedup.lo`, `speedup.hi` (numbers) - production speedup of the set in `opts`, i.e. stock production
  latency divided by your set's production latency.  Graded against YOUR set, not against the optimal one,
  so a suboptimal set costs you one item and not two.  `(hi-lo)/truth` must be <= %.2f.
* `weakest` (string) - must be one of the ids in your own `opts`.

Nothing else in the file is read, and there is no partial credit inside an item.
""" % (p["cap"], WIDTH_CAP),
            "docs/stack.md": """# Stack and traffic model

Latency of one request is the sum of the time it spends in each of the stack's internal stages.  Two facts
about the queued changes, from the vendor's engineering notes:

* Each change **divides** the stage or stages it touches by a fixed factor, and **adds** a small fixed
  fraction of one other stage.  Those factors and fractions are properties of the change.  They do not
  depend on which request family is running, and they do not depend on which other changes are shipped
  alongside it.
* How much of a request's time lands in each stage depends on the **request family**.  The stage breakdown
  is not exported by the runtime, for any family.

Consequences worth being explicit about, because the last two quarterly plans were rejected over them:

* A change's measured effect on one request family does not carry over to another family, and the effect of
  a *set* of changes is not the combination of their individual effects - two changes that relieve the same
  stage overlap, and a change that looks worthless alone can be the best one once another change has
  removed the stage that was dominating.
* Production is not one of the replayable families.  It is traffic made of all five of them, in proportions
  nobody has measured.

## The canary

The canary rollout gate exists because a bad change on live traffic costs real money, and it admits one
queued change per window.  This is a hard operational limit, not a budget matter: there is no way to put a
multi-change bundle in front of production traffic before the plan is approved.
"""}


def hints(p):
    return {1: """
Hint 1: production latency and the five replay latencies are not five unrelated numbers.  Ask yourself what
the mean latency of a mixture of request families is, given the mean latency of each family.
""", 2: """
Hint 2: mean production latency for ANY set of changes is the same fixed convex combination of that set's
five replay latencies - the combination being the traffic shares, which are the same for every set.  The
single-change canary points are enough to identify those shares, and there are more of them than there are
shares, so the leftovers can check the fit.
"""}
