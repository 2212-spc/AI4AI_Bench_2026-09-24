"""A-STACK - **CUT ON 2026-09-28 BY THE SEARCH GATE.**  Kept as that gate's calibration case; not shipped.

This family was built, hardened and accepted: 22 instances out of 12000 screened seeds, all 22 ACCEPT at five
noise salts, a 17-instance pool with `best_fixed == 1`, eight closed-form instance conditions, a two-principle
ablation certificate, and nine decoys that all failed as designed.  Then one probe killed it:

    benchmark every feasible option set on the TARGET workload at one repetition, take the argmax, and read
    the leave-one-out ranking off the measurements you already made

passed **18/18** runs across nine instances, spending 65-88% of the budget, with no insight in it whatsoever.
The arithmetic: ~25 feasible sets, a target benchmark costs 4 of 160 credits, so exhaustive search costs
<=136; and because the instance screen selected for instances where only 1-2 feasible sets clear TAU_GAIN, the
gap between best and second-best was far larger than the ~0.8% measurement noise, so one repetition per set
sufficed to rank them.

Why patching does not save it:
  * Capping target benchmarks does not work.  At a cap of 14 with ~25 feasible sets, RANDOM search finds a
    qualifying set ~80% of the time; pushing that below 20% needs the answer space an order of magnitude
    larger (|F| >= ~80, i.e. eight or more options) AND the search allowance down to ~5 calls.
  * The root cause is not the size of the search.  It is that the naive measurement of a candidate - bench
    this exact set on the production workload - is UNBIASED.  Non-additive composition only punishes an agent
    that tries to PREDICT a set's latency from its parts; an agent that measures the composite directly never
    meets the non-additivity at all.  Contrast b_control, where an exhaustive sweep of the answer space is the
    skeleton of the reference solution and still fails, because each arm of the naive sweep estimates the
    wrong quantity.

The generalisation, now enforced by `l15.v8gates.search_gate` and stated in DESIGN.md: a "choose the best
configuration" task is searchable and therefore not hard, unless the target configuration or target workload
cannot be measured at all.  The replacement family `a_span` keeps this file's latency model and makes the
production workload unmeasurable - the canary admits at most one change at a time, so no candidate set can
ever be benchmarked on the target, and the only route to an answer is to reconstruct the target from a
spanning set of cheap trace replays.

Everything below is the original, unmodified.
--------------------------------------------------------------------------------------------------------------
A-STACK: pick a set of serving optimizations under a review-cost cap, and predict the speedup.

Difficulty mechanism (this family's *signature*, distinct from the other v8 families):
  P1  MARGINAL-IN-CONTEXT.  Latency is a sum over pipeline stages and each optimization multiplies the
      stages it touches, so two optimizations that relieve the SAME stage do not stack, and one that looks
      worthless alone becomes the best once the dominant stage is gone.  An option's effect measured
      against the stock stack is not its effect inside the shipped set, so ranking options by their solo
      gain - the default move - selects the wrong set and over-predicts the result.
  P2  PROXY-TO-TARGET.  The replica the lab benches cheaply (`workload="dev"`) and the production replay
      (`workload="prod"`, 4x the price) put their time in DIFFERENT internal stages - which stage dominates
      each is re-drawn per seed and is never disclosed.  A change's per-stage effect is the same on both
      workloads, but the mix is not, so the dev ranking is rank-correlated and argmax-wrong: the cheap box
      can exclude candidates but cannot pick the winner.  The budget is set so that neither "decide on dev"
      nor "measure everything on prod at depth" fits.

Both wrong protocols produce a complete plan whose interval is confirmed by the agent's own measurements.
That is the invariant this family is built to satisfy: the agent's own validation passes on the wrong answer.

Ground truth is exact: with K options there are 2^K subsets; the cost filter and each subset's latency are
closed-form, so the optimal feasible set, its speedup, and the within-set ranking are computed, not sampled.
"""
import itertools, json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for

STAGES = ["prefill", "attn", "mlp", "sample", "comm"]
OPTS = ["C-114", "C-127", "C-203", "C-241", "C-318", "C-352"]
BUDGET = 160.0
DEV_COST = 1.0
PROD_COST = 4.0
MAX_REPS = 4
TAU_GAIN = 0.90
WIDTH_CAP = 0.06
WEAK_EPS = 0.02
SHORTLIST = 14          # oracle keeps this many dev-best sets; the gate demands rank <= SHORTLIST-2

# There is deliberately NO public per-change description.  An earlier draft named the changes after the
# stage they touch (`prefix_cache`, `kv_int8`, ...) and, once the role->name map was permuted per seed to
# close a pool leak, those names became false advertising: the catalogue text would have implied a stage
# the change no longer touched.  The stage decomposition is not observable through `bench` anyway - it
# returns end-to-end latency only - so the honest catalogue is the change IDs, their review cost, and an
# explicit statement that no per-change performance data is published.


def _role_specs(p_dom, d_dom, rest):
    """Six latent roles, generated RELATIVE to the two workload-dominant stages.

    `(targets, mult_range, (overhead_stage, overhead_range), cost_range)`.  The shape is what creates the
    two failure modes, and it has to be generated rather than hardcoded so that nothing carries across
    seeds (an earlier fixed catalogue let one hardcoded option list be near-optimal on 5 of 6 instances):

      * roles 0,1,2 all divide the PROD-dominant stage.  They are substitutes: their multipliers compose,
        so each one's marginal value collapses once the others are in.  Ranking by solo gain therefore
        over-states the pack and mis-orders it from the inside.
      * role 3 divides prod's SECOND stage, which nothing else touches.  Its solo gain is smaller than a
        contended option's but it is not discounted in context - so the member with the smallest solo gain
        is usually NOT the weakest member of the winning set.  That is the whole content of R3.
      * roles 4,5 divide the DEV-dominant stage and role 4 taxes the prod-dominant one: top of the dev
        leaderboard, actively harmful on prod.
    Costs are attached to roles, not drawn independently, so the review cap keeps forcing the same *kind*
    of trade-off (you cannot afford all three substitutes plus the uncontended helper) on every seed.
    """
    m1, m2, m3 = rest
    return [
        ([p_dom],       (0.40, 0.80), (m2, (0.02, 0.05)), (4, 6)),
        ([p_dom],       (0.40, 0.80), (d_dom, (0.03, 0.08)), (4, 6)),
        ([p_dom, m3],   (0.50, 0.85), (m2, (0.03, 0.08)), (3, 5)),
        ([m1],          (0.30, 0.60), (m3, (0.01, 0.04)), (2, 3)),
        ([d_dom],       (0.30, 0.65), (p_dom, (0.06, 0.14)), (2, 4)),
        ([d_dom, m2],   (0.45, 0.80), (m1, (0.03, 0.08)), (3, 5)),
    ]


def sample_params(seed):
    g = np.random.default_rng(41000 + seed)
    ns = len(STAGES)
    order = [int(x) for x in g.permutation(ns)]
    p_dom, d_dom, rest = order[0], order[1], order[2:]

    def mix(dom, sec, scale):
        w = g.uniform(0.04, 0.10, ns)
        w[dom] = g.uniform(0.34, 0.50)
        w[sec] = g.uniform(0.14, 0.24)
        return w / w.sum() * scale

    # both workloads put their SECOND-largest share on rest[0].  That shared stage is what makes the dev
    # box a usable screen (the prod optimum stays in dev's top few) even though its argmax is wrong - the
    # proxy is rank-correlated, not rank-preserving.
    prod = mix(p_dom, rest[0], float(g.uniform(320, 460)))
    dev = mix(d_dom, rest[0], float(g.uniform(180, 260)))

    roles = _role_specs(p_dom, d_dom, rest)
    perm = [int(x) for x in g.permutation(len(OPTS))]
    mult = np.ones((len(OPTS), ns))
    ovf = np.zeros((len(OPTS), ns))
    cost = {}
    for i, o in enumerate(OPTS):
        tgt, (mlo, mhi), (ost, (lo, hi)), (clo, chi) = roles[perm[i]]
        for st in tgt:
            mult[i, st] = float(g.uniform(mlo, mhi))
        ovf[i, ost] = float(g.uniform(lo, hi))
        cost[o] = int(g.integers(clo, chi + 1))
    cap = int(g.integers(9, 13))
    return {"dev": [round(x, 3) for x in dev], "prod": [round(x, 3) for x in prod],
            "mult": [[round(x, 4) for x in row] for row in mult],
            "ovf": [[round(x, 4) for x in row] for row in ovf],
            "role_perm": perm, "cost": cost, "cap": cap,
            "sig": round(float(g.uniform(0.006, 0.010)), 4)}


def latency(p, S, wl):
    b = np.array(p[wl], float)
    m = np.array(p["mult"], float)
    ov = np.array(p["ovf"], float)
    idx = [OPTS.index(o) for o in S]
    pm = np.ones(len(STAGES)) if not idx else m[idx].prod(axis=0)
    ex = np.zeros(len(STAGES)) if not idx else ov[idx].sum(axis=0)
    return float((b * (pm + ex)).sum())


def speedup(p, S, wl):
    return latency(p, [], wl) / latency(p, S, wl)


def feasible_sets(p):
    out = []
    for r in range(len(OPTS) + 1):
        for S in itertools.combinations(OPTS, r):
            if sum(p["cost"][o] for o in S) <= p["cap"]:
                out.append(list(S))
    return out


def best_set(p, wl):
    F = feasible_sets(p)
    v = [speedup(p, S, wl) for S in F]
    i = int(np.argmax(v))
    return F[i], v[i]


def weakest_member(p, S, wl):
    if not S:
        return None, {}
    g0 = speedup(p, S, wl) - 1.0
    rem = {o: g0 - (speedup(p, [x for x in S if x != o], wl) - 1.0) for o in S}
    return min(rem, key=lambda k: rem[k]), rem


def truth(p):
    S, sp = best_set(p, "prod")
    Sd, _ = best_set(p, "dev")
    w, rem = weakest_member(p, S, "prod")
    rv = sorted(rem.values())
    F = feasible_sets(p)
    rank_dev = sorted(F, key=lambda X: -speedup(p, X, "dev"))
    return {"best_set": S, "best_speedup": round(sp, 5), "best_set_dev": Sd,
            "dev_set_on_prod_gain_ratio": round((speedup(p, Sd, "prod") - 1) / max(1e-9, sp - 1), 4),
            "dev_rank_of_prod_optimum": 1 + rank_dev.index(S), "n_feasible": len(F),
            "weakest": w, "weak_margin": round((rv[1] - rv[0]) if len(rv) > 1 else 9.9, 4),
            "base_dev": round(latency(p, [], "dev"), 2), "base_prod": round(latency(p, [], "prod"), 2)}


# ------------------------------------------------------------------ ops
def _parse(a, p):
    S = a.get("opts", [])
    if isinstance(S, str):
        S = [x.strip() for x in S.split(",") if x.strip()]
    if not isinstance(S, list):
        raise LabError("opts must be a list of option names (or a comma-separated string)")
    S = [str(x) for x in S]
    for o in S:
        if o not in OPTS:
            raise LabError("unknown option %r; catalogue: %s" % (o, OPTS))
    if len(set(S)) != len(S):
        raise LabError("opts contains duplicates")
    wl = a.get("workload", "dev")
    if wl not in ("dev", "prod"):
        raise LabError("workload must be 'dev' or 'prod'")
    reps = int(num(a.get("reps", 1), "reps", 1, MAX_REPS, integer=True))
    return sorted(set(S)), wl, reps


def _cost_bench(w, a):
    _S, wl, reps = _parse(a, w.p)
    return reps * (DEV_COST if wl == "dev" else PROD_COST)


def _run_bench(w, a, ctx):
    S, wl, reps = _parse(a, w.p)
    L = latency(w.p, S, wl)
    r = rng_for(w.salt, "bench", wl, ",".join(S), ctx["i"])
    vals = [L * (1.0 + float(r.normal(0, w.p["sig"]))) for _ in range(reps)]
    return {"workload": wl, "opts": S, "reps": reps, "latency_ms": round(float(np.mean(vals)), 3),
            "per_rep_ms": [round(v, 3) for v in vals],
            "review_cost_of_set": sum(w.p["cost"][o] for o in S)}


class World(_W):
    NAME = "a_stack"
    ARTIFACTS = ["plan.json"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "bench units"
    OPS = {"bench": (_cost_bench, _run_bench, "measure end-to-end latency of an option set on a workload")}

    def public_spec(self):
        return {"ops": {"bench": {
            "args": {"opts": "list of catalogue option names (empty list = stock stack)",
                     "workload": "'dev' (staging replica) or 'prod' (production traffic mix)",
                     "reps": "independent repeats, 1..%d" % MAX_REPS},
            "cost": "reps x %g on dev, reps x %g on prod" % (DEV_COST, PROD_COST),
            "returns": "mean and per-rep end-to-end latency in ms, and the review cost of that set"}},
            "catalogue": {o: {"review_cost": self.p["cost"][o]} for o in OPTS},
            "review_cost_cap": self.p["cap"],
            "deliverable": "/app/plan.json (see /app/docs/plan_format.md)"}

    def grade(self, art_dir, ledger=None):
        t = truth(self.p)
        items = {}
        try:
            rep = json.load(open(os.path.join(art_dir, "plan.json")))
        except Exception as e:
            return {"pass": False, "score": 0.0, "diag": {}, "ref": t,
                    "items": {"R0_plan": {"ok": False,
                                          "detail": "plan.json missing or unreadable: %s" % type(e).__name__}}}
        why, ok0, S = [], True, rep.get("opts")
        if not isinstance(S, list) or any(o not in OPTS for o in S) or len(set(S)) != len(S):
            ok0 = False; why.append("opts must be a list of distinct catalogue names, got %r" % (S,)); S = []
        else:
            S = sorted(set(S))
            if sum(self.p["cost"][o] for o in S) > self.p["cap"]:
                ok0 = False
                why.append("review cost %d exceeds the cap %d"
                           % (sum(self.p["cost"][o] for o in S), self.p["cap"]))
        iv = rep.get("speedup")
        if not (isinstance(iv, dict) and "lo" in iv and "hi" in iv):
            ok0 = False; why.append("speedup must be an object with numeric lo/hi")
        if rep.get("weakest") is not None and rep.get("weakest") not in OPTS:
            ok0 = False; why.append("weakest must be an option name")
        items["R0_plan"] = (ok0, "; ".join(why) or "plan.json is well-formed and within the review-cost cap")
        if not ok0:
            return {"pass": False, "score": 0.0, "diag": {"declared": S}, "ref": t,
                    "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()}}

        sp = speedup(self.p, S, "prod")
        ratio = (sp - 1.0) / max(1e-9, t["best_speedup"] - 1.0)
        items["R1_set_is_near_optimal"] = (
            ratio >= TAU_GAIN,
            "your set reaches %.4fx on the production mix; the best feasible set reaches %.4fx -> you capture "
            "%.1f%% of the achievable gain, need %.0f%%" % (sp, t["best_speedup"], 100 * ratio, 100 * TAU_GAIN))

        try:
            lo, hi = float(iv["lo"]), float(iv["hi"])
            bad = not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi)
        except Exception:
            lo = hi = float("nan"); bad = True
        if bad:
            items["R2_speedup_predicted"] = (False, "malformed interval")
        else:
            wide = (hi - lo) / max(1e-9, sp) > WIDTH_CAP + 1e-9
            cov = lo <= sp <= hi
            items["R2_speedup_predicted"] = (
                bool(cov and not wide),
                "the true production speedup of YOUR set is %.4fx; you reported [%.4f, %.4f] (%s, relative width "
                "%.3f, cap %.3f)" % (sp, lo, hi, "covers" if cov else "misses",
                                     (hi - lo) / max(1e-9, sp), WIDTH_CAP))

        w_true, rem = weakest_member(self.p, S, "prod")
        wk = rep.get("weakest")
        if not S:
            items["R3_weakest_member"] = (False, "empty set has no weakest member")
        elif wk not in S:
            items["R3_weakest_member"] = (False, "weakest must name one of the options you shipped")
        else:
            mn = min(rem.values())
            items["R3_weakest_member"] = (
                rem[wk] <= mn + WEAK_EPS,
                "removing `%s` from your own set costs %.4f of speedup gain; the least-contributing member is "
                "`%s` at %.4f (slack %.3f).  Ranking inside your set: %s"
                % (wk, rem[wk], w_true, mn, WEAK_EPS,
                   ", ".join("%s %.4f" % (k, v) for k, v in sorted(rem.items(), key=lambda x: x[1]))))

        diag = {"declared": S, "true_best": t["best_set"], "sp": round(sp, 5), "gain_ratio": round(ratio, 4),
                "dev_optimal": t["best_set_dev"], "picked_dev_optimum": sorted(S) == sorted(t["best_set_dev"])}
        if ledger is not None:
            diag["n_dev"] = sum(1 for r in ledger if r["op"] == "bench" and r["args"].get("workload", "dev") == "dev")
            diag["n_prod"] = sum(1 for r in ledger if r["op"] == "bench" and r["args"].get("workload") == "prod")
            diag["spent"] = round(sum(r["cost"] for r in ledger), 2)
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(float(ratio), 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": t}


# ------------------------------------------------------------------ strategies
def _write(art_dir, S, lo, hi, weakest):
    json.dump({"opts": list(S), "speedup": {"lo": lo, "hi": hi}, "weakest": weakest},
              open(os.path.join(art_dir, "plan.json"), "w"))


def _bench(sess, S, wl="dev", reps=1):
    return sess.call("bench", {"opts": list(S), "workload": wl, "reps": reps})["latency_ms"]


def strat_oracle(sess, art_dir, rng):
    """Generic protocol, no instance knowledge.  Budget plan:

        dev screen of every feasible set, 1 rep                  <= 36 units
        prod: shortlist of the best SHORTLIST dev sets, 1 rep        48 units
        prod: leave-one-out of the winner, 2 reps                <= 40 units
        prod: stock stack and the winner, 3 reps each                24 units

    The shortlist is what makes it fit: the dev replica cannot RANK the candidates (its stage mix is
    different) but it can EXCLUDE most of them, and the target workload only has to rank what survives.
    """
    p = sess.w.p
    F = [S for S in feasible_sets(p) if S]
    dev = {}
    for S in sorted(F, key=lambda X: -sum(p["cost"][o] for o in X)):
        if sess.left() < 124.0:
            break
        dev[tuple(S)] = _bench(sess, S, "dev")
    short = [list(k) for k in sorted(dev, key=lambda k: dev[k])[:SHORTLIST]]
    prod = {}
    for S in short:
        if sess.left() < 60.0:
            break
        prod[tuple(S)] = _bench(sess, S, "prod")
    win = list(min(prod, key=lambda k: prod[k])) if prod else (short[0] if short else [])
    # Within-set ranking: remove each member and re-measure ON THE TARGET WORKLOAD.  The member whose
    # removal costs the LEAST is the weakest, i.e. the smallest latency without it.
    rem = {}
    for o in win:
        if sess.left() < 32.0:
            break
        rem[o] = _bench(sess, [x for x in win if x != o], "prod", 2)
    wk = min(rem, key=lambda o: rem[o]) if rem else (win[0] if win else None)
    reps = 3 if sess.left() >= 6 * PROD_COST else (2 if sess.left() >= 4 * PROD_COST else 1)
    base_p = _bench(sess, [], "prod", reps)
    Lw = _bench(sess, win, "prod", reps)
    sp = base_p / Lw
    half = max(0.004, 2.6 * sp * p["sig"] * math.sqrt(2.0 / reps))
    _write(art_dir, win, round(sp - half, 4), round(sp + half, 4), wk)


def strat_dev_only(sess, art_dir, rng):
    """P2 ABLATION: identical search, every decision made on the cheap replica.  The final number is still
    measured on prod, so the interval is honest and only the SET is wrong."""
    p = sess.w.p
    F = [S for S in feasible_sets(p) if S]
    dev = {}
    for S in F:
        if sess.left() < 6 * PROD_COST:
            break
        dev[tuple(S)] = _bench(sess, S, "dev", 2)
    win = list(min(dev, key=lambda k: dev[k]))
    rem = {}
    Lw = dev[tuple(win)]
    for o in win:
        k = tuple(sorted(x for x in win if x != o))
        rem[o] = dev.get(k, Lw + 1.0) - Lw
    wk = min(rem, key=lambda o: rem[o]) if rem else win[0]
    b = _bench(sess, [], "prod")
    f = _bench(sess, win, "prod")
    sp = b / f
    half = max(0.004, 1.5 * sp * p["sig"])
    _write(art_dir, win, round(sp - half, 4), round(sp + half, 4), wk)


def strat_solo_rank_prod(sess, art_dir, rng):
    """P1 ABLATION: everything measured on the TARGET workload, but options ranked by their solo gain and
    packed greedily by gain per review-day; the shipped set is then measured, so only R1/R3 can fail."""
    p = sess.w.p
    b = _bench(sess, [], "prod", 2)
    solo = {o: b - _bench(sess, [o], "prod", 2) for o in OPTS}
    cur, tot = [], 0
    for o in sorted(OPTS, key=lambda o: -solo[o]):  # P1 ablation ranks by measured solo gain
        if solo[o] > 0 and tot + p["cost"][o] <= p["cap"]:
            cur.append(o); tot += p["cost"][o]
    f = _bench(sess, cur, "prod")
    sp = b / f
    half = max(0.004, 1.5 * sp * p["sig"])
    wk = min(cur, key=lambda o: solo[o]) if cur else None
    _write(art_dir, cur, round(sp - half, 4), round(sp + half, 4), wk)


def strat_solo_rank_dev(sess, art_dir, rng):
    """Both ablations at once, plus the prediction made by multiplying solo speedups: the plan a competent
    engineer writes in ten minutes."""
    p = sess.w.p
    b = _bench(sess, [], "dev", 2)
    solo = {o: b - _bench(sess, [o], "dev", 2) for o in OPTS}
    cur, tot = [], 0
    for o in sorted(OPTS, key=lambda o: -solo[o]):  # P1 ablation ranks by measured solo gain
        if solo[o] > 0 and tot + p["cost"][o] <= p["cap"]:
            cur.append(o); tot += p["cost"][o]
    sp = 1.0
    for o in cur:
        sp *= b / max(1e-9, b - solo[o])
    half = max(0.004, 1.2 * sp * p["sig"])
    wk = min(cur, key=lambda o: solo[o]) if cur else None
    _write(art_dir, cur, round(sp - half, 4), round(sp + half, 4), wk)


def strat_take_all_affordable(sess, art_dir, rng):
    p = sess.w.p
    cur, tot = [], 0
    for o in sorted(OPTS, key=lambda o: p["cost"][o]):
        if tot + p["cost"][o] <= p["cap"]:
            cur.append(o); tot += p["cost"][o]
    b = _bench(sess, [], "prod")
    f = _bench(sess, cur, "prod")
    sp = b / f
    _write(art_dir, cur, round(sp * 0.995, 4), round(sp * 1.005, 4), cur[0] if cur else None)


def strat_wide(sess, art_dir, rng):
    """Correct set, hedged interval: must be rejected by the width cap, never by coverage."""
    p = sess.w.p
    S, sp = best_set(p, "prod")
    w, _ = weakest_member(p, S, "prod")
    _write(art_dir, S, round(sp * 0.85, 4), round(sp * 1.15, 4), w)


def strat_right_set_wrong_weakest(sess, art_dir, rng):
    """Isolates R3: the optimal set and an honest interval, but the within-set ranking read off solo gains
    instead of in-context removals."""
    p = sess.w.p
    S, sp = best_set(p, "prod")
    b = latency(p, [], "prod")
    solo = {o: b - latency(p, [o], "prod") for o in S}
    _write(art_dir, S, round(sp * 0.99, 4), round(sp * 1.01, 4), min(solo, key=lambda o: solo[o]))


def strat_nop(sess, art_dir, rng):
    pass


def strat_bad_json(sess, art_dir, rng):
    open(os.path.join(art_dir, "plan.json"), "w").write("{oops")


def strat_prod_enumerate(sess, art_dir, rng):
    """THE STRATEGY THAT KILLED THIS FAMILY.  No insight: benchmark every feasible set on the target workload
    at one repetition, take the argmax, and recover the leave-one-out ranking from the subset measurements
    already in hand.  Declared `fail`, and it passes - which is why a_stack is cut and `search_gate` exists.

    Keep it here.  Running `python3 -m l15.gates a_stack 64` must now REJECT, and that rejection is the
    regression test for the gate."""
    p = sess.w.p
    F = [S for S in feasible_sets(p) if S]
    prod = {}
    for S in sorted(F, key=lambda X: -len(X)):
        if sess.left() < 3 * PROD_COST:
            break
        prod[tuple(S)] = _bench(sess, S, "prod", 1)
    win = list(min(prod, key=lambda k: prod[k]))
    rem = {}
    for o in win:                                   # every size-(|win|-1) subset is already measured
        k = tuple(sorted(x for x in win if x != o))
        if k in prod:
            rem[o] = prod[k]
    wk = min(rem, key=lambda o: rem[o]) if len(rem) == len(win) else win[0]
    reps = 2 if sess.left() >= 4 * PROD_COST else 1
    b = _bench(sess, [], "prod", reps)
    L = _bench(sess, win, "prod", reps)
    sp = b / L
    half = max(0.004, 2.6 * sp * p["sig"] * math.sqrt(2.0 / reps))
    _write(art_dir, win, round(sp - half, 4), round(sp + half, 4), wk)


STRATEGIES = {"oracle": (strat_oracle, "pass"),
              "prod_enumerate": (strat_prod_enumerate, "fail"),   # passes 18/18 -> family cut
              "dev_only": (strat_dev_only, "fail"),
              "solo_rank_prod": (strat_solo_rank_prod, "fail"),
              "solo_rank_dev": (strat_solo_rank_dev, "fail"),
              "take_all_affordable": (strat_take_all_affordable, "fail"),
              "right_set_wrong_weakest": (strat_right_set_wrong_weakest, "fail"),
              "wide": (strat_wide, "fail"),
              "nop": (strat_nop, "fail"),
              "bad_json": (strat_bad_json, "fail")}
NOISY_FAIL = ("take_all_affordable",)
SEARCH = ["prod_enumerate"]
PRINCIPLES = {"P1_marginal_in_context": "solo_rank_prod", "P2_rank_on_target": "dev_only"}


def instance_gate(p):
    t = truth(p)
    S_star, sp_star = t["best_set"], t["best_speedup"]
    gain = sp_star - 1.0
    b = latency(p, [], "prod")
    solo = {o: b - latency(p, [o], "prod") for o in OPTS}
    cur, tot = [], 0
    for o in sorted(OPTS, key=lambda o: -solo[o]):  # P1 ablation ranks by measured solo gain
        if solo[o] > 0 and tot + p["cost"][o] <= p["cap"]:
            cur.append(o); tot += p["cost"][o]
    r_solo = (speedup(p, cur, "prod") - 1) / max(1e-9, gain)
    naive = 1.0
    for o in cur:
        naive *= b / max(1e-9, b - solo[o])
    pred_err = abs(naive - speedup(p, cur, "prod")) / speedup(p, cur, "prod")
    solo_in_star = {o: solo[o] for o in S_star}
    solo_weak = min(solo_in_star, key=lambda o: solo_in_star[o]) if S_star else None
    # Which ablation bites WHICH item is the point: P2 (rank on the proxy) mis-picks the SET, so it is
    # screened on R1; P1 (solo gains instead of in-context marginals) picks a defensible set but gets the
    # PREDICTION and the WITHIN-SET ranking wrong, so it is screened on R2 and R3.  An earlier version also
    # demanded that P1 mis-pick the set (c4 below) - that held on only 14% of seeds and was screening out
    # exactly the instances where P1's failure is non-salient, i.e. the interesting ones.
    c = [gain >= 0.30,                                             # 1 there is a real decision to make
         len(S_star) < len(OPTS) and len(S_star) >= 3,             # 2 the cap binds and R3 is non-trivial
         t["dev_set_on_prod_gain_ratio"] <= TAU_GAIN - 0.05,       # 3 P2 bites on R1
         t["weak_margin"] >= 4.0 * WEAK_EPS,                       # 4 the weakest member is decidable
         pred_err >= 1.2 * WIDTH_CAP,                              # 5 P1 bites on R2
         solo_weak != t["weakest"],                                # 6 P1 bites on R3
         t["dev_rank_of_prod_optimum"] <= SHORTLIST - 2,   # 2 ranks of slack for dev-side noise               # 7 SOLVABLE: a dev shortlist keeps it
         22 <= t["n_feasible"] <= 36]                              # 8 dev screen fits, prod enum does not
    info = dict(t)
    info.update({"solo_prod_set": cur, "solo_prod_ratio": round(r_solo, 4), "solo_weak": solo_weak,
                 "naive_pred_err": round(pred_err, 4), "c": [bool(x) for x in c]})
    return bool(all(c)), info


def instruction(p, t):
    cat = "\n".join("| `%s` | %d |" % (o, p["cost"][o]) for o in OPTS)
    return """# Serving-stack optimization plan

You own the inference serving stack for a production LLM endpoint.  Six changes are queued in the runtime
vendor's release branch.  Each costs review-days out of a fixed quarterly budget: you may ship any subset
whose total review cost is at most **%d review-days**.

| change | review cost (days) |
|---|---|
%s

The vendor publishes no per-change performance data: the changelog gives the ID and the review cost and
nothing else.  Anything you want to know about what a change does, you measure.

You have **%g bench units**.  `lab bench` measures end-to-end latency for any subset you name, on either
workload:

* `workload=dev` - the staging replica.  **%g unit per repeat.**
* `workload=prod` - a replay of the production traffic mix.  **%g units per repeat.**

The two workloads run the same binary on the same hardware; what differs is the traffic they serve, and
the traffic shape of the production replay is not published.

The stock stack (no options) serves the production mix at roughly **%.0f ms** per request and the dev
replica at roughly **%.0f ms**.  A single repeat carries **%.1f%% relative noise** (independent across
repeats, so averaging %d of them halves it); the noise is measurement noise only - the underlying latency of
a given set on a given workload is fixed.

Write `/app/plan.json`:

```json
{"opts": ["<option>", "..."],
 "speedup": {"lo": 1.23, "hi": 1.27},
 "weakest": "<option>"}
```

* `opts` - the change IDs you would ship; total review cost within the cap.
* `speedup` - an interval for the **production** speedup of the exact set you shipped: stock-stack
  production latency divided by your set's production latency.  Relative width may not exceed %.0f%% of
  your own point estimate.
* `weakest` - which single member of *your own* shipped set contributes least to the production speedup,
  i.e. the one you would drop first if a review-day were taken away.

You are scored on the production mix.
""" % (p["cap"], cat, BUDGET, DEV_COST, PROD_COST, t["base_prod"], t["base_dev"],
       100 * p["sig"], MAX_REPS, 100 * WIDTH_CAP)


def docs(p):
    return {"docs/plan_format.md": """# /app/plan.json

```json
{"opts": ["C-114", "C-352"],
 "speedup": {"lo": 1.31, "hi": 1.36},
 "weakest": "C-352"}
```

* `opts` (list of strings) - catalogue change IDs, no duplicates, total review cost <= the cap in `lab spec`.
* `speedup.lo`, `speedup.hi` (numbers) - interval for `stock_prod_latency / your_set_prod_latency`.
  `(hi - lo) / point` must be <= %.2f.
* `weakest` (string) - the member of `opts` whose removal would cost the least production speedup.

Nothing else in the file is read.  The verifier evaluates your set against the simulator; it does not read
your lab log, and partial credit does not exist.
""" % WIDTH_CAP,
            "docs/stack.md": """# What the runtime looks like from the inside

Enough of the internals are public to tell you the *shape* of the problem; none of the numbers are.

* A request passes through a handful of internal stages, and end-to-end latency is the **sum** of the time
  spent in each stage.  The per-stage breakdown is not exported by any counter you can read - `bench`
  returns end-to-end latency and nothing else.
* Each queued change **divides** the time spent in one or two stages by some factor, and adds a small
  fixed amount of work to some other stage.  Both the stages involved and the factors are unpublished, and
  they are properties of the change: they are the same on the staging replica as in production.
* How much of a request's time lands in each stage is a property of the **workload**, not of the changes.
  The staging replica and the production replay have different traffic, so they have different breakdowns.
  Neither breakdown is published.

Two consequences worth writing down before you spend anything:

1. Two changes that divide the *same* stage do not add up - the second one is working on time the first
   one already removed.  Two changes that divide *different* stages very nearly do add up.
2. A change's value depends on the workload's breakdown, so a ranking produced on the staging replica is a
   ranking of a different quantity than the one you are scored on.

Operational note: the review board rejected the last two quarterly plans because the shipped speedup did
not match the predicted one.  Both plans had predicted the result by combining the numbers each change
scored on its own.
"""}


def hints(p):
    return {1: """
Hint 1: an option's effect measured against the stock stack is not its effect inside a set that already
contains other options.
""", 2: """
Hint 2: the staging replica and the production mix spend their time in different stages, so the same
option is worth different amounts on each.  You are scored on production.
"""}


def pool_gate(seeds):
    """v7 gate 6: no single fixed plan may satisfy R1 on more than one admitted instance."""
    ps = [sample_params(s) for s in seeds]
    ps = [p for p in ps if instance_gate(p)[0]]
    if len(ps) < 2:
        return True, {"n": len(ps)}
    cand = set()
    for p in ps:
        for S in feasible_sets(p):
            cand.add(tuple(sorted(S)))
    best, arg = 0, None
    for S in cand:
        k = 0
        for p in ps:
            if sum(p["cost"][o] for o in S) > p["cap"]:
                continue
            if (speedup(p, list(S), "prod") - 1) / max(1e-9, truth(p)["best_speedup"] - 1) >= TAU_GAIN:
                k += 1
        if k > best:
            best, arg = k, S
    return best <= 1, {"n": len(ps), "best_fixed": best, "arg": arg}
