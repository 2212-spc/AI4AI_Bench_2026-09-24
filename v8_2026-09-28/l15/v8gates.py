"""v8 gates: the checks that are NOT "does one strategy pass or fail".

`l15/gates.py` (inherited unchanged from v7) answers one question per instance: does the reference solution
pass and does every scripted decoy fail?  That is necessary and nowhere near sufficient.  Three ways an
instance can clear that bar and still be worthless, one gate each:

  1. A FAILING DECOY IS NOT A DIFFICULTY CERTIFICATE.  A decoy can fail because it wrote malformed JSON or
     ran out of budget, which says nothing about the principle it was supposed to lack.  `p_ablation` pins
     each decoy to a named principle AND to the rubric items that principle is responsible for, then
     requires the failure to land there.  v7 gate 9: the identity of a decoy is *which item* it fails.

  2. A RUBRIC ITEM CAN BE INERT.  An item every strategy passes is decoration; an item the reference fails
     is a bug.  `item_activity` reports the per-item pass rate across all strategies and salts, and
     separately the reference's own rate.

  3. A GRADER CAN BE UNFALSIFIABLE.  `mutation_test` perturbs the ANSWER - not the grader - starting from a
     real passing run, and requires the verdict to survive a semantically neutral rewrite and to flip on a
     clearly-wrong value.  Both directions matter: v4 shipped two graders that would have REJECTED correct
     answers, which no amount of decoy-fails-as-expected testing would have caught.

Plus `select_pool`, because a pool-level gate that can only be *checked* is useless - the builder has to be
able to *construct* a pool satisfying it.  Plain greedy insertion gets stuck (an early pick can make every
remaining candidate illegal, which is exactly what happened to b_control's modal-answer cap: greedy
returned 3 seeds), so this is randomised-restart greedy keeping the largest pool found.

CORRECTNESS AND DIFFICULTY ARE CERTIFIED SEPARATELY AND NEVER INFERRED FROM EACH OTHER.  Selecting tasks
because frontier models fail them enriches for broken tasks (HLE 29%, SWE-V 59.4% contested); the
counter-measure is that `certify` reports the two claims in different fields and requires both.
"""
import json, math, os, random, shutil, sys, tempfile
import numpy as np

from . import gates as _g


# ------------------------------------------------------------------ shared plumbing
def _world(mod, params, salt):
    """Build the same World `run_strategy` would, including any per-instance budget."""
    bud = mod.BUDGET
    if hasattr(mod, "instance_budget"):
        _, info = mod.instance_gate(params)
        t = dict(info)
        if hasattr(mod, "instance_truth"):
            t.update(mod.instance_truth({"params": params, "salt": salt, "truth": t}))
        bud = mod.instance_budget(params, t)
    return mod.World({"task": mod.__name__.split(".")[-1], "params": params, "salt": salt, "budget": bud})


def _ledger(art_dir):
    p = os.path.join(art_dir, "_ledger.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def _declared(mod, pname):
    """PRINCIPLES values may be `"strategy"` or `("strategy", ["R1_x", ...])`.

    The second form is what makes p_ablation a real gate; the bare form is accepted so an old family still
    runs, but it is reported with `items_declared: false` so the weaker check is visible in the record.
    """
    v = mod.PRINCIPLES[pname]
    if isinstance(v, str):
        return v, None
    return v[0], list(v[1])


# ------------------------------------------------------------------ 1. principle ablation
def p_ablation(mod, seed, n_salts=4):
    """For each named principle: its ablation must fail, and fail ON the items it is supposed to protect.

    `min_rate` is 0.5 rather than 1.0 because the ablation still runs through the noisy lab - on some salts
    a decoy that is wrong about R2 gets a lucky interval - but it must land on the declared item at least
    half the time, and it must never pass overall.
    """
    params = mod.sample_params(seed)
    ok0, _ = mod.instance_gate(params)
    out = {"seed": seed, "screen_ok": bool(ok0), "principles": {}}
    if not ok0:
        out["ok"] = False
        return out
    allok = True
    for pname in getattr(mod, "PRINCIPLES", {}):
        sname, want = _declared(mod, pname)
        runs = [_g.run_strategy(mod, params, "pabl-%d-%d" % (seed, k), sname) for k in range(n_salts)]
        npass = sum(r["pass"] for r in runs)
        failed = [sorted(k for k, v in r["items"].items() if not v["ok"]) for r in runs]
        on_target = sum(1 for f in failed if (any(k in f for k in want) if want
                                             else any(not k.startswith("R0") for k in f)))
        ok = (npass == 0) and (on_target >= math.ceil(0.5 * n_salts))
        allok = allok and ok
        out["principles"][pname] = {"strategy": sname, "items_declared": want is not None,
                                    "protects": want, "n": n_salts, "n_pass": npass,
                                    "n_fail_on_target": on_target, "failed_items": failed, "ok": bool(ok)}
    out["ok"] = bool(allok and out["principles"])
    return out


# ------------------------------------------------------------------ 2. item activity
def item_activity(mod, seeds, n_salts=2):
    """Per rubric item over the pool: how many (strategy, salt) runs passed it, and how many oracle runs.

    Counts rather than a threshold, because "enough" depends on how many decoys target that item; `certify`
    only asserts the two unambiguous failures - inert (nothing ever fails it) and broken (the reference
    never passes it).
    """
    tab = {}
    for seed in seeds:
        params = mod.sample_params(seed)
        if not mod.instance_gate(params)[0]:
            continue
        for sname in mod.STRATEGIES:
            for k in range(n_salts):
                r = _g.run_strategy(mod, params, "act-%d-%d" % (seed, k), sname)
                for item, v in r["items"].items():
                    d = tab.setdefault(item, {"n": 0, "pass": 0, "oracle_n": 0, "oracle_pass": 0})
                    d["n"] += 1
                    d["pass"] += int(bool(v["ok"]))
                    if sname == "oracle":
                        d["oracle_n"] += 1
                        d["oracle_pass"] += int(bool(v["ok"]))
    for d in tab.values():
        d["pass_rate"] = round(d["pass"] / max(1, d["n"]), 3)
        d["oracle_rate"] = round(d["oracle_pass"] / max(1, d["oracle_n"]), 3)
        d["inert"] = bool(d["pass"] == d["n"])
        d["broken"] = bool(d["oracle_n"] and d["oracle_pass"] == 0)
    return tab


# ------------------------------------------------------------------ 3. grader falsification
def mutation_test(mod, seed, salt="mut", n_try=3):
    """Start from a REAL passing run, mutate the artifact, re-grade with the same ledger.

    `mod.MUTATE = [(name, fn), ...]` with `fn(params, truth, art_dir, leg)`; `leg` is "small" or "large".
      small  - a semantically neutral rewrite: reorder a list, round a number well inside tolerance, add an
               extra key.  The verdict must still be PASS.  (Catches hair-trigger graders that reject
               correct answers - the v6 class of bug.)
      large  - a clearly wrong value: the other label, an interval shifted by several tolerances.  The
               verdict must be FAIL, and `fn` may return the item name it should break, which is then
               required to be among the failures.  (Catches a grader that accepts anything.)
    Re-grading reuses the oracle's own ledger, so items that read process facts out of the ledger keep
    seeing a legitimate session and only the artifact changes.
    """
    muts = getattr(mod, "MUTATE", None)
    if not muts:
        return {"seed": seed, "checked": False, "ok": None,
                "note": "no MUTATE list; grader falsification not attempted"}
    params = mod.sample_params(seed)
    src = None
    for k in range(n_try):                      # the oracle is allowed to miss a salt; retry before giving up
        r = _g.run_strategy(mod, params, "%s-%d" % (salt, k), "oracle", keep=True)
        if r["pass"]:
            src = r
            break
        shutil.rmtree(r.get("art_dir", ""), ignore_errors=True)
    if src is None:
        return {"seed": seed, "checked": False, "ok": False,
                "note": "oracle did not pass in %d tries; cannot seed a mutation" % n_try}
    art0, led = src["art_dir"], _ledger(src["art_dir"])
    _, info = mod.instance_gate(params)
    truth = dict(info)
    if hasattr(mod, "instance_truth"):
        truth.update(mod.instance_truth({"params": params, "salt": src["salt"], "truth": truth}))
    w = _world(mod, params, src["salt"])
    res, allok = {}, True
    for name, fn in muts:
        row = {}
        for leg in ("small", "large"):
            d = tempfile.mkdtemp(prefix="mut_")
            shutil.rmtree(d); shutil.copytree(art0, d)
            want_item = fn(params, truth, d, leg)
            g = w.grade(d, led)
            row[leg] = {"pass": bool(g["pass"]), "want_item": want_item,
                        "failed": sorted(k for k, v in g["items"].items() if not v["ok"])}
            shutil.rmtree(d, ignore_errors=True)
        ok = row["small"]["pass"] and not row["large"]["pass"]
        if ok and row["large"]["want_item"]:
            ok = row["large"]["want_item"] in row["large"]["failed"]
        row["ok"] = bool(ok)
        allok = allok and ok
        res[name] = row
    shutil.rmtree(art0, ignore_errors=True)
    return {"seed": seed, "checked": True, "salt": src["salt"], "ok": bool(allok), "mutations": res}


# ------------------------------------------------------------------ 4. search gate
def search_gate(mod, seed, n_salts=3):
    """THE gate that v7 did not have, and the one that cost the most to learn.

    `a_stack` cleared every v7 gate, eight closed-form instance conditions and a two-principle ablation, and
    was then passed 18/18 by a strategy with no insight in it at all: benchmark every feasible option set on
    the target workload at one repetition, take the argmax, and read the leave-one-out ranking off the
    measurements you already made.  It used 65-88% of the budget.  The answer space was ~25 subsets, a target
    measurement cost 4 of 160 credits, and the noise was far smaller than the gap between the best and
    second-best set - so exhaustive search was both affordable and reliable.

    The lesson is not "make the answer space bigger".  It is that DIFFICULTY LIVES IN WHETHER THE CHEAP
    MEASUREMENT IS BIASED, NOT IN HOW BIG THE SEARCH IS.  Compare `b_control`: an exhaustive sweep there
    (retrain all five single reversions and re-measure) is the skeleton of the reference solution itself, yet
    the sweep still fails, because the naive way to measure each arm estimates the wrong quantity.  In
    `a_stack` the naive measurement of a candidate was unbiased, so sweeping won.

    So: every family must declare `SEARCH`, a list of strategy names that sweep the answer space using the
    most obvious measurement, and every one of them must fail.  A family that declares none is reported as
    unchecked, and `certify` will not pass it.
    """
    names = list(getattr(mod, "SEARCH", ()))
    if not names:
        return {"seed": seed, "checked": False, "ok": False,
                "note": "family declares no SEARCH strategies; searchability is unproven"}
    params = mod.sample_params(seed)
    out, allok = {}, True
    for nm in names:
        runs = [_g.run_strategy(mod, params, "srch-%d-%d" % (seed, k), nm) for k in range(n_salts)]
        npass = sum(r["pass"] for r in runs)
        out[nm] = {"n": n_salts, "n_pass": npass, "spent_frac": [r["spent_frac"] for r in runs],
                   "failed_items": [sorted(k for k, v in r["items"].items() if not v["ok"]) for r in runs],
                   "ok": bool(npass == 0)}
        allok = allok and out[nm]["ok"]
    return {"seed": seed, "checked": True, "ok": bool(allok), "searches": out}


# ------------------------------------------------------------------ 5. pool construction
def select_pool(mod, candidates, target, restarts=80, rng_seed=7):
    """Largest subset of `candidates` (<= target) satisfying `mod.pool_gate`, by randomised-restart greedy.

    Needed because pool gates are not monotone in a way greedy respects.  b_control caps how many pool
    instances may share a culprit label at ceil(n/4); in-order greedy admitted seeds 0,1,2 and then every
    remaining candidate violated the cap, so it returned a pool of 3.  Shuffling the insertion order fixes
    it without the selector needing to know what the gate is checking.
    """
    rng = random.Random(rng_seed)
    cand = [s for s in candidates if mod.instance_gate(mod.sample_params(s))[0]]
    best = []
    for _ in range(restarts):
        order = cand[:]
        rng.shuffle(order)
        pool = []
        for s in order:
            if len(pool) >= target:
                break
            if mod.pool_gate(pool + [s])[0]:
                pool.append(s)
        if len(pool) > len(best):
            best = pool
        if len(best) >= target:
            break
    return sorted(best), mod.pool_gate(sorted(best))[1]


# ------------------------------------------------------------------ 5. one certificate per family
def certify(mod, candidates, target=12, n_salts=4, act_salts=2, out=None, verbose=True):
    rep = {"family": mod.World.NAME, "n_candidates": len(candidates)}
    screened = [s for s in candidates if mod.instance_gate(mod.sample_params(s))[0]]
    rep["n_screened"] = len(screened)
    clean, rejected = [], {}
    for s in screened:
        r = _g.gate(mod, s, n_salts=n_salts, verbose=False)
        if r.get("accept"):
            clean.append(s)
        else:
            rejected[s] = r.get("bad")
    rep["n_strategy_gate_clean"] = len(clean)
    rep["strategy_gate_rejected"] = rejected
    if verbose:
        print("screened %d/%d, strategy-gate clean %d" % (len(screened), len(candidates), len(clean)))
    pool, pinfo = select_pool(mod, clean, target)
    rep["admitted"] = pool
    rep["pool_gate"] = pinfo
    n_abl = max(3, len(pool) // 3) if pool else 0
    rep["p_ablation"] = {s: p_ablation(mod, s, n_salts=n_salts) for s in pool[:n_abl]}
    rep["p_ablation_ok"] = bool(rep["p_ablation"]) and all(v["ok"] for v in rep["p_ablation"].values())
    rep["item_activity"] = item_activity(mod, pool[:max(2, len(pool) // 4)], n_salts=act_salts)
    rep["items_inert"] = sorted(k for k, v in rep["item_activity"].items() if v["inert"])
    rep["items_broken"] = sorted(k for k, v in rep["item_activity"].items() if v["broken"])
    rep["search"] = {s: search_gate(mod, s) for s in pool[:n_abl]}
    rep["search_ok"] = bool(rep["search"]) and all(v["ok"] for v in rep["search"].values())
    rep["mutation"] = {s: mutation_test(mod, s) for s in pool[:2]}
    rep["mutation_ok"] = all((v["ok"] is not False) for v in rep["mutation"].values())
    rep["mutation_checked"] = all(v["checked"] for v in rep["mutation"].values())
    rep["ok"] = bool(pool and rep["p_ablation_ok"] and rep["search_ok"] and not rep["items_inert"]
                     and not rep["items_broken"] and rep["mutation_ok"])
    if out:
        json.dump(rep, open(out, "w"), indent=1, default=str)
    return rep


if __name__ == "__main__":
    import importlib
    a = sys.argv[1:]
    mod = importlib.import_module("l15.tasks." + a[0])
    lo, hi = (int(a[1]), int(a[2])) if len(a) > 2 else (0, 400)
    tgt = int(a[3]) if len(a) > 3 else 12
    r = certify(mod, list(range(lo, hi)), target=tgt, out=(a[4] if len(a) > 4 else None))
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("p_ablation", "item_activity", "mutation")}, indent=1, default=str))
    for s, v in r["p_ablation"].items():
        for pn, d in v["principles"].items():
            print("  ablate %-26s seed %-5s pass %d/%d on-target %d/%d %s"
                  % (pn, s, d["n_pass"], d["n"], d["n_fail_on_target"], d["n"], "OK" if d["ok"] else "BAD"))
    for k, v in sorted(r["item_activity"].items()):
        print("  item %-28s pass %3d/%-3d oracle %.2f %s%s"
              % (k, v["pass"], v["n"], v["oracle_rate"], "INERT" if v["inert"] else "", "BROKEN" if v["broken"] else ""))
    for s, v in r["search"].items():
        print("  search   seed %-5s checked=%s ok=%s %s" % (s, v["checked"], v["ok"],
              {k: (d["n_pass"], d["n"], d["spent_frac"][:1]) for k, d in v.get("searches", {}).items()}
              or v.get("note", "")))
    for s, v in r["mutation"].items():
        print("  mutation seed %s checked=%s ok=%s %s" % (s, v["checked"], v["ok"],
              {k: (d["small"]["pass"], d["large"]["pass"]) for k, d in v.get("mutations", {}).items()}))
    print("OK" if r["ok"] else "NOT OK")
