"""Resumable, parallel driver for `v8gates.certify`.

    python3 -m l15.v8cert <family> <lo> <hi> <target> <state.json> [time_limit_s]

Same checks, same verdict, same report keys as `v8gates.certify`; two differences, both operational:

  * RESUMABLE.  Every unit of work (screen one seed, strategy-gate one seed, one p_ablation / search /
    mutation / item-activity job) is written to `state.json` as soon as it finishes, and the process stops
    cleanly once `time_limit_s` is used up.  Re-running the same command continues where it stopped.  Needed
    because a certificate over a few hundred seeds takes several minutes and our sandbox kills any shell call
    after ~2 minutes (and kills background children with it).
  * PARALLEL and MEMOISED.  Seeds are independent, so they are farmed out to a process pool; and
    `sample_params` is cached per worker, because some families (a_span) construct each instance by a search
    that costs ~0.1 s and the gates call it a dozen times per seed.  The cache hands out deep copies, so a
    strategy that mutated its params could not contaminate another run.

It prints `INCOMPLETE <stage>` while there is work left and the final report (and `OK` / `NOT OK`) once there
is none.  Nothing in here decides anything: every verdict comes from `gates.gate` and the `v8gates`
functions, unchanged.
"""
import copy, importlib, json, math, multiprocessing as mp, os, sys, time

from . import gates as G
from . import v8gates as V

_MODS = {}


def _mod(name):
    if name not in _MODS:
        mod = importlib.import_module("l15.tasks." + name)
        raw, cache = mod.sample_params, {}

        def sample_params(seed, _raw=raw, _cache=cache):
            if seed not in _cache:
                _cache[seed] = _raw(seed)
            return copy.deepcopy(_cache[seed])

        mod.sample_params = sample_params
        _MODS[name] = mod
    return _MODS[name]


def _screen(a):
    name, s = a
    mod = _mod(name)
    return s, bool(mod.instance_gate(mod.sample_params(s))[0])


def _gate(a):
    name, s, n_salts = a
    r = G.gate(_mod(name), s, n_salts=n_salts, verbose=False)
    return s, {"accept": bool(r.get("accept")), "bad": r.get("bad")}


def _post(a):
    name, kind, s, n = a
    mod = _mod(name)
    if kind == "pabl":
        return kind, s, V.p_ablation(mod, s, n_salts=n)
    if kind == "search":
        return kind, s, V.search_gate(mod, s)
    if kind == "mut":
        return kind, s, V.mutation_test(mod, s)
    if kind == "act":
        return kind, s, V.item_activity(mod, [s], n_salts=n)
    raise ValueError(kind)


def _merge_activity(tabs):
    tab = {}
    for t in tabs:
        for item, d in t.items():
            e = tab.setdefault(item, {"n": 0, "pass": 0, "oracle_n": 0, "oracle_pass": 0})
            for k in e:
                e[k] += d[k]
    for d in tab.values():
        d["pass_rate"] = round(d["pass"] / max(1, d["n"]), 3)
        d["oracle_rate"] = round(d["oracle_pass"] / max(1, d["oracle_n"]), 3)
        d["inert"] = bool(d["pass"] == d["n"])
        d["broken"] = bool(d["oracle_n"] and d["oracle_pass"] == 0)
    return tab


def run(name, lo, hi, target, state, tlimit=95.0, n_salts=4, act_salts=2, procs=4, clean_factor=2.5):
    t0 = time.time()
    st = json.load(open(state)) if os.path.exists(state) else {
        "family": name, "lo": lo, "hi": hi, "target": target, "screen": {}, "gate": {}, "post": {}}

    def save():
        tmp = state + ".tmp"
        json.dump(st, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, state)

    left = lambda: tlimit - (time.time() - t0)
    pool = mp.Pool(procs)
    try:
        # 1. screen every candidate
        todo = [s for s in range(lo, hi) if str(s) not in st["screen"]]
        while todo and left() > 0:
            batch, todo = todo[:procs * 4], todo[procs * 4:]
            for s, ok in pool.map(_screen, [(name, s) for s in batch]):
                st["screen"][str(s)] = ok
            save()
        if todo:
            print("INCOMPLETE screen: %d seeds left" % len(todo))
            return None
        screened = sorted(int(s) for s, v in st["screen"].items() if v)

        # 2. strategy gate, in seed order, until there are comfortably more clean seeds than the pool needs
        def clean():
            return sorted(int(s) for s, v in st["gate"].items() if v["accept"])
        todo = [s for s in screened if str(s) not in st["gate"]]
        while todo and left() > 0 and len(clean()) < math.ceil(clean_factor * target):
            batch, todo = todo[:procs], todo[procs:]
            for s, r in pool.map(_gate, [(name, s, n_salts) for s in batch]):
                st["gate"][str(s)] = r
            save()
        if todo and len(clean()) < math.ceil(clean_factor * target):
            print("INCOMPLETE gate: %d clean so far, %d screened seeds not yet gated" % (len(clean()), len(todo)))
            return None

        # 3. pool
        mod = _mod(name)
        if "pool" not in st:
            p, info = V.select_pool(mod, clean(), target)
            st["pool"], st["pool_info"] = p, info
            save()
        P = st["pool"]
        n_abl = max(3, len(P) // 3) if P else 0
        jobs = ([("pabl", s, n_salts) for s in P[:n_abl]] + [("search", s, 0) for s in P[:n_abl]]
                + [("mut", s, 0) for s in P[:2]] + [("act", s, act_salts) for s in P[:max(2, len(P) // 4)]])
        todo = [j for j in jobs if "%s:%d" % (j[0], j[1]) not in st["post"]]
        while todo and left() > 0:
            batch, todo = todo[:procs], todo[procs:]
            for kind, s, r in pool.map(_post, [(name,) + j for j in batch]):
                st["post"]["%s:%d" % (kind, s)] = r
            save()
        if todo:
            print("INCOMPLETE post: %d jobs left" % len(todo))
            return None
    finally:
        pool.terminate()

    # 4. assemble exactly the report `certify` would have written
    post = st["post"]
    get = lambda kind, seeds: {s: post["%s:%d" % (kind, s)] for s in seeds}
    rep = {"family": mod.World.NAME, "n_candidates": hi - lo, "n_screened": len(screened),
           "n_strategy_gate_clean": len(clean()),
           "n_strategy_gate_run": len(st["gate"]),
           "strategy_gate_rejected": {s: v["bad"] for s, v in st["gate"].items() if not v["accept"]},
           "admitted": P, "pool_gate": st["pool_info"]}
    rep["p_ablation"] = get("pabl", P[:n_abl])
    rep["p_ablation_ok"] = bool(rep["p_ablation"]) and all(v["ok"] for v in rep["p_ablation"].values())
    rep["item_activity"] = _merge_activity(post["act:%d" % s] for s in P[:max(2, len(P) // 4)])
    rep["items_inert"] = sorted(k for k, v in rep["item_activity"].items() if v["inert"])
    rep["items_broken"] = sorted(k for k, v in rep["item_activity"].items() if v["broken"])
    rep["search"] = get("search", P[:n_abl])
    rep["search_ok"] = bool(rep["search"]) and all(v["ok"] for v in rep["search"].values())
    rep["mutation"] = get("mut", P[:2])
    rep["mutation_ok"] = all((v["ok"] is not False) for v in rep["mutation"].values())
    rep["mutation_checked"] = all(v["checked"] for v in rep["mutation"].values())
    rep["ok"] = bool(P and len(P) >= target and rep["p_ablation_ok"] and rep["search_ok"]
                     and not rep["items_inert"] and not rep["items_broken"] and rep["mutation_ok"])
    return rep


def _print(r):
    print(json.dumps({k: v for k, v in r.items() if k not in ("p_ablation", "item_activity", "mutation", "search")},
                     indent=1, default=str))
    for s, v in r["p_ablation"].items():
        for pn, d in v["principles"].items():
            print("  ablate %-32s seed %-5s pass %d/%d on-target %d/%d %s"
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


if __name__ == "__main__":
    a = sys.argv[1:]
    r = run(a[0], int(a[1]), int(a[2]), int(a[3]), a[4], float(a[5]) if len(a) > 5 else 95.0)
    if r is not None:
        out = a[4].replace(".state.json", ".json") if a[4].endswith(".state.json") else a[4] + ".report.json"
        json.dump(r, open(out, "w"), indent=1, default=str)
        _print(r)
