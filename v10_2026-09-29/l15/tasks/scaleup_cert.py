"""Certify scaleup instances (resumable; every number is measured with the grader's own code path).

  python3 -m l15.tasks.scaleup_cert <time_limit_s> <n_proc> <seed> [<seed> ...]

Per seed:
  1. team sweep at sample scale (lr x wd, seed 0, S_VIS steps) -> dev accuracy; config.json = best row.
  2. production scores (scaleup.production_score = the grader) of
       ship_as_is     the team config                                   (decoy: ship what dev chose)
       retune_sample  best dev config of a 24-point sample-scale search over lr, wd, aug, smoothing, width
                      (decoy: the standard hygiene, done more thoroughly than the team)
       scale_rule     the team config with wd * N_VIS / N_FULL          (reference: the published rule)
       curve_wd*      team lr, wd in {0.03, 0.1, 0.3, 1.0}              (information only)
  3. T = midpoint of reference and the best decoy, floored to 0.005;  ok iff both margins >= MARGIN.
Results cached in $CERT_CACHE (default /tmp/v10/scaleup/cert_cache_e<EPOCHS>.json); the final summary is written to
l15/tasks/scaleup_cert.json (read by scaleup.sample_params / instance_gate)."""
import json, math, os, shutil, sys, tempfile, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
from . import scaleup as S

CACHE = os.environ.get("CERT_CACHE", "/tmp/v10/scaleup/cert_cache_e%d.json" % S.EPOCHS)
sys.path.insert(0, S.REPO_SRC)
import train as TR  # noqa: E402  (the repo's own code: sample-scale runs use it in-process)


def _key(*a):
    return json.dumps(a, sort_keys=True)


def sweep_job(seed, cfg):
    p = S.base_params(seed)
    dd = S.data(p)
    Xs, ys = dd["sample"]
    P, norm = TR.train(Xs, ys, S.S_VIS, cfg, seed=0)
    Xd, yd = dd["dev"]
    return float((TR.predict(P, norm, Xd) == yd).mean())


def prod_rep_job(seed, cfg, team, rep):
    """One production replica (grader code path: scaleup.run_job with the grader's data files)."""
    p = S.base_params(seed)
    p.update({"team_lr": team["lr"], "team_wd": team["weight_decay"], "T": 0.0, "job_cpu_s": 0.0})
    repo = tempfile.mkdtemp(prefix="cert_repo_"); work = tempfile.mkdtemp(prefix="cert_data_")
    try:
        shutil.copy(os.path.join(S.REPO_SRC, "train.py"), repo)
        json.dump(cfg, open(os.path.join(repo, "config.json"), "w"))
        S._write_data(work, p)
        acc, info = S.run_job(repo, p, rep, data_dir=work)
        return {"acc": acc, "cpu_s": info.get("cpu_s"), "err": info.get("error")}
    finally:
        shutil.rmtree(repo, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)


def prod_result(seed, cfg, cache):
    """Aggregate exactly as scaleup.production_score: mean over replicas 0,1; None if any replica fails.
    Falls back to a whole-job cache entry written by the earlier (non-replica) version of this script."""
    k = _key("prod", seed, cfg)
    if k in cache:
        return cache[k], []
    reps, pend = [], []
    for r in (0, 1):
        kr = _key("prodrep", seed, cfg, r)
        if kr in cache: reps.append(cache[kr])
        else: pend.append((kr, r))
    if pend:
        return None, pend
    accs = [x["acc"] for x in reps]
    return {"acc": None if any(a is None for a in accs) else float(np.mean(accs)),
            "cpu_s": [x["cpu_s"] for x in reps], "per_seed": accs, "err": [x["err"] for x in reps if x["err"]]}, []


def team_grid():
    out = []
    for lr in S.TEAM_LR_GRID:
        for wd in S.TEAM_WD_GRID:
            c = dict(S.BASE_CFG); c.update({"lr": lr, "weight_decay": wd}); out.append(c)
    return out


def search_grid(seed):
    r = np.random.default_rng([seed, 99])
    out = []
    for _ in range(24):
        c = dict(S.BASE_CFG)
        c.update({"lr": float(r.choice([0.001, 0.003, 0.01])), "weight_decay": float(r.choice([0.3, 1.0, 2.0, 3.0, 5.0])),
                  "aug_sigma": float(r.choice([0.0, 0.1, 0.3])), "label_smoothing": float(r.choice([0.0, 0.1])),
                  "hidden": int(r.choice([128, 256, 384]))})
        out.append(c)
    return out


def best(rows):
    return max(rows, key=lambda r: (r[1], -rows.index(r)))  # highest dev acc, earliest on ties


def plan(seed, cache):
    """-> list of pending (key, fn, args); the summary if everything is cached."""
    pend = []
    tg = team_grid()
    sw = []
    for c in tg:
        k = _key("sweep", seed, c)
        if k in cache: sw.append((c, cache[k]))
        else: pend.append((k, sweep_job, (seed, c)))
    sg = search_grid(seed)
    ss = []
    for c in sg:
        k = _key("sweep", seed, c)
        if k in cache: ss.append((c, cache[k]))
        else: pend.append((k, sweep_job, (seed, c)))
    if pend:
        return pend, None
    team = best(sw)[0]
    retune = best(ss + [(team, best(sw)[1])])[0]   # the search also sees the team's own best row
    cands = {"ship_as_is": team, "retune_sample": retune}
    ref = dict(team); ref["weight_decay"] = team["weight_decay"] * S.N_VIS / S.N_FULL
    cands["scale_rule"] = ref
    for wd in (0.03, 0.1, 0.3, 1.0):
        c = dict(team); c["weight_decay"] = wd; cands["curve_wd%g" % wd] = c
    res = {}
    for name, c in cands.items():
        r, pr = prod_result(seed, c, cache)
        if r is not None: res[name] = dict(r, cfg=c)
        for kr, rep in pr:
            if kr not in [q[0] for q in pend]: pend.append((kr, prod_rep_job, (seed, c, team, rep)))
    if pend:
        return pend, None
    decoys = {k: res[k] for k in ("ship_as_is", "retune_sample")}
    ok_runs = all(v["acc"] is not None for v in res.values())
    summ = {"team": team, "team_sweep": [dict(lr=c["lr"], weight_decay=c["weight_decay"], dev_acc=a) for c, a in sw],
            "retune_search": [dict(c, dev_acc=a) for c, a in ss], "ref": res["scale_rule"], "decoys": decoys,
            "curve": {k: res[k] for k in res if k.startswith("curve")},
            "job_cpu_s": float(np.mean(res["ship_as_is"]["cpu_s"])), "margin": S.MARGIN}
    if ok_runs:
        top = max(v["acc"] for v in decoys.values())
        T = math.floor(((res["scale_rule"]["acc"] + top) / 2) / 0.005) * 0.005
        summ.update({"T": round(T, 3), "ref_margin": round(res["scale_rule"]["acc"] - T, 5),
                     "decoy_margin": round(T - top, 5)})
        summ["ok"] = bool(summ["ref_margin"] >= S.MARGIN and summ["decoy_margin"] >= S.MARGIN)
    else:
        summ["ok"] = False
    return [], summ


def main():
    tl, npr = float(sys.argv[1]), int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3:]]
    t0 = time.time()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    with Pool(npr) as pool:
        while time.time() - t0 < tl:
            todo, summaries = [], {}
            for s in seeds:
                pend, summ = plan(s, cache)
                todo += pend
                if summ is not None:
                    summaries[s] = summ
            if not todo:
                break
            # production replicas are ~1 min each: submit only what can plausibly finish in time
            futs = [(k, pool.apply_async(fn, args)) for k, fn, args in todo[:max(npr, 1) * (8 if todo[0][1] is sweep_job else 2)]]
            for k, f in futs:
                rem = tl - (time.time() - t0)
                if rem <= 0: break
                try:
                    cache[k] = f.get(timeout=rem)
                    tmp = CACHE + ".tmp"; json.dump(cache, open(tmp, "w")); os.replace(tmp, CACHE)
                except Exception as ex:
                    print("job %s after %.0fs: %s" % ("timeout" if isinstance(ex, __import__("multiprocessing").TimeoutError)
                                                      else "error", time.time() - t0, repr(ex)[:300]), k[:90], flush=True)
                    break
        pool.terminate()
    allc = json.load(open(S.CERT_PATH)) if os.path.exists(S.CERT_PATH) else {}
    for s in seeds:
        pend, summ = plan(s, cache)
        if summ is not None:
            allc[str(s)] = summ
            print("seed %d: T=%s ok=%s ref=%.4f decoys=%s curve=%s" % (
                s, summ.get("T"), summ.get("ok"), summ["ref"]["acc"] or -1,
                {k: round(v["acc"], 4) for k, v in summ["decoys"].items() if v["acc"] is not None},
                {k: round(v["acc"], 4) for k, v in summ["curve"].items() if v["acc"] is not None}))
        else:
            print("seed %d: %d jobs pending" % (s, len(pend)))
    tmp = S.CERT_PATH + ".tmp"; json.dump(allc, open(tmp, "w"), indent=1); os.replace(tmp, S.CERT_PATH)


if __name__ == "__main__":
    main()
