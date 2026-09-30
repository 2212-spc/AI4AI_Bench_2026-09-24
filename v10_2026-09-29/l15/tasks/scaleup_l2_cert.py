"""Certify scaleup L2 instances (resumable; every production number is the grader's own code path).

  python3 -m l15.tasks.scaleup_l2_cert <time_limit_s> <n_proc> <phase> <seed> [<seed> ...]
      phase: "sample" = sample-scale jobs only (cheap; safe to run next to other load)
             "all"    = also production replicas (run with the machine otherwise idle: the grader's wall cap)

Per seed (world = scaleup_l2.base_params(seed): the duplicated corpus):
  team            best dev row of the team's lr x wd sweep on the L2 sample (same grid as L1) -> config.json
  DECOYS (must miss T by >= MARGIN)
    ship_as_is          the team config
    retune_sample       best dev config of the 24-point sample-scale search (L1's decoy)
    row_rule            team wd * N_VIS / N_FULL                   (L1's REFERENCE: row count is the wrong N)
    l1_fable_injob      Fable's L1 repo verbatim (in-job wd selection on a random 5% slice of the corpus)
  REFERENCE (must clear T by >= MARGIN) - "replicate the production regime at sample scale":
    k_hat = 1 + pairs (N-1) / C(n,2) from the sample's exact-duplicate pairs; production = N/k_hat distinct
    rows seen 32 k_hat times.  wd*(m) on m = n_u/4, n_u/2, n_u DISTINCT sample rows at 32 k_hat passes,
    log-log fit, extrapolated to N/k_hat; ship the team config with that wd (no code change).
  INFORMATION (reported, not gated)
    row_extrap          the L1-validated experiment: random row-subsets of the sample (1k/2k/4k) at the
                        production pass count (32), wd*(m) per size, log-log fit, extrapolated to 80,000 rows
                        (not gated: its outcome depends on the noise in the fitted slope, see DECOYS)
    l1_gpt6_ensemble    GPT-6's L1 repo verbatim (forward-corrected noise loss, 4-member ensemble, hedged
                        row-count decay rule) - an L1 solution L2 does not defeat
    curve_wd*           team lr, wd in {0.3, 0.6, 1, 2}
  T = midpoint of the reference and the best decoy, floored to 0.005; ok iff both margins >= MARGIN.
Sample-scale wd*(m): dev accuracy averaged over 3 training seeds on the grid WD_GRID, vertex of the parabola
(in log wd) through the best grid point and its neighbours.  Cache: $CERT_CACHE (default
/tmp/v10/scaleup/cert_cache_l2_e<EPOCHS>.json).  Summary: l15/tasks/scaleup_l2_cert.json."""
import json, math, os, shutil, sys, tempfile, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
from . import scaleup as S
from . import scaleup_l2 as L2
from .scaleup_cert import team_grid, search_grid, best

CACHE = os.environ.get("CERT_CACHE", "/tmp/v10/scaleup/cert_cache_l2_e%d.json" % S.EPOCHS)
HERE = os.path.dirname(os.path.abspath(__file__))
DECOY_REPOS = {"l1_fable_injob": os.path.join(HERE, "scaleup_l2_decoys", "l1_fable_s0_injob_holdout"),
               "l1_gpt6_ensemble": os.path.join(HERE, "scaleup_l2_decoys", "l1_gpt6_s1_robust_ensemble")}
WD_GRID = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
# gated decoys: policies whose failure does not hinge on experimental noise.  row_extrap (the L1-validated
# equal-pass subsampling experiment) is NOT gated: on seed 0 its fitted slope came out -0.41 (wd 0.63, which
# passes) where L1's runs saw ~-0.6 (wd 0.3, which fails) - its outcome depends on the noise in the fit, so it is
# reported as information together with its production score.
DECOYS = ("ship_as_is", "retune_sample", "row_rule", "l1_fable_injob")
RS = (0, 1, 2)
sys.path.insert(0, S.REPO_SRC)
import train as TR  # noqa: E402


def _key(*a):
    return json.dumps(a, sort_keys=True)


def _uniq_first(X, y):
    Z = np.concatenate([X, y[:, None].astype(np.float32)], 1)
    _, first, cnt = np.unique(Z, axis=0, return_index=True, return_counts=True)
    return np.sort(first), cnt


def dup_stats(seed):
    """What the agent can compute from sample.npz alone."""
    Xs, ys = S.data(L2.base_params(seed))["sample"]
    first, cnt = _uniq_first(Xs, ys)
    n = len(Xs); pairs = int((cnt * (cnt - 1) // 2).sum())
    k_hat = 1.0 + pairs * (S.N_FULL - 1) / (n * (n - 1) / 2.0)
    return {"n": n, "n_distinct": int(len(first)), "pairs": pairs, "k_hat": round(k_hat, 4),
            "mult_hist": np.bincount(cnt).tolist()}


# ------------------------------------------------------------------ jobs
def sweep_job(seed, cfg):
    dd = S.data(L2.base_params(seed))
    Xs, ys = dd["sample"]
    P, norm = TR.train(Xs, ys, S.S_VIS, cfg, seed=0)
    Xd, yd = dd["dev"]
    return float((TR.predict(P, norm, Xd) == yd).mean())


def sd_job(seed, mode, m, passes, wd, rs, team):
    """Sample-scale run on m rows of the sample: mode 'rows' = random row subset (duplicates kept, what a
    plain subsample gives), mode 'distinct' = random subset of the distinct rows."""
    dd = S.data(L2.base_params(seed))
    Xs, ys = dd["sample"]
    r = np.random.default_rng([seed, 17, m, rs, 0 if mode == "rows" else 1])
    if mode == "rows":
        sub = r.permutation(len(Xs))[:m]
    else:
        first, _ = _uniq_first(Xs, ys)
        sub = r.permutation(first)[:m]
    cfg = dict(team); cfg["weight_decay"] = wd
    P, norm = TR.train(Xs[sub], ys[sub], max(1, int(round(passes * m / S.BS))), cfg, seed=rs)
    Xd, yd = dd["dev"]
    return float((TR.predict(P, norm, Xd) == yd).mean())


def prod_rep_job(seed, spec, team, rep):
    p = L2.base_params(seed)
    p.update({"team_lr": team["lr"], "team_wd": team["weight_decay"], "T": 0.0, "job_cpu_s": 0.0})
    work = tempfile.mkdtemp(prefix="cert2_data_"); repo = tempfile.mkdtemp(prefix="cert2_repo_")
    try:
        if spec[0] == "cfg":
            shutil.copy(os.path.join(S.REPO_SRC, "train.py"), repo)
            json.dump(spec[1], open(os.path.join(repo, "config.json"), "w"))
        else:
            shutil.rmtree(repo); shutil.copytree(DECOY_REPOS[spec[1]], repo)
        S._write_data(work, p)
        acc, info = S.run_job(repo, p, rep, data_dir=work)
        return {"acc": acc, "cpu_s": info.get("cpu_s"), "err": info.get("error")}
    finally:
        shutil.rmtree(repo, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ procedures
def vertex(accs):
    """wd*(m) from mean dev accuracies on WD_GRID: parabola vertex in log wd through the argmax and neighbours."""
    a = np.asarray(accs); i = int(np.argmax(a)); x = np.log(WD_GRID)
    if 0 < i < len(a) - 1:
        x0, x1, x2 = x[i - 1:i + 2]; y0, y1, y2 = a[i - 1:i + 2]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
        B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
        if A < 0:
            return float(np.exp(np.clip(-B / (2 * A), x0, x2)))
    return float(WD_GRID[i])


def extrapolate(ms, wds, n_target):
    lm, lw = np.log(ms), np.log(wds)
    b, a = np.polyfit(lm, lw, 1)
    return float(np.clip(np.exp(a + b * np.log(n_target)), 0.01, 10.0)), float(b)


def scaling_experiment(seed, mode, ms, passes, n_target, team, cache, pend):
    rows, done = [], True
    for m in ms:
        accs = []
        for wd in WD_GRID:
            v = []
            for rs in RS:
                k = _key("sd", seed, mode, m, round(passes, 4), wd, rs, team)
                if k in cache: v.append(cache[k])
                else: pend.append((k, sd_job, (seed, mode, m, passes, wd, rs, team))); done = False
            accs.append(float(np.mean(v)) if len(v) == len(RS) else None)
        rows.append(accs)
    if not done:
        return None
    wstar = [vertex(a) for a in rows]
    wd, slope = extrapolate(ms, wstar, n_target)
    return {"mode": mode, "m": list(ms), "passes": round(passes, 4), "dev_acc": [[round(x, 4) for x in a] for a in rows],
            "wd_grid": list(WD_GRID), "wd_star": [round(w, 4) for w in wstar], "slope": round(slope, 4),
            "n_target": round(n_target, 1), "wd": round(wd, 4)}


def prod_result(seed, spec, cache):
    reps, pend = [], []
    for r in (0, 1):
        kr = _key("prodrep", seed, spec, r)
        if kr in cache: reps.append(cache[kr])
        else: pend.append((kr, r))
    if pend:
        return None, pend
    accs = [x["acc"] for x in reps]
    return {"acc": None if any(a is None for a in accs) else float(np.mean(accs)),
            "cpu_s": [x["cpu_s"] for x in reps], "per_seed": accs, "err": [x["err"] for x in reps if x["err"]]}, []


def plan(seed, cache, phase):
    pend = []
    sw, ss = [], []
    for c in team_grid():
        k = _key("sweep", seed, c)
        if k in cache: sw.append((c, cache[k]))
        else: pend.append((k, sweep_job, (seed, c)))
    for c in search_grid(seed):
        k = _key("sweep", seed, c)
        if k in cache: ss.append((c, cache[k]))
        else: pend.append((k, sweep_job, (seed, c)))
    if pend:
        return pend, None
    team = best(sw)[0]
    retune = best(ss + [(team, best(sw)[1])])[0]
    ds = dup_stats(seed)
    nu = ds["n_distinct"]
    ref_exp = scaling_experiment(seed, "distinct", [nu // 4, nu // 2, nu], S.EPOCHS * ds["k_hat"],
                                 S.N_FULL / ds["k_hat"], team, cache, pend)
    row_exp = scaling_experiment(seed, "rows", [S.N_VIS // 4, S.N_VIS // 2, S.N_VIS], S.EPOCHS, S.N_FULL,
                                 team, cache, pend)
    if pend:
        return pend, None
    def with_wd(wd):
        c = dict(team); c["weight_decay"] = round(float(wd), 6); return ("cfg", c)
    specs = {"ship_as_is": ("cfg", team), "retune_sample": ("cfg", retune),
             "row_rule": with_wd(team["weight_decay"] * S.N_VIS / S.N_FULL), "row_extrap": with_wd(row_exp["wd"]),
             "l1_fable_injob": ("repo", "l1_fable_injob"), "reference": with_wd(ref_exp["wd"]),
             "l1_gpt6_ensemble": ("repo", "l1_gpt6_ensemble")}
    for wd in (0.3, 0.6, 1.0, 2.0):
        specs["curve_wd%g" % wd] = with_wd(wd)
    summ = {"team": team, "team_sweep": [dict(lr=c["lr"], weight_decay=c["weight_decay"], dev_acc=a) for c, a in sw],
            "retune_search": [dict(c, dev_acc=a) for c, a in ss], "dup_stats": ds,
            "reference_experiment": ref_exp, "row_experiment": row_exp, "margin": S.MARGIN}
    if phase != "all":
        summ["phase"] = "sample"
        return [], summ
    res = {}
    for name, spec in specs.items():
        r, pr = prod_result(seed, spec, cache)
        if r is not None: res[name] = dict(r, spec=spec)
        for kr, rep in pr:
            pend.append((kr, prod_rep_job, (seed, spec, team, rep)))
    if pend:
        return pend, None
    dec = DECOYS
    summ.update({"ref": res["reference"], "decoys": {k: res[k] for k in dec},
                 "info": {k: res[k] for k in res if k not in dec and k != "reference"},
                 "job_cpu_s": float(np.mean(res["ship_as_is"]["cpu_s"]))})
    if all(v["acc"] is not None for v in res.values()):
        top = max(res[k]["acc"] for k in dec)
        T = math.floor(((res["reference"]["acc"] + top) / 2) / 0.005) * 0.005
        summ.update({"T": round(T, 3), "ref_margin": round(res["reference"]["acc"] - T, 5),
                     "decoy_margin": round(T - top, 5)})
        summ["ok"] = bool(summ["ref_margin"] >= S.MARGIN and summ["decoy_margin"] >= S.MARGIN)
        summ["passing_info"] = {k: v["acc"] >= T for k, v in summ["info"].items()}
    else:
        summ["ok"] = False
    return [], summ


def main():
    tl, npr, phase = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    seeds = [int(s) for s in sys.argv[4:]]
    t0 = time.time()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    with Pool(npr) as pool:
        while time.time() - t0 < tl:
            todo, n_sum = [], 0
            for s in seeds:
                pend, summ = plan(s, cache, phase)
                todo += pend
            if not todo:
                break
            prod = todo[0][1] is prod_rep_job
            futs = [(k, pool.apply_async(fn, args)) for k, fn, args in todo[:max(npr, 1) * (1 if prod else 8)]]
            stop = False
            for k, f in futs:
                rem = tl - (time.time() - t0)
                if rem <= 0: stop = True; break
                try:
                    cache[k] = f.get(timeout=rem)
                    tmp = CACHE + ".tmp"; json.dump(cache, open(tmp, "w")); os.replace(tmp, CACHE)
                except Exception as ex:
                    print("job %s after %.0fs: %s" % ("timeout" if isinstance(ex, __import__("multiprocessing").TimeoutError)
                                                      else "error", time.time() - t0, repr(ex)[:300]), k[:100], flush=True)
                    stop = True; break
            if stop: break
        pool.terminate()
    allc = json.load(open(S.CERT_PATH_L2)) if os.path.exists(S.CERT_PATH_L2) else {}
    for s in seeds:
        pend, summ = plan(s, cache, phase)
        if summ is None:
            print("seed %d: %d jobs pending" % (s, len(pend))); continue
        re_, ro = summ["reference_experiment"], summ["row_experiment"]
        print("seed %d team=%s dup=%s\n  ref-exp wd*(m)=%s slope %.2f -> wd %.3f | row-exp wd*(m)=%s slope %.2f -> wd %.3f" % (
            s, {k: summ["team"][k] for k in ("lr", "weight_decay")}, {k: summ["dup_stats"][k] for k in ("pairs", "k_hat")},
            re_["wd_star"], re_["slope"], re_["wd"], ro["wd_star"], ro["slope"], ro["wd"]))
        if phase == "all":
            allc[str(s)] = summ
            print("  T=%s ok=%s ref=%s decoys=%s info=%s" % (summ.get("T"), summ.get("ok"), summ["ref"]["acc"],
                  {k: v["acc"] and round(v["acc"], 4) for k, v in summ["decoys"].items()},
                  {k: v["acc"] and round(v["acc"], 4) for k, v in summ["info"].items()}))
    if phase == "all":
        tmp = S.CERT_PATH_L2 + ".tmp"; json.dump(allc, open(tmp, "w"), indent=1); os.replace(tmp, S.CERT_PATH_L2)


if __name__ == "__main__":
    main()
