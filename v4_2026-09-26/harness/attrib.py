"""Failure attribution: for every failed item of every graded run, find the nearest named rival.
Labels for point/set items: rival (strong|weak) / precision (err < 2T, no rival) / unattributed (err >= 2T, no rival).
Rebuilds each task's instance from (blueprint, world seed) with the frozen code, checks that the rebuilt
key equals the shipped key (determinism check), then compares the model's answer with every rival's
answer in units of the item's tolerance T.
  LAB_RUN_ROOT=/tmp/v3/runs python3 harness/attrib.py      (run from the v3 root; tasks/ must be the shipped set)"""
import glob, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v3/runs")
sys.path.insert(0, ROOT)
from scalelab import build as B, queries as Q

TASK = {"t01ws1": ("t01_lr_horizon", 1, "t01-lr-horizon-ws1"), "t01ws2": ("t01_lr_horizon", 2, "t01-lr-horizon-ws2"),
        "t01bws1": ("t01b_lr_horizon_tight", 1, "t01b-lr-horizon-tight-ws1"), "t01bws2": ("t01b_lr_horizon_tight", 2, "t01b-lr-horizon-tight-ws2"),
        "t02ws2": ("t02_stability_edge", 2, "t02-stability-edge-ws2"), "t02ws3": ("t02_stability_edge", 3, "t02-stability-edge-ws3"),
        "t03ws11": ("t03_filter_repeat", 11, "t03-filter-repeat-ws11"), "t03ws28": ("t03_filter_repeat", 28, "t03-filter-repeat-ws28"),
        "t03bws11": ("t03b_filter_repeat_tight", 11, "t03b-filter-repeat-tight-ws11"),
        "t07ws1": ("t07_checkpoint_ledger", 1, "t07-checkpoint-ledger-ws1"), "t07ws3": ("t07_checkpoint_ledger", 3, "t07-checkpoint-ledger-ws3"),
        "t07ws4": ("t07_checkpoint_ledger", 4, "t07-checkpoint-ledger-ws4")}
cache = {}


def inst_for(tag):
    if tag not in cache:
        bp, ws, tid = TASK[tag]
        inst = B.build_instance(bp, ws)
        shipped = json.load(open(os.path.join(ROOT, "tasks", tid, "tests", "key.json")))
        rebuilt = {it["id"]: it["key"] for it in inst["cal"]["items"]}
        same = all(json.dumps(rebuilt[it["id"]], sort_keys=True) == json.dumps(it["key"], sort_keys=True) for it in shipped)
        cache[tag] = (inst, shipped, same)
        print("# rebuilt %s: key identical to shipped = %s" % (tid, same))
    return cache[tag]


def num(a):
    try:
        return float(a["lo"]), float(a["hi"])
    except Exception:
        return None


for rd in sorted(glob.glob(RUN_ROOT + "/*/")):
    run = os.path.basename(rd.rstrip("/")); tag = run.split("_")[0]
    st = json.load(open(rd + "state.json"))
    ap = rd + "app/answers.json"
    if st["status"] != "done" or not os.path.exists(ap):
        print("%-28s status=%s (not graded)" % (run, st["status"])); continue
    ans = json.load(open(ap)); inst, shipped, same = inst_for(tag)
    rivals = inst["rivals"]; out = []
    for it in shipped:
        a = ans.get(it["id"]); g = Q.grade_item(it, a)
        if (g[0] if isinstance(g, tuple) else g.get("pass")):
            continue
        if it["kind"] in ("point", "set"):
            m = num(a) if isinstance(a, dict) else None
            if m is None:
                out.append("%s:blank" % it["id"]); continue
            T = it["tol"]; k = it["key"]
            ek = max(abs(m[0] - k["lo"]), abs(m[1] - k["hi"])) / T
            best = []
            for rn, ra in rivals.items():
                r = num(ra.get(it["id"], {})) if isinstance(ra.get(it["id"]), dict) else None
                if r is None:
                    continue
                d = max(abs(m[0] - r[0]), abs(m[1] - r[1])) / T
                rk = max(abs(r[0] - k["lo"]), abs(r[1] - k["hi"])) / T
                if rk >= 1.0:                       # only rivals that are themselves wrong on this item
                    best.append((d, rn, rk))
            best.sort()
            near = best[0] if best else (math.inf, "-", 0.0)
            # labels: a rival match is 'strong' only if that rival is itself clearly wrong (>= 2T from the key);
            # otherwise the model sits between key and a barely-wrong rival and the match says little.
            if near[0] < min(1.0, ek):
                lab = "rival %s (%.2fT away; rival err %.2fT, %s)" % (near[1], near[0], near[2],
                                                                      "strong" if near[2] >= 2.0 else "weak")
            else:
                lab = "precision" if ek < 2.0 else "unattributed"
            out.append("%s:%s err=%.2fT -> %s" % (it["id"], it["kind"], ek, lab))
        else:
            v = list(a.values())[0] if isinstance(a, dict) and a else None
            same_r = [rn for rn, ra in rivals.items() if isinstance(ra.get(it["id"]), dict) and list(ra[it["id"]].values())[0] == v]
            out.append("%s:%s answered %s (key %s) -> same as %s" % (it["id"], it["kind"], v, list(it["key"].values())[0],
                                                                    ",".join(same_r) or "no rival"))
    g = Q.grade(shipped, ans)
    print("%-28s score=%.2f all=%s  %s" % (run, g["score"], g["all_pass"], " | ".join(out) or "-"))
