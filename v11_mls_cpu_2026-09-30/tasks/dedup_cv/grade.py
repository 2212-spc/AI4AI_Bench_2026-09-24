"""Host-side grader for dedup_cv.  python3 grade.py <agent_workdir>  -> JSON on stdout.

Metric per setting = test MSE (new entities, noise-free target) of the config the solution selects,
divided by the best test MSE among all candidates (oracle).  Pass needs ratio <= THRESHOLD everywhere."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hidden")); sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import numpy as np
import world as W
from sandbox_call import call

THRESHOLD = 1.20
CACHE = os.path.join(HERE, "hidden", "_oracle_cache.json")


def _load_candidates():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cand_ref", os.path.join(HERE, "workdir", "candidates.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.candidates()


def oracle_table():
    key = hashlib.md5((open(os.path.join(HERE, "hidden", "world.py")).read() +
                       open(os.path.join(HERE, "workdir", "candidates.py")).read()).encode()).hexdigest()
    if os.path.exists(CACHE):
        c = json.load(open(CACHE))
        if c.get("key") == key:
            return c["table"]
    C = _load_candidates(); table = {}
    for row in W.SETTINGS:
        s = W.build_setting(row)
        table[s["name"]] = {n: float(np.mean((mk().fit(s["X"], s["y"]).predict(s["Xte"]) - s["yte"]) ** 2)) for n, mk in C}
    json.dump({"key": key, "table": table}, open(CACHE, "w"), indent=1)
    return table


def grade(sol_dir):
    table = oracle_table(); settings, ok_all = {}, True
    for row in W.SETTINGS:
        s = W.build_setting(row); t = table[s["name"]]
        try:
            pick, secs = call(sol_dir, "solution.py", "select", (s["X"], s["y"]), timeout=1200)
            pick = str(pick)
        except Exception as e:
            pick, secs = f"ERROR:{str(e)[-200:]}", 0.0
        ratio = t[pick] / min(t.values()) if pick in t else 99.0
        ok = ratio <= THRESHOLD; ok_all &= ok
        settings[s["name"]] = dict(metric=round(ratio, 3), threshold=THRESHOLD, ok=ok, hidden=s["hidden"],
                                   pick=pick, oracle=min(t, key=t.get), secs=round(secs, 1))
    return {"pass": ok_all, "settings": settings}


if __name__ == "__main__":
    print(json.dumps(grade(os.path.abspath(sys.argv[1])), indent=1))
