"""Host-side grader for ab_interference.  python3 grade.py <agent_workdir> -> JSON on stdout.

Each setting = N_REP independent replicate experiments drawn from one hidden marketplace world.
Metric per setting = RMSE over replicates of (estimate - true GTE) / B, where B is the mean seller
outcome with nobody treated.  Pass needs metric <= THRESHOLD on every setting (2 dev + 4 hidden)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hidden")); sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import numpy as np
import world as W
from sandbox_call import call

THRESHOLD = 0.04
SOL_FILE = "solution.py"


def grade(sol_dir):
    settings, ok_all, err = {}, True, None
    for row in W.SETTINGS:
        s = W.build_setting(row)
        try:
            est, secs = call(sol_dir, "_runner.py", "run", (s["units"],), timeout=600,
                             extra_files=[os.path.join(HERE, "hidden", "_runner.py")])
            e = np.array([(float(x) - g) / b for x, g, b in zip(est, s["gte"], s["base"])])
            m = float(np.sqrt(np.mean(e ** 2))) if np.all(np.isfinite(e)) and len(e) == len(s["gte"]) else 99.0
        except Exception as ex:
            m, secs, e, err = 99.0, 0.0, np.array([]), str(ex)[-500:]
        ok = m <= THRESHOLD; ok_all &= ok
        settings[s["name"]] = dict(metric=m, threshold=THRESHOLD, ok=ok, hidden=s["hidden"], secs=round(secs, 1),
                                   mean_err=round(float(e.mean()), 4) if len(e) else None,
                                   true_gte_rel=round(float(np.mean(np.array(s["gte"]) / np.array(s["base"]))), 4))
    out = {"pass": ok_all, "settings": settings}
    if err: out["error"] = err
    return out


if __name__ == "__main__":
    print(json.dumps(grade(os.path.abspath(sys.argv[1])), indent=1))
