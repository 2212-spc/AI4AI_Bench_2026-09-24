"""Host-side grader for sched_law.  python3 grade.py <agent_workdir>  -> JSON on stdout."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hidden")); sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import numpy as np
import world as W
from sandbox_call import call

THRESHOLD = 0.10   # mean |pred - true| / schedule-effect-scale, required on EVERY world


def grade(sol_dir):
    settings, ok_all = {}, True
    for row in W.SETTINGS:
        s = W.build_setting(row)
        pred, secs = call(sol_dir, "solution.py", "fit_predict", (s["train"], s["queries"]), timeout=900)
        errs = {k: abs(float(pred[k]) - s["truth"][k]) / s["scale"] if np.isfinite(pred.get(k, np.nan)) else 99.0
                for k in s["queries"]}
        m = float(np.mean(list(errs.values())))
        ok = m <= THRESHOLD
        ok_all &= ok
        settings[s["name"]] = dict(metric=m, threshold=THRESHOLD, ok=ok, hidden=s["hidden"], secs=round(secs, 1),
                                   per_query={k: round(v, 4) for k, v in errs.items()})
    return {"pass": ok_all, "settings": settings}


if __name__ == "__main__":
    print(json.dumps(grade(os.path.abspath(sys.argv[1])), indent=1))
