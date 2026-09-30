"""Host-side grader for delayed_labels.  python3 grade.py <agent_workdir> -> JSON.

Metric per setting = decile calibration error: split the new applicants into 10 deciles of TRUE risk,
mean over deciles of |mean predicted - mean true| / mean true.  (Portfolio-level loss forecasts by risk
band - what a credit team actually uses; insensitive to per-account noise, sensitive to systematic bias.)
Pass needs <= THRESHOLD on every setting."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hidden")); sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import numpy as np
import world as W
from sandbox_call import call

THRESHOLD = 0.25


def decile_error(pred, truth):
    q = np.argsort(np.argsort(truth)) * 10 // len(truth)
    return float(np.mean([abs(pred[q == i].mean() - truth[q == i].mean()) / truth[q == i].mean() for i in range(10)]))


def grade(sol_dir):
    settings, ok_all = {}, True
    for row in W.SETTINGS:
        s = W.build_setting(row)
        try:
            pred, secs = call(sol_dir, "solution.py", "fit_predict", (s["history"], s["snapshot_month"], s["new"]), timeout=1200)
            pred = np.asarray(pred, float).ravel()
            assert len(pred) == len(s["truth"]) and np.all(np.isfinite(pred))
            m = decile_error(np.clip(pred, 0, 1), s["truth"]); err = None
        except Exception as e:
            m, secs, err = 99.0, 0.0, str(e)[-300:]
        ok = m <= THRESHOLD; ok_all &= ok
        settings[s["name"]] = dict(metric=round(m, 4), threshold=THRESHOLD, ok=ok, hidden=s["hidden"], secs=round(secs, 1),
                                   **({"error": err} if err else {}))
    return {"pass": ok_all, "settings": settings}


if __name__ == "__main__":
    print(json.dumps(grade(os.path.abspath(sys.argv[1])), indent=1))
