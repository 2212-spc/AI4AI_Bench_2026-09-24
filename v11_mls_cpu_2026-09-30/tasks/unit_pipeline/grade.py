"""Host-side grader for unit_pipeline.  python3 grade.py <agent_workdir> -> JSON.

The agent's preprocess.py is fit on the raw training export and applied to each deployment batch (in a
sandboxed child).  The grader's own copy of model.py is trained on the transformed training data.
Metric per deployment batch = test MSE / MSE of the same model trained and tested on correctly-unit
(clean) data.  Pass needs metric <= THRESHOLD on every batch."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hidden")); sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import importlib.util
import numpy as np
import world as W
from sandbox_call import call

THRESHOLD = 1.10
SOL_FILE = "preprocess.py"   # the file the agent edits


def _model():
    spec = importlib.util.spec_from_file_location("model_ref", os.path.join(HERE, "workdir", "model.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def grade(sol_dir):
    M = _model()
    Xtr, ytr = W.build_train(); Xc, yc = W.build_train(clean=True)
    orc = M.fit(Xc.drop(columns=["site"]), yc)
    tests = [W.build_setting(r) for r in W.SETTINGS]; cleans = [W.build_setting(r, clean=True) for r in W.SETTINGS]
    base = {c["name"]: float(np.mean((M.predict(orc, c["X"].drop(columns=["site"])) - c["y"]) ** 2)) for c in cleans}
    settings, ok_all = {}, True
    try:
        (Ttr, Tte), secs = call(sol_dir, "_runner.py", "run", (Xtr, [t["X"] for t in tests]), timeout=1200,
                                extra_files=[os.path.join(HERE, "hidden", "_runner.py")])
        m = M.fit(Ttr, ytr); err = None
    except Exception as e:
        err, secs = str(e)[-500:], 0.0
    for i, t in enumerate(tests):
        try:
            assert err is None, err
            assert len(Tte[i]) == len(t["X"]), "transform must keep every row, in order"
            mse = float(np.mean((M.predict(m, Tte[i]) - t["y"]) ** 2))
            ratio = mse / base[t["name"]] if np.isfinite(mse) else 99.0
        except Exception as e:
            ratio, err = 99.0, str(e)[-300:]
        ok = ratio <= THRESHOLD; ok_all &= ok
        settings[t["name"]] = dict(metric=round(ratio, 3), threshold=THRESHOLD, ok=ok, hidden=t["hidden"], secs=round(secs, 1))
    out = {"pass": ok_all, "settings": settings}
    if err: out["error"] = err
    return out


if __name__ == "__main__":
    print(json.dumps(grade(os.path.abspath(sys.argv[1])), indent=1))
