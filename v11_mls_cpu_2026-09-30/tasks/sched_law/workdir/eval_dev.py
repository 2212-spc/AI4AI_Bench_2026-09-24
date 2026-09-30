"""Evaluate solution.py (or another file) on the visible dev worlds.
    python3 eval_dev.py [path/to/solution.py]
Metric per world: mean over query schedules of |pred - true final loss| / scale,
scale = (final loss of the constant-peak-lr run) - (final loss of the cosine run).
Pass bar (applies to every world, including hidden ones): metric <= 0.10."""
import importlib.util, pickle, sys, time
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "solution.py"
spec = importlib.util.spec_from_file_location("sol", path)
sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
import os
for w in pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dev_worlds.pkl"), "rb")):
    t = time.time()
    pred = sol.fit_predict(w["train"], w["queries"])
    errs = {k: abs(pred[k] - w["truth"][k]) / w["scale"] for k in w["queries"]}
    m = np.mean(list(errs.values()))
    print(f"{w['name']}: metric={m:.4f} {'PASS' if m <= 0.10 else 'fail'}  ({time.time()-t:.1f}s)  "
          + " ".join(f"{k}={v:.3f}" for k, v in errs.items()))
