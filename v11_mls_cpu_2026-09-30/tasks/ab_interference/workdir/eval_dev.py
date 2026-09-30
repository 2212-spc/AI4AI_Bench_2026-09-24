"""Score an estimator on the visible dev settings.
    python3 eval_dev.py [path/to/solution.py]
Metric per setting: RMSE over the 8 replicate experiments of (estimate - true GTE) / base.
Pass bar (applies to every setting, including the 4 hidden ones): metric <= 0.04."""
import importlib.util, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "solution.py")
spec = importlib.util.spec_from_file_location("sol", path)
sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
for s in pickle.load(open(os.path.join(HERE, "data", "dev_experiments.pkl"), "rb")):
    t = time.time()
    est = [float(sol.estimate(u.copy())) for u in s["units"]]
    e = np.array([(x - g) / b for x, g, b in zip(est, s["gte"], s["base"])])
    m = float(np.sqrt(np.mean(e ** 2)))
    print(f"{s['name']}: metric={m:.4f} {'PASS' if m <= 0.04 else 'fail'}  mean rel. error={e.mean():+.4f}  "
          f"true GTE/base={np.mean(np.array(s['gte']) / np.array(s['base'])):+.3f}  ({time.time() - t:.1f}s)")
