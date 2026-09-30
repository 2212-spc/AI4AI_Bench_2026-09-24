"""Local check on the 2 visible dev portfolios: python3 eval_dev.py [solution.py]
Each dev item has history, snapshot_month, new applicants and their TRUE 12-month default probability
(available only for dev). Prints the graded metric (decile calibration error; pass needs <= 0.25 on
every dev AND hidden portfolio)."""
import importlib.util, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "solution.py")
spec = importlib.util.spec_from_file_location("solution", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def decile_error(pred, truth):
    q = np.argsort(np.argsort(truth)) * 10 // len(truth)
    return float(np.mean([abs(pred[q == i].mean() - truth[q == i].mean()) / truth[q == i].mean() for i in range(10)]))


for s in pickle.load(open(os.path.join(HERE, "data", "dev_settings.pkl"), "rb")):
    p = np.clip(np.asarray(mod.fit_predict(s["history"].copy(), s["snapshot_month"], s["new"].copy()), float), 0, 1)
    print(f"{s['name']}: decile_error={decile_error(p, s['truth']):.3f}  mean_pred={p.mean():.3f}  mean_true={s['truth'].mean():.3f}")
