"""Replay the two visible dev settings exactly the way the grader does (in-process here, no sandbox).
    python3 eval_dev.py [path/to/features.py]
Train: model.fit(features.build(dump, train_rows)) where dump = the warehouse as of the end of day n_hist-1.
Live:  for each of the next 60 days d, features.build(warehouse as of the end of day d-1, rows of day d),
       prediction compared with the FINAL units of day d.
Pass on a setting: wape / oracle_wape <= 1.55 and |bias| <= 0.03 (same bar on the 4 hidden settings)."""
import importlib.util, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "features.py")
spec = importlib.util.spec_from_file_location("features", path)
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)


def as_of(t, end):
    return dict(sales_log=t["sales_log"][t["sales_log"].recorded_at <= end].reset_index(drop=True),
                promos=t["promos"][t["promos"].planned_at <= end].reset_index(drop=True),
                weather=t["weather"][t["weather"].recorded_at <= end].reset_index(drop=True),
                stockouts=t["stockouts"][t["stockouts"].recorded_at <= end].reset_index(drop=True))


for s in pickle.load(open(os.path.join(HERE, "data", "dev_settings.pkl"), "rb")):
    t0 = time.time()
    tr = s["train_rows"]
    X = F.build(as_of(s["tables"], s["n_hist"] - 1), tr[["store", "day"]].copy())
    cols = list(X.columns); m = model.fit(X, tr.units.to_numpy())
    P, Y = [], []
    for d in s["live_days"]:
        rows = s["final"][s["final"].day == d].sort_values("store").reset_index(drop=True)
        Xd = F.build(as_of(s["tables"], d - 1), rows[["store", "day"]].copy())
        P.append(np.clip(np.asarray(model.predict(m, Xd[cols]), float), 0, None)); Y.append(rows.units.to_numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    ratio = np.abs(p - y).sum() / np.abs(s["oracle"] - y).sum(); bias = p.sum() / y.sum() - 1
    ok = ratio <= 1.55 and abs(bias) <= 0.03
    print(f"{s['name']}: wape={np.abs(p - y).sum() / y.sum():.3f}  wape/oracle={ratio:.3f}  bias={bias:+.3f}  "
          f"{'PASS' if ok else 'fail'}  ({time.time() - t0:.1f}s)")
