"""Local check on the visible dev batches: python3 eval_dev.py [preprocess.py]
Prints test MSE per dev batch. (The hidden grader divides MSE by what the same model reaches on data
with correct units; that reference MSE is ~0.25 on every batch, pass needs ratio <= 1.10 everywhere.)"""
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model as M
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "preprocess.py")
spec = importlib.util.spec_from_file_location("pp", path); pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
tr = pd.read_csv(os.path.join(HERE, "data", "train.csv")); ytr = tr.pop("outcome").values
st = pp.fit_preprocess(tr.copy()); m = M.fit(pp.transform(st, tr.copy()), ytr)
for n in ["dev_mix", "dev_S4"]:
    te = pd.read_csv(os.path.join(HERE, "data", f"{n}.csv")); y = te.pop("outcome").values
    print(f"{n}: MSE={np.mean((M.predict(m, pp.transform(st, te.copy())) - y) ** 2):.4f}")
