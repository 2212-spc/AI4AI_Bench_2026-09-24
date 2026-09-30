"""Local check on the two visible dev settings: python3 eval_dev.py [solution.py]
Prints the selected config and its test-MSE ratio to the best candidate (the hidden grader uses the same
metric on these two plus several unseen settings; pass needs ratio <= 1.20 on every one)."""
import importlib.util, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "solution.py")
spec = importlib.util.spec_from_file_location("solution", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for s in pickle.load(open(os.path.join(HERE, "data", "dev_settings.pkl"), "rb")):
    pick = mod.select(s["X"], s["y"]); t = s["test_mse"]
    print(f"{s['name']}: pick={pick}  ratio={t[pick] / min(t.values()):.3f}  (best={min(t, key=t.get)})")
