#!/usr/bin/env python3
import json, sys, subprocess
b, t, lr, wd = sys.argv[1:5]
seeds = sys.argv[5] if len(sys.argv) > 5 else "1"
out = subprocess.run(["lab", "run", f"batch={b}", f"tokens={t}", f"lr={lr}", f"wd={wd}", f"seeds={seeds}"], capture_output=True, text=True).stdout
r = json.loads(out)
if "final_val_loss" not in r:
    print("ERR", out); sys.exit(1)
with open("/app/work/results.jsonl", "a") as f:
    f.write(json.dumps(r) + "\n")
print(f"B={b} T={t} lr={lr} wd={wd} s={seeds} loss={r['final_val_loss']:.5f} per_seed={r['per_seed']} left={r['budget_left']}")
