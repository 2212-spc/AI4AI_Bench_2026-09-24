import sys, json
from run import train
n = int(sys.argv[1]); steps = int(sys.argv[2]); lo=float(sys.argv[3]); hi=float(sys.argv[4]); iters=int(sys.argv[5])
# assume lo stable, hi diverged; bisect in log space
import math
for i in range(iters):
    mid = math.sqrt(lo*hi)
    r = train(n, steps, mid)
    d = r["diverged"] or (r.get("final_loss") or 0) > 20
    print(n, steps, "%.4g"%mid, "DIV" if d else "ok", r.get("final_loss"), r.get("diverged_at_step"), "left=%.3g"%r["budget_left"], flush=True)
    if d: hi = mid
    else: lo = mid
print("edge between", lo, hi)
