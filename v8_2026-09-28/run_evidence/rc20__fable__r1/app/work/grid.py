import sys, json, itertools
from concurrent.futures import ThreadPoolExecutor
from lab import run
batch=int(sys.argv[1]); tokens=float(sys.argv[2])
lrs=[float(x) for x in sys.argv[3].split(",")]; wds=[float(x) for x in sys.argv[4].split(",")]
seeds=int(sys.argv[5]) if len(sys.argv)>5 else 1
pts=list(itertools.product(lrs,wds))
with ThreadPoolExecutor(8) as ex:
    res=list(ex.map(lambda p: run(batch,tokens,p[0],p[1],seeds), pts))
print("%10s"%"lr\\wd"+"".join("%10g"%w for w in wds))
for lr in lrs:
    print("%10g"%lr+"".join("%10.4f"%(r["final_val_loss"] if r else float('nan')) for (p,r) in zip(pts,res) if p[0]==lr))
print("budget_left", res[-1]["budget_left"])
