import sys
from run import train
n=int(sys.argv[1]); steps=int(sys.argv[2]); seeds=int(sys.argv[3])
for eta in sys.argv[4:]:
    res=[]
    for s in range(seeds):
        r=train(n,steps,float(eta),seed=100+s)
        res.append("DIV@%s"%r.get("diverged_at_step") if r["diverged"] else "%.3f"%r["final_loss"])
    print(n,steps,eta,res,flush=True)
