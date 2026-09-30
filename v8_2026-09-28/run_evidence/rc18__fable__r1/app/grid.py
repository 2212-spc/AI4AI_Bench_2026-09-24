import json, subprocess, sys, itertools
def run(batch, tokens, lr, wd, seeds=1):
    out = subprocess.run(["bin/lab","run",f"batch={batch}",f"tokens={tokens}",f"lr={lr}",f"wd={wd}",f"seeds={seeds}"],capture_output=True,text=True).stdout
    r = json.loads(out)
    return r
if __name__=="__main__":
    batch=int(sys.argv[1]); tokens=float(sys.argv[2])
    lrs=[float(x) for x in sys.argv[3].split(",")]; wds=[float(x) for x in sys.argv[4].split(",")]
    seeds=int(sys.argv[5]) if len(sys.argv)>5 else 1
    for lr,wd in itertools.product(lrs,wds):
        r=run(batch,tokens,lr,wd,seeds)
        print(f"B={batch} T={tokens} lr={lr:.5g} wd={wd:.4g} steps={r['steps']} loss={r['final_val_loss']:.4f} per_seed={r['per_seed']} left={r['budget_left']}")
