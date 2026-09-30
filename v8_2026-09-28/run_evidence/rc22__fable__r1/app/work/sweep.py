import json, sys, subprocess, concurrent.futures as cf
def run(batch, tokens, lr, wd, seeds=1):
    out = subprocess.run(["/app/bin/lab","run",f"batch={batch}",f"tokens={tokens}",f"lr={lr}",f"wd={wd}",f"seeds={seeds}"],capture_output=True,text=True).stdout
    try:
        r = json.loads(out)
    except Exception:
        return dict(batch=batch,tokens=tokens,lr=lr,wd=wd,err=out)
    return r
def sweep(cfgs, workers=8):
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: run(*c), cfgs))
    for r in res:
        if 'err' in r: print("ERR", r); continue
        print(f"B={r['batch']:5d} T={r['tokens_B']:5} lr={r['lr']:.5g} wd={r['wd']:.4g} seeds={r['seeds']} loss={r['final_val_loss']:.4f} per={r['per_seed']}")
    return res
if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    sweep([tuple(c) for c in cfgs])
