import json, numpy as np
R=[json.loads(l) for l in open('results.jsonl')]
rng=np.random.default_rng(1)
def est(rs, steps, excl_wd=(), note=""):
    rs=[r for r in rs if r['wd'] not in excl_wd]
    lnP=np.array([np.log(r['lr']*r['wd']*steps) for r in rs]); y=np.array([r['final_val_loss'] for r in rs])
    A=np.vstack([np.ones_like(lnP),lnP,lnP**2]).T
    def f(y):
        c=np.linalg.lstsq(A,y,rcond=None)[0]; return -c[1]/(2*c[2]), c[2]
    m,aw=f(y); ms=np.array([f(y+rng.normal(0,0.0016,len(y)))[0] for _ in range(3000)])
    lo,hi=np.percentile(ms,[16,84]); print(f"{note} n={len(rs)} P*={np.exp(m):.2f} [{np.exp(lo):.2f},{np.exp(hi):.2f}] a_w={aw:.4f}")
for B,T,lr0 in [(256,12,0.00113),(512,12,0.0023),(256,48,0.00081),(512,96,0.0011)]:
    steps=T*1e9/(B*4096)
    rs=[r for r in R if r['batch']==B and r['tokens_B']==T and abs(r['lr']-lr0)<1e-9]
    est(rs,steps,note=f"B={B} T={T} on-axis all wd")
    if T==12: est(rs,steps,excl_wd=(1.0,),note=f"B={B} T={T} on-axis excl wd=1.0"); est(rs,steps,excl_wd=(0.03,),note=f"B={B} T={T} on-axis excl wd=0.03")
