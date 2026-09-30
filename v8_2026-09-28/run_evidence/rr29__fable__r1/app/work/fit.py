import json, numpy as np
R=[json.loads(l) for l in open('results.jsonl')]
def get(B,T,fix,val):
    return sorted([(r['lr'] if fix=='wd' else r['wd'], r['final_val_loss']) for r in R if r['batch']==B and r['tokens_B']==T and abs(r[fix]-val)<1e-9])
def parab(pts):
    x=np.log([p[0] for p in pts]); y=np.array([p[1] for p in pts])
    A=np.vstack([x**2,x,np.ones_like(x)]).T
    c=np.linalg.lstsq(A,y,rcond=None)[0]
    xo=-c[1]/(2*c[0]); return np.exp(xo), c[0], c[2]-c[1]**2/(4*c[0])
for B,T,fix,val in [(256,3,'wd',0.0976),(512,3,'wd',0.0976),(256,12,'wd',0.0976),(512,12,'wd',0.0976),
                    (256,3,'lr',0.0023),(256,12,'lr',0.00113),(512,12,'lr',0.0023)]:
    pts=get(B,T,fix,val)
    if len(pts)>=3:
        o,a,m=parab(pts); steps=T*1e9/(B*4096)
        print(f"B={B} T={T} fixed {fix}={val}: n={len(pts)} opt={o:.5f} curv={a:.4f} min={m:.4f} steps={steps:.0f}", [f"{p[0]:.5g}:{p[1]:.4f}" for p in pts])
