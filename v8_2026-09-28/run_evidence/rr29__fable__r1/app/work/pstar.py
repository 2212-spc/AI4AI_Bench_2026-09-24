import json, numpy as np
R=[json.loads(l) for l in open('results.jsonl')]
a=0.1025; c0,c1,c2=-3.7020,-0.3190,0.4239
def lrstar(B,steps): return np.exp(c0+c1*np.log(steps)+c2*np.log(B/256))
rng=np.random.default_rng(0)
for B,T in [(256,12),(512,12),(256,48),(512,96)]:
    rs=[r for r in R if r['batch']==B and r['tokens_B']==T]
    steps=T*1e9/(B*4096); ls=lrstar(B,steps)
    x=np.array([np.log(r['lr'])-np.log(ls) for r in rs]); lnP=np.array([np.log(r['lr']*r['wd']*steps) for r in rs]); y=np.array([r['final_val_loss'] for r in rs])
    def fit(y):
        # y - a x^2 = L0 + aw*(lnP - m)^2 = L0 + aw lnP^2 - 2 aw m lnP + aw m^2 -> linear in [1, lnP, lnP^2]
        A=np.vstack([np.ones_like(lnP),lnP,lnP**2]).T; c=np.linalg.lstsq(A,y-a*x**2,rcond=None)[0]
        aw=c[2]; m=-c[1]/(2*aw); return m,aw
    m,aw=fit(y)
    ms=[]; 
    for _ in range(2000):
        mm,_a=fit(y+rng.normal(0,0.0016,len(y))); ms.append(mm)
    ms=np.array(ms); lo,hi=np.percentile(ms,[16,84])
    print(f"B={B} T={T} steps={steps:.0f} n={len(rs)}: P*={np.exp(m):.2f} [{np.exp(lo):.2f},{np.exp(hi):.2f}] (lnP*={m:.3f} ±{(hi-lo)/2:.3f}) a_w={aw:.4f}")
