import json, numpy as np
R=[json.loads(l) for l in open('results.jsonl')]
def pts(B,T,wd): return sorted([(r['lr'],r['final_val_loss']) for r in R if r['batch']==B and r['tokens_B']==T and abs(r['wd']-wd)<1e-9])
def parab(p,a=None):
    x=np.log([q[0] for q in p]); y=np.array([q[1] for q in p])
    if a is None:
        c=np.polyfit(x,y,2); return np.exp(-c[1]/(2*c[0])), c[0]
    # fixed curvature: y = a x^2 + b x + c
    A=np.vstack([x,np.ones_like(x)]).T; b,c=np.linalg.lstsq(A,y-a*x**2,rcond=None)[0]
    return np.exp(-b/(2*a)), a
rows=[]
for B,T in [(256,3),(512,3),(256,12),(512,12),(512,48)]:
    p=pts(B,T,0.0976); o,a=parab(p); of,_=parab(p,0.11)
    steps=T*1e9/(B*4096); rows.append((B,T,steps,o,of)); print(B,T,int(steps),f"free opt {o:.5f} curv {a:.4f} | fixed-curv opt {of:.5f}")
rows.append((256,48,45776,0.00081,0.00081))
# fit log lr* = c0 + c1 log steps + c2 log(B/256)
X=np.array([[1,np.log(s),np.log(B/256)] for B,T,s,o,of in rows]); y=np.log([of for *_,of in rows])
c=np.linalg.lstsq(X,y,rcond=None)[0]; print("coef",c, "resid", (y-X@c))
pred=lambda s,B: np.exp(c[0]+c[1]*np.log(s)+c[2]*np.log(B/256))
print("pred lr* 96B/512:",pred(45776,512)," 384B/512:",pred(183105,512))
# B=512 only
m=[r for r in rows if r[0]==512]; X2=np.array([[1,np.log(s)] for B,T,s,o,of in m]); y2=np.log([of for *_,of in m]); c2=np.polyfit(np.log([s for _,_,s,_,_ in m]),y2,1); print("B512 slope",c2, "pred 96B:",np.exp(np.polyval(c2,np.log(45776))),"384B:",np.exp(np.polyval(c2,np.log(183105))))
