import json, numpy as np
from lab import load
R = load()
def pts(batch, tokens):
    out = {}
    for r in R:
        if r["batch"]==batch and abs(r["tokens"]-tokens)<1e-9:
            k=(r["lr"], r["wd"]); out.setdefault(k, []).extend(r["result"]["per_seed"])
    return out
def fit2d(batch, tokens, cross=True):
    P = pts(batch, tokens)
    X=[]; Y=[]; W=[]
    for (lr,wd),ys in P.items():
        x=np.log(lr); z=np.log(wd)
        row=[1,x,z,x*x,z*z]+([x*z] if cross else [])
        X.append(row); Y.append(np.mean(ys)); W.append(len(ys))
    X=np.array(X); Y=np.array(Y); W=np.sqrt(np.array(W,float))
    n=len(Y)
    if n < X.shape[1]+1: return None
    beta,res,_,_=np.linalg.lstsq(X*W[:,None], Y*W, rcond=None)
    c0,cx,cz,cxx,czz=beta[:5]; cxz=beta[5] if cross else 0.0
    H=np.array([[2*cxx,cxz],[cxz,2*czz]]); g=np.array([cx,cz])
    opt=-np.linalg.solve(H,g)
    lmin=c0+g@opt+0.5*opt@H@opt
    resid=Y-X@beta; rms=np.sqrt(np.sum(resid**2)/max(1,n-X.shape[1]))
    return dict(n=n, lr=np.exp(opt[0]), wd=np.exp(opt[1]), lmin=lmin, a_lr=cxx, a_wd=czz, a_x=cxz, rms=rms)
if __name__=="__main__":
    for b,t in [(2048,1.5),(2048,6),(2048,12),(2048,24),(2048,48),(1024,6),(1024,12)]:
        for cross in (False,True):
            f=fit2d(b,t,cross)
            if f: print(b,t,"cross" if cross else "nocross", {k:(round(v,5) if isinstance(v,float) else v) for k,v in f.items()})

def trend():
    rows=[]
    for b,t in [(2048,6),(2048,12),(2048,24),(2048,48),(2048,96)]:
        f=fit2d(b,t,True) or fit2d(b,t,False)
        rows.append((t,f["lr"],f["wd"],f["lmin"],f["a_lr"],f["a_wd"],f["a_x"]))
    T=np.array([r[0] for r in rows]); LR=np.array([r[1] for r in rows]); WD=np.array([r[2] for r in rows])
    for name,Y in (("lr",LR),("wd",WD)):
        p=np.polyfit(np.log(T),np.log(Y),1)
        print(name,"exponent",p[0],"pred@384",np.exp(np.polyval(p,np.log(384))))
    print("lr*wd:",LR*WD)
