import json, numpy as np
rows=[json.loads(l)["result"] for l in open("/app/lab_log.jsonl")]
def get(B,D):
    return [(r['lr'],r['wd'],r['final_val_loss'],r['seeds']) for r in rows if r.get('batch')==B and r.get('tokens_B')==D and 'final_val_loss' in r]
def fit2d(B,D,verbose=True):
    d=np.array(get(B,D)); u=np.log(d[:,0]); v=np.log(d[:,1]); y=d[:,2]; w=np.sqrt(d[:,3])
    X=np.stack([np.ones_like(u),u,v,u*u,v*v,u*v],1)
    beta,*_=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)
    c0,c1,c2,a,b,c=beta
    H=np.array([[2*a,c],[c,2*b]]); g=np.array([c1,c2]); opt=-np.linalg.solve(H,g)
    Lmin=c0+g@opt+0.5*opt@H@opt
    res=y-X@beta
    if verbose:
        print(f"B={B} D={D} n={len(y)} lr*={np.exp(opt[0]):.5g} wd*={np.exp(opt[1]):.4g} Lmin={Lmin:.4f} a={a:.4f} b={b:.4f} c={c:.4f} rms_res={res.std():.4f}")
    return dict(lr=np.exp(opt[0]),wd=np.exp(opt[1]),L=Lmin,a=a,b=b,c=c,res=res)
if __name__=="__main__":
    import sys
    r=fit2d(4096,3.0); print(np.round(r['res'],4))
