import json, numpy as np
rows=[json.loads(l) for l in open('lab_log.jsonl')]
D=[]
for r in rows:
    a=r.get('args',r); res=r.get('result',r)
    if 'batch' not in a: continue
    D.append((a['batch'],a['tokens'],a['lr'],a['wd'],res['final_val_loss'],res['steps']))
D=np.array(D,float)
print(len(D),"runs")
# per-(B,T) local quadratic fits in (u=ln lr, v=ln wd)
for (B,T) in sorted(set(map(tuple,D[:,:2]))):
    m=(D[:,0]==B)&(D[:,1]==T); d=D[m]
    print(f"B={int(B)} T={T} steps={int(d[0,5])} n={len(d)}")
    for x in d: print(f"   lr={x[2]:.5g} wd={x[3]:.3g} loss={x[4]:.4f}")

print("\n=== 2D quadratic fit at B=4096 T=2 ===")
m=(D[:,0]==4096)&(D[:,1]==2); d=D[m]
u=np.log(d[:,2]); v=np.log(d[:,3]); y=d[:,4]
X=np.c_[np.ones_like(u),u,v,u*u,v*v,u*v]
coef,res,_,_=np.linalg.lstsq(X,y,rcond=None)
a,b,c,dd,e,f=coef
H=np.array([[2*dd,f],[f,2*e]]); g=np.array([b,c])
opt=-np.linalg.solve(H,g)
print("coef",coef); print("opt lr,wd:",np.exp(opt)); print("Hessian",H)
print("resid rms",np.sqrt(np.mean((X@coef-y)**2)))
# eigen
w,V=np.linalg.eigh(H); print("eig",w,"\n",V)
