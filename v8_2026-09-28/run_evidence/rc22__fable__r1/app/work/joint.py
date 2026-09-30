import json, numpy as np
rows=[json.loads(l)["result"] for l in open("/app/lab_log.jsonl")]
conds=sorted({(r['batch'],r['tokens_B']) for r in rows})
conds=[c for c in conds if sum(1 for r in rows if (r['batch'],r['tokens_B'])==c)>=5]
# model: y = L_k + a(u-u_k)^2 + b(v-v_k)^2 + c(u-u_k)(v-v_k); linearize: y = q0_k + q1_k u + q2_k v + a u^2 + b v^2 + c uv
X=[];Y=[];W=[]
for r in rows:
    k=(r['batch'],r['tokens_B'])
    if k not in conds: continue
    u=np.log(r['lr']);v=np.log(r['wd'])
    row=np.zeros(3*len(conds)+3); i=conds.index(k)
    row[3*i:3*i+3]=[1,u,v]; row[-3:]=[u*u,v*v,u*v]
    X.append(row);Y.append(r['final_val_loss']);W.append(np.sqrt(r['seeds']))
X=np.array(X);Y=np.array(Y);W=np.array(W)
beta,*_=np.linalg.lstsq(X*W[:,None],Y*W,rcond=None)
a,b,c=beta[-3:]; res=Y-X@beta
print(f"shared a={a:.4f} b={b:.4f} c={c:.4f}  rms res={res.std():.4f}  n={len(Y)}")
H=np.array([[2*a,c],[c,2*b]])
out={}
for i,k in enumerate(conds):
    q0,q1,q2=beta[3*i:3*i+3]; g=np.array([q1,q2]); opt=-np.linalg.solve(H,g)
    L=q0+g@opt+0.5*opt@H@opt
    n=sum(1 for r in rows if (r['batch'],r['tokens_B'])==k)
    S=k[1]*1e9/(k[0]*4096)
    out[k]=(np.exp(opt[0]),np.exp(opt[1]),L)
    print(f"B={k[0]:5d} D={k[1]:5} S={S:7.0f} n={n:2d} lr*={np.exp(opt[0]):.5g} wd*={np.exp(opt[1]):.4g} lr*wd*={np.exp(opt.sum()):.3g} lr*wd*S={np.exp(opt.sum())*S:.3g} Lmin={L:.4f}")
json.dump({f"{k[0]},{k[1]}":v for k,v in out.items()},open('/app/work/opts.json','w'))
