import json, numpy as np
rows=[json.loads(l)["result"] for l in open("/app/lab_log.jsonl")]
conds=sorted({(r['batch'],r['tokens_B']) for r in rows})
conds=[c for c in conds if sum(1 for r in rows if (r['batch'],r['tokens_B'])==c)>=5]
# y = q0_k + a u^2 + q1_k u + b_k z^2 + q2_k z, z=u+v
X=[];Y=[];W=[]
for r in rows:
    k=(r['batch'],r['tokens_B'])
    if k not in conds: continue
    u=np.log(r['lr']);v=np.log(r['wd']);z=u+v
    row=np.zeros(4*len(conds)+1); i=conds.index(k)
    row[4*i:4*i+4]=[1,u,z*z,z]; row[-1]=u*u
    X.append(row);Y.append(r['final_val_loss']);W.append(np.sqrt(r['seeds']))
X=np.array(X);Y=np.array(Y);W=np.array(W)
beta,*_=np.linalg.lstsq(X*W[:,None],Y*W,rcond=None)
a=beta[-1]; res=Y-X@beta
print(f"shared a={a:.4f} rms res={res.std():.4f} n={len(Y)}")
for i,k in enumerate(conds):
    q0,q1,b,q2=beta[4*i:4*i+4]; uk=-q1/(2*a); pk=-q2/(2*b); L=q0-q1*q1/(4*a)-q2*q2/(4*b)
    S=k[1]*1e9/(k[0]*4096); m=np.array([(r["batch"],r["tokens_B"])==k for r in rows if (r["batch"],r["tokens_B"]) in conds])
    print(f"B={k[0]:5d} D={k[1]:5} S={S:6.0f} n={sum(m):2d} lr*={np.exp(uk):.5g} b={b:.4f} (lr*wd)*={np.exp(pk):.4g} wd*={np.exp(pk-uk):.4g}  *S={np.exp(pk)*S:.3g} *D={np.exp(pk)*k[1]:.3g} Lmin={L:.4f} rms={res[m].std():.4f}")
