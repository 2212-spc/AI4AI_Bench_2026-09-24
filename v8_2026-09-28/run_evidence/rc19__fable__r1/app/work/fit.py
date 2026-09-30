import json, numpy as np
R=[json.loads(l) for l in open('results.jsonl')]
groups=sorted(set((r['batch'],r['tokens']) for r in R))
gi={g:i for i,g in enumerate(groups)}
X=[]; 
for r in R:
    B,T=r['batch'],r['tokens']; S=round(T*1e9/(B*4096))
    X.append((gi[(B,T)],np.log(r['lr']),np.log(r['lr']*r['wd']*S),np.log(S),r['result']['final_val_loss']))
X=np.array(X); G=len(groups)
# params: base[G], lnlr*[G], k_lr, k_lr3 (asym), lnc, p, k_wd[G], c3
def unpack(th):
    base=th[:G]; lnl=th[G:2*G]; k_lr,k3,lnc,p=th[2*G:2*G+4]; kwd=th[2*G+4:3*G+4]; c3=th[3*G+4]
    return base,lnl,k_lr,k3,lnc,p,kwd,c3
def model(th,X):
    base,lnl,k_lr,k3,lnc,p,kwd,c3=unpack(th)
    g=X[:,0].astype(int); u=X[:,1]-lnl[g]; v=X[:,2]-(lnc+p*X[:,3])
    return base[g]+k_lr*u**2+k3*u**3+kwd[g]*v**2+c3*v**3
def resid(th): return model(th,X)-X[:,4]
# simple Gauss-Newton / LM by hand
def lm(th,f,iters=200):
    lam=1e-3
    for it in range(iters):
        r=f(th); J=np.zeros((len(r),len(th)))
        for j in range(len(th)):
            d=np.zeros(len(th)); d[j]=1e-6; J[:,j]=(f(th+d)-r)/1e-6
        A=J.T@J; g=J.T@r
        while True:
            step=np.linalg.solve(A+lam*np.diag(np.diag(A)+1e-12),-g)
            r2=f(th+step)
            if r2@r2<r@r: th=th+step; lam*=0.3; break
            lam*=10
            if lam>1e8: return th
    return th
th0=np.concatenate([np.full(G,3.5),np.full(G,np.log(0.01)),[0.1,0.01,np.log(0.0013),0.6],np.full(G,0.005),[0.002]])
th=lm(th0,resid)
base,lnl,k_lr,k3,lnc,p,kwd,c3=unpack(th)
r=resid(th); print("rms resid %.4f  n=%d"%(np.sqrt(np.mean(r**2)),len(r)))
print("k_lr=%.4f k3=%.4f  lambda*: c=%.5f p=%.3f  c3=%.4f"%(k_lr,k3,np.exp(lnc),p,c3))
for g,(B,T) in enumerate(groups):
    S=round(T*1e9/(B*4096))
    print("B=%4d T=%4.1f S=%5d base=%.4f lr*=%.5f k_wd=%.4f  lambda*=%.3f wd*@lr*=%.4f"%(B,T,S,base[g],np.exp(lnl[g]),kwd[g],np.exp(lnc+p*np.log(S)),np.exp(lnc+p*np.log(S))/(np.exp(lnl[g])*S)))
print("worst residuals:")
idx=np.argsort(-np.abs(r))[:6]
for i in idx: print(groups[int(X[i,0])], "lr=%.4g lam=%.3g loss=%.4f pred=%.4f"%(np.exp(X[i,1]),np.exp(X[i,2]),X[i,4],X[i,4]+r[i]))
np.save('theta.npy',th)
