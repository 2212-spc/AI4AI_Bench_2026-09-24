import numpy as np,json
DOM=['web','code','math','papers']; EV=['general','code','math']
def load():return [json.loads(l) for l in open('/app/runs.jsonl')]
def lm(fun,x,maxiter=250):
    x=np.array(x,float);r=fun(x);v=r@r;lam=.001
    for it in range(maxiter):
        J=np.empty((len(r),len(x)))
        for j in range(len(x)):
            h=1e-5*(1+abs(x[j]));xx=x.copy();xx[j]+=h;J[:,j]=(fun(xx)-r)/h
        g=J.T@r;H=J.T@J
        if np.max(abs(g))<1e-10:break
        accepted=False
        for t in range(20):
            dx=np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-8)), -g)
            xx=x+dx
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):rr=fun(xx);vv=rr@rr
            if np.isfinite(vv) and vv<v:
                x,r,v=xx,rr,vv;lam=max(lam/3,1e-10);accepted=True;break
            lam*=5
        if not accepted:break
        if np.linalg.norm(dx)<1e-8:break
    return x,v

def pred(p,n,t):
    E,A,alpha,beta=p[:4];c=np.exp(p[4:]);return E+A*n**(-alpha)+(t@c)**(-beta)
if __name__=='__main__':
    rs=[r for r in load() if 'pool' not in r]
    n=np.array([r['N']/1e8 for r in rs]);t=np.array([[r['D']/1e9*r['mix'][d] for d in DOM] for r in rs]);y=np.array([[r['eval_loss'][e] for e in EV] for r in rs])
    ps=[]
    for e in range(3):
        p,v=lm(lambda p:pred(p,n,t)-y[:,e],[1,.6,.3,.3,0,0,0,0]);ps.append(p)
        print(EV[e],p[:4],np.exp(p[4:]),'rms',np.sqrt(v/len(rs)))
        print('residuals',np.round(pred(p,n,t)-y[:,e],4))
    np.save('base_params.npy',ps)
