import numpy as np, csv, json, collections
g=collections.defaultdict(list)
for r in csv.DictReader(open('/app/notebook/runs.csv')): g[(float(r['N']),float(r['D']))].append((float(r['lr']),float(r['loss'])))
for l in open('/app/lab_runs.jsonl'):
    x=json.loads(l); c=x['config']
    if x['status']=='ok' and c['seed']==0: g[(c['N'],c['D'])].append((c['lr'],x['loss']))
keys=sorted(g)
Ns=[];Ds=[];lrs=[];ys=[];idx=[]
for i,k in enumerate(keys):
    for lr,y in g[k]: Ns.append(k[0]);Ds.append(k[1]);lrs.append(lr);ys.append(y);idx.append(i)
Ns,Ds,lrs,ys,idx=map(np.array,(Ns,Ds,lrs,ys,idx))
lN=np.log(Ns/1e8); lD=np.log(Ds/2e9); llr=np.log(lrs); K=len(keys)

def nelder_mead(f,x0,step=0.1,iters=4000,tol=1e-12):
    n=len(x0); sim=[np.array(x0,float)]
    for i in range(n):
        x=np.array(x0,float); x[i]+=step if np.isscalar(step) else step[i]; sim.append(x)
    vals=[f(x) for x in sim]
    for it in range(iters):
        o=np.argsort(vals); sim=[sim[i] for i in o]; vals=[vals[i] for i in o]
        if abs(vals[-1]-vals[0])<tol and it>200: break
        cen=np.mean(sim[:-1],axis=0)
        xr=cen+(cen-sim[-1]); fr=f(xr)
        if fr<vals[0]:
            xe=cen+2*(cen-sim[-1]); fe=f(xe)
            if fe<fr: sim[-1],vals[-1]=xe,fe
            else: sim[-1],vals[-1]=xr,fr
        elif fr<vals[-2]: sim[-1],vals[-1]=xr,fr
        else:
            xc=cen+0.5*(sim[-1]-cen); fc=f(xc)
            if fc<vals[-1]: sim[-1],vals[-1]=xc,fc
            else:
                for i in range(1,len(sim)):
                    sim[i]=sim[0]+0.5*(sim[i]-sim[0]); vals[i]=f(sim[i])
    return sim[0],vals[0]

def shape(u,s):
    # asymmetric bowl: (exp(s u) - s u - 1)/s^2 -> u^2/2 as s->0
    if abs(s)<1e-6: return u*u/2
    return (np.exp(s*u)-s*u-1)/s**2

def design(theta, mode):
    # returns design matrix for linear params [L0_1..L0_K, k]
    if mode=='powerlaw':
        a,b,c,s=theta; u=llr-(a+b*lN+c*lD)
    A=np.zeros((len(ys),K+1))
    A[np.arange(len(ys)),idx]=1; A[:,K]=shape(u,s)
    return A
def sse(theta,mode='powerlaw'):
    A=design(theta,mode); beta,res,_,_=np.linalg.lstsq(A,ys,rcond=None)
    r=ys-A@beta; return (r*r).sum()

th0=[np.log(0.0017),-0.13,-0.21,0.5]
th,v=nelder_mead(lambda t: sse(t), th0, step=[0.1,0.05,0.05,0.3])
A=design(th,'powerlaw'); beta=np.linalg.lstsq(A,ys,rcond=None)[0]
r=ys-A@beta
print("theta a,b,c,s =",th.round(4)," k=%.4f"%beta[K]," rms resid=%.4f n=%d"%(np.sqrt((r*r).sum()/(len(ys)-K-5)),len(ys)))
a,b,c,s=th
for i,k in enumerate(keys):
    print("N=%.1e D=%.1e L0=%.4f lr*=%.5f"%(k[0],k[1],beta[i],np.exp(a+b*np.log(k[0]/1e8)+c*np.log(k[1]/2e9))))
np.save('/app/fit_theta.npy',np.r_[th,beta])
