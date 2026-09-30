import numpy as np
exec(open("/app/joint.py").read().split("def model_resid")[0])
d=np.load('/app/final_res.npy',allow_pickle=True).item(); keys=d['keys']
Nn=np.array([k[0] for k in keys]); Dd=np.array([k[1] for k in keys])
Lm=d['res']['expq'][1]
cnt=np.array([np.sum(G==i) for i in range(ng)]); sig=0.0075*(Nn/3e7)**(-0.35)/np.sqrt(cnt)
def fitlaw(y):
    def f(t):
        E,lA,al,lB,be=t; pred=E+np.exp(lA)*Nn**(-al)+np.exp(lB)*Dd**(-be); return np.sum(((y-pred)/sig)**2)
    t,fv=nm(f,[1.7,np.log(500),0.33,np.log(3600),0.37],[0.2,0.5,0.03,0.5,0.03],iters=5000)
    t,fv=nm(f,t,[0.05,0.2,0.01,0.2,0.01],iters=5000); return t,fv
t,fv=fitlaw(Lm); E,lA,al,lB,be=t
pred=lambda n,dd,t: t[0]+np.exp(t[1])*n**(-t[2])+np.exp(t[3])*dd**(-t[4])
r=Lm-pred(Nn,Dd,t); print('rms resid %.4f chi2/dof %.2f'%(np.sqrt(np.mean(r**2)),fv/(len(Lm)-5)))
print('L(1e9,1e12)=%.4f'%pred(1e9,1e12,t))
rng=np.random.default_rng(1); P=[]
for b in range(100):
    y=pred(Nn,Dd,t)+rng.standard_normal(len(Lm))*sig*max(1,np.sqrt(fv/(len(Lm)-5)))
    tb,_=fitlaw(y); P.append([pred(1e9,1e12,tb),tb[0],tb[2],tb[4]])
P=np.array(P); print('boot: L(1e9,1e12) sd %.4f  (E sd %.3f alpha sd %.3f beta sd %.3f)'%tuple(P.std(0)))
print('boot mean L %.4f  percentiles 5/95: %.4f %.4f'%(P[:,0].mean(),*np.percentile(P[:,0],[5,95])))
