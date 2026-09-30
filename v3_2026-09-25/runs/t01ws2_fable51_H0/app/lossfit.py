import numpy as np
d=np.load('/app/final_res.npy',allow_pickle=True).item(); keys=d['keys']
exec(open("/app/joint.py").read().split("def model_resid")[0])
Nn=np.array([k[0] for k in keys]); Dd=np.array([k[1] for k in keys])
for shape in ['expq','expqN','asym']:
    Lm=d['res'][shape][1]
    sig=(Nn/3e7)**(-0.35)/np.sqrt(np.array([5]*len(keys)))  # rough
    def f(t):
        E,lA,al,lB,be=t; pred=E+np.exp(lA)*Nn**(-al)+np.exp(lB)*Dd**(-be); return np.sum(((Lm-pred)/sig)**2)
    best=None
    for al0 in [0.2,0.3,0.4]:
        for be0 in [0.2,0.3,0.4]:
            t,fv=nm(f,[1.5,np.log(300),al0,np.log(300),be0],[0.3,1,0.05,1,0.05],iters=6000)
            t,fv=nm(f,t,[0.1,0.3,0.02,0.3,0.02],iters=6000)
            if best is None or fv<best[1]: best=(t,fv)
    t,fv=best; E,lA,al,lB,be=t
    pred=lambda n,dd: E+np.exp(lA)*n**(-al)+np.exp(lB)*dd**(-be)
    print(shape,'E=%.3f A=%.1f alpha=%.3f B=%.1f beta=%.3f  chi2=%.2f'%(E,np.exp(lA),al,np.exp(lB),be,fv))
    print('   resid',np.round(Lm-pred(Nn,Dd),4))
    print('   L(1e9,1e12)=%.4f  L(1e9,2e10)=%.4f L(1e9,2e11)=%.4f'%(pred(1e9,1e12),pred(1e9,2e10),pred(1e9,2e11)))
    # leave-one-out sensitivity / alternative: fit in exp space with A*N^-a form also with multiplicative? check E fixed alternatives
    for Efix in [1.0,1.5,2.0,2.5]:
        def f2(t):
            lA,al,lB,be=t; pred=Efix+np.exp(lA)*Nn**(-al)+np.exp(lB)*Dd**(-be); return np.sum(((Lm-pred)/sig)**2)
        t2,fv2=nm(f2,[np.log(300),0.3,np.log(300),0.3],[1,0.05,1,0.05],iters=6000); t2,fv2=nm(f2,t2,[0.3,0.02,0.3,0.02],iters=6000)
        lA,al,lB,be=t2; print('   Efix=%.1f chi2=%.2f alpha=%.3f beta=%.3f L(1e9,1e12)=%.4f'%(Efix,fv2,al,be,Efix+np.exp(lA)*1e9**(-al)+np.exp(lB)*1e12**(-be)))
