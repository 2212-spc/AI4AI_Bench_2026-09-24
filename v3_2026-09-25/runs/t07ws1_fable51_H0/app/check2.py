import numpy as np
exec(open('fit.py').read().split("np.random.seed(0)")[0])
mc=np.array([s=='cosine' for s in S]); mw=~mc; mdec=mw&(F>0.8)&(F<1.0)
p6,_=fit('raw',6); p7,_=fit('raw',7)
for name,p in [('6p',p6),('7p',p7)]:
    r=model(p,N,D,F,LR,AR,'raw')-L
    print(name,'rmse cosine %.4f wsd-stable/final %.4f wsd-decay(n=%d) %.4f'%(np.sqrt(np.mean(r[mc]**2)),np.sqrt(np.mean(r[mw&~mdec]**2)),mdec.sum(),np.sqrt(np.mean(r[mdec]**2))))
    print('   wsd-decay residuals by f:',{f:round(float(r[mdec&(F==f)].mean()),4) for f in sorted(set(F[mdec]))})
# fit excluding wsd decay-phase checkpoints
mask=~mdec
for npar in [6,7]:
    p,v=fit('raw',npar,mask); print('excl decay',npar,np.round(p,4),'rmse %.4f'%np.sqrt(v))
    exec(open('final.py').read().split("for mode,npar")[0].split("np.random.seed(1)")[1])
    q1,q2,q3,q4,q5=answers(p); print('   q2 %.4f q3 %.4f q4 [%.4f,%.4f] q5 %.4f'%(q2,q3,q4[0],q4[1],q5), {k:round(x,4) for k,x in q1.items()})
