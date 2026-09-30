import json, numpy as np
P=np.load('pts.npy'); N,T,LR,L=P.T; tag=np.array(json.load(open('tags.json')))
def lr_cos(f): return 0.1+0.9*0.5*(1+np.cos(np.pi*f))
def solve(a,b,p,mask):
    X=np.c_[np.ones(mask.sum()), N[mask]**-a, T[mask]**-b, LR[mask]**p]
    coef=np.linalg.lstsq(X,L[mask],rcond=None)[0]; r=L[mask]-X@coef
    return (r**2).sum(), coef
def quantities(a,b,p,coef):
    E,A,B,c=coef
    Lfin=lambda n,d,lr: E+A*n**-a+B*d**-b+c*lr**p
    q1={k:Lfin(n,1.08e21/(6*n),0.1) for k,n in zip('ABCD',[1.2e9,2.4e9,4.8e9,9.6e9])}
    q2=Lfin(3e9,6e10,0.0)
    q3=c*lr_cos(0.4)**p
    fs=np.linspace(0.5,0.9,401)
    q4=[B*(6e10)**-b*(f**-b-1)+c*(lr_cos(f)**p-0.1**p) for f in fs]
    q5=B*(6e10)**-b*(0.7**-b-0.4**-b)+c*lr_cos(0.7)**p
    return q1,q2,q3,(min(q4),max(q4)),q5, Lfin(3e9,6e10,0.1)
masks={'all':np.ones(len(L),bool),'no_ckpt':np.isin(tag,['fin_wsd','fin_cosine','cool']), 'no_cos_ckpt':tag!='ck_cosine'}
for name,mask in masks.items():
    res=[]
    for a in np.linspace(0.30,0.50,81):
      for b in np.linspace(0.28,0.42,71):
        for p in [1.0,1.05,1.1,1.15,1.2,1.25,1.3]:
          s,coef=solve(a,b,p,mask); res.append((s,a,b,p,coef))
    res.sort(key=lambda x:x[0]); s0=res[0][0]; n=mask.sum(); sig2=s0/n
    print(f'\n== {name}: n={n} rms={np.sqrt(sig2):.4f} best a={res[0][1]:.4f} b={res[0][2]:.4f} p={res[0][3]} E,A,B,c={np.round(res[0][4],4)}')
    q1,q2,q3,q4,q5,cosfin=quantities(*res[0][1:])
    print('  q1',{k:round(v,4) for k,v in q1.items()},' q2 %.4f  q3 %.4f  q4 (%.4f,%.4f)  q5 %.4f  cos_final %.4f'%(q2,q3,*q4,q5,cosfin))
    # profile: sets within delta chi2 <= 4 (i.e. SSE <= s0 + 4*sig2)
    ok=[r for r in res if r[0]<=s0+4*sig2]
    Q=[quantities(*r[1:]) for r in ok]
    print('  n sets within 2sigma:',len(ok),' a range',min(r[1] for r in ok),max(r[1] for r in ok),' b range',min(r[2] for r in ok),max(r[2] for r in ok),' p',sorted(set(r[3] for r in ok)))
    print('  q2 range',round(min(q[1] for q in Q),4),round(max(q[1] for q in Q),4))
    print('  q3 range',round(min(q[2] for q in Q),4),round(max(q[2] for q in Q),4))
    print('  q4 lo range',round(min(q[3][0] for q in Q),4),round(max(q[3][0] for q in Q),4),' hi range',round(min(q[3][1] for q in Q),4),round(max(q[3][1] for q in Q),4))
    print('  q5 range',round(min(q[4] for q in Q),4),round(max(q[4] for q in Q),4))
    print('  q1 best choice over sets:',{k:sum(1 for q in Q if min(q[0],key=q[0].get)==k) for k in 'ABCD'})
    print('  q1 B-A margin range',round(min(q[0]['B']-q[0]['A'] for q in Q),4),round(max(q[0]['B']-q[0]['A'] for q in Q),4),' C-B',round(min(q[0]['C']-q[0]['B'] for q in Q),4),round(max(q[0]['C']-q[0]['B'] for q in Q),4))
