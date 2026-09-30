import numpy as np, json; from gfit import *; from nm import nelder_mead
rows=load(); Pg=unpack(np.array(json.load(open('params.json')))); R=Pg['R']
N=np.array([r['N'] for r in rows]); DT=np.array([deff_tokens(r['D'],r['mix'],r['pool'],R) for r in rows])
M={'web':0.3,'code':0.3,'math':0.3,'papers':0.1}
cands=[(4e8,1e9),(3e8,1.5e9),(4e8,1.2e9),(2e8,2.4e9),(5e7,9e9)]
for e in EV:
    L=np.array([r['loss'][e] for r in rows]); ref={'general':0,'code':1,'math':2}[e]; others=[i for i in range(4) if i!=ref]
    lt0=np.log(Pg[e][5][others]); fits=[]
    for al in np.arange(0.20,0.46,0.025):
        for be in np.arange(0.20,0.36,0.025):
            def solve(lt):
                T=np.ones(4); T[others]=np.exp(lt)
                X=np.stack([np.ones_like(N),N**(-al),(DT@T)**(-be)],1)
                coef,*_=np.linalg.lstsq(X,L,rcond=None); r=X@coef-L; return np.sum(r*r),coef,T
            lt,v=nelder_mead(lambda z:solve(z)[0],lt0,step=0.2,iters=400); v,coef,T=solve(lt)
            fits.append((al,be,np.sqrt(v/len(L)),coef,T))
    best=min(f[2] for f in fits); ok=[f for f in fits if f[2]<=best*1.15 and f[3][0]>=0]
    print(e,len(ok))
    for (n,d) in cands:
        dt=deff_tokens(d,M,None,R)
        preds=[f[3][0]+f[3][1]*n**(-f[0])+f[3][2]*(dt@f[4])**(-f[1]) for f in ok]
        tg=[f[3][0]+f[3][1]*2.5e9**(-f[0])+f[3][2]*(deff_tokens(5e11,{'web':0.49,'code':0.37,'math':0.025,'papers':0.115},None,R)@f[4])**(-f[1]) for f in ok]
        c=np.corrcoef(preds,tg)[0,1]
        print('  N=%.1e D=%.1e spread=%.4f corr_with_target=%.2f'%(n,d,max(preds)-min(preds),c))
