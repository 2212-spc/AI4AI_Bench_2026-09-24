import numpy as np, json, sys; from gfit import *; from nm import nelder_mead
rows=load(); Pg=unpack(np.array(json.load(open('params.json')))); R=Pg['R']
N=np.array([r['N'] for r in rows]); 
DT=np.array([deff_tokens(r['D'],r['mix'],r['pool'],R) for r in rows])
MIX={'web':0.49,'code':0.37,'math':0.025,'papers':0.115}
dt_t=deff_tokens(5e11,MIX,None,R)
Ls={e:np.array([r['loss'][e] for r in rows]) for e in EV}
def fit_e(e,al,be):
    L=Ls[e]; ref={'general':0,'code':1,'math':2}[e]; others=[i for i in range(4) if i!=ref]
    lt0=np.log(Pg[e][5][others])
    def solve(lt):
        T=np.ones(4); T[others]=np.exp(lt)
        X=np.stack([np.ones_like(N),N**(-al),(DT@T)**(-be)],1)
        coef,*_=np.linalg.lstsq(X,L,rcond=None); r=X@coef-L; return np.sum(r*r),coef,T
    lt,v=nelder_mead(lambda z:solve(z)[0],lt0,step=0.2,iters=500)
    v,coef,T=solve(lt)
    return v,coef,coef[0]+coef[1]*2.5e9**(-al)+coef[2]*(dt_t@T)**(-be)
out=[]
for al in np.arange(0.20,0.46,0.025):
    for be in np.arange(0.20,0.36,0.025):
        tot=0; preds=[]; Es=[]
        for e in EV:
            v,coef,pr=fit_e(e,al,be); tot+=v; preds.append(pr); Es.append(coef[0])
        comp=0.34*preds[0]+0.33*preds[1]+0.33*preds[2]
        out.append((al,be,np.sqrt(tot/(3*len(N))),comp,*preds,min(Es)))
out=np.array(out); best=out[np.argmin(out[:,2])]
print('best shared: al=%.3f be=%.3f rmse=%.4f comp=%.3f g=%.3f c=%.3f m=%.3f minE=%.2f'%tuple(best))
for row in out[np.argsort(out[:,2])][:15]: print('  al=%.3f be=%.3f rmse=%.4f comp=%.3f g=%.3f c=%.3f m=%.3f minE=%.2f'%tuple(row))
