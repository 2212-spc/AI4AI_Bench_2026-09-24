import numpy as np, json, sys; from gfit import *; from nm import nelder_mead
rows=load(); Pg=unpack(np.array(json.load(open('params.json')))); R=Pg['R']
N=np.array([r['N'] for r in rows]); 
DT=np.array([deff_tokens(r['D'],r['mix'],r['pool'],R) for r in rows])  # rows x 4
MIX={'web':0.49,'code':0.37,'math':0.025,'papers':0.115}
dt_t=deff_tokens(5e11,MIX,None,R)
res={}
for ei,e in enumerate(EV):
    L=np.array([r['loss'][e] for r in rows]); ref={'general':0,'code':1,'math':2}[e]; others=[i for i in range(4) if i!=ref]
    T0=Pg[e][5]; lt0=np.log(T0[others])
    out=[]
    for al in np.arange(0.10,0.71,0.025):
        for be in np.arange(0.15,0.61,0.025):
            def solve(lt):
                T=np.ones(4); T[others]=np.exp(lt)
                X=np.stack([np.ones_like(N),N**(-al),(DT@T)**(-be)],1)
                coef,*_=np.linalg.lstsq(X,L,rcond=None)
                r=X@coef-L; return np.sum(r*r),coef,T
            lt,v=nelder_mead(lambda z:solve(z)[0],lt0,step=0.2,iters=600)
            v,coef,T=solve(lt)
            pred=coef[0]+coef[1]*2.5e9**(-al)+coef[2]*(dt_t@T)**(-be)
            out.append((al,be,np.sqrt(v/len(L)),coef[0],coef[1],coef[2],pred))
    out=np.array(out); res[e]=out
    best=out[np.argmin(out[:,2])]
    print(e,'best al=%.3f be=%.3f rmse=%.4f E=%.3f pred=%.3f'%(best[0],best[1],best[2],best[3],best[6]))
    thr=best[2]*1.15
    ok=out[(out[:,2]<=thr)&(out[:,3]>=0)]
    print('  plausible (rmse<=%.4f, E>=0): n=%d pred range %.3f..%.3f  al %.2f..%.2f be %.2f..%.2f'%(thr,len(ok),ok[:,6].min(),ok[:,6].max(),ok[:,0].min(),ok[:,0].max(),ok[:,1].min(),ok[:,1].max()))
    for row in ok[np.argsort(ok[:,6])][::max(1,len(ok)//8)]: print('   al=%.3f be=%.3f rmse=%.4f E=%.3f A=%.1f B=%.1f pred=%.3f'%tuple(row))
np.save('profile.npy',res,allow_pickle=True)
