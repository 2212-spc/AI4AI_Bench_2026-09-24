import numpy as np, json; from gfit import *; from nm import nelder_mead
np.random.seed(0)
rows=load(); Pg=unpack(np.array(json.load(open('params.json')))); R=Pg['R']
N=np.array([r['N'] for r in rows]); DT=np.array([deff_tokens(r['D'],r['mix'],r['pool'],R) for r in rows]); n=len(N)
ens={}
for e in EV:
    L=np.array([r['loss'][e] for r in rows]); ref={'general':0,'code':1,'math':2}[e]; others=[i for i in range(4) if i!=ref]
    lt0=np.log(Pg[e][5][others]); fits=[]
    for al in np.arange(0.15,0.50,0.0125):
        for be in np.arange(0.18,0.38,0.0125):
            def solve(lt):
                T=np.ones(4); T[others]=np.exp(lt)
                X=np.stack([np.ones_like(N),N**(-al),(DT@T)**(-be)],1)
                coef,*_=np.linalg.lstsq(X,L,rcond=None); r=X@coef-L; return np.sum(r*r),coef,T
            lt,v=nelder_mead(lambda z:solve(z)[0],lt0,step=0.15,iters=300); v,coef,T=solve(lt)
            if coef[0]<0: continue
            fits.append((al,be,v,coef,T))
    sse=np.array([f[2] for f in fits]); s2=sse.min()/(n-6)
    w=np.exp(-(sse-sse.min())/(2*s2)); w/=w.sum()
    ens[e]=(fits,w); 
    print(e,'sigma=%.4f  effective fits=%.1f'%(np.sqrt(s2),1/np.sum(w**2)))
def pred_e(e,mix,N_=2.5e9,D_=5e11):
    fits,w=ens[e]; dt=deff_tokens(D_,mix,None,R)
    return np.array([f[3][0]+f[3][1]*N_**(-f[0])+f[3][2]*(dt@f[4])**(-f[1]) for f in fits]),w
def comp_mean(mix):
    return sum(wt*np.sum(pred_e(e,mix)[0]*pred_e(e,mix)[1]) for e,wt in zip(EV,[0.34,0.33,0.33]))
# optimize mixture under posterior-mean composite
def obj(z):
    w=np.exp(z-z.max()); w/=w.sum(); return comp_mean(dict(zip(DOM,w)))
best=None
for s in range(4):
    z,v=nelder_mead(obj,np.random.randn(4)*0.5,step=0.5,iters=1500)
    if best is None or v<best[1]: best=(z,v)
w=np.exp(best[0]-best[0].max()); w/=w.sum(); MIX=dict(zip(DOM,[float(round(x,4)) for x in w]))
print('optimal mix',MIX,'mean comp %.4f'%best[1])
# regret check: for each ensemble sample, opt mixture vs MIX
# sample joint ensemble members
S=400; samp=[]
for e in EV:
    fits,wt=ens[e]; idx=np.random.choice(len(fits),S,p=wt); samp.append(idx)
def comp_member(k,mix):
    dt=deff_tokens(5e11,mix,None,R); tot=0
    for e,wt,idx in zip(EV,[0.34,0.33,0.33],samp):
        f=ens[e][0][idx[k]]; tot+=wt*(f[3][0]+f[3][1]*2.5e9**(-f[0])+f[3][2]*(dt@f[4])**(-f[1]))
    return tot
vals=np.array([comp_member(k,MIX) for k in range(S)])
print('composite at MIX: mean %.4f sd %.4f  q2.5 %.4f q50 %.4f q97.5 %.4f min %.4f max %.4f'%(vals.mean(),vals.std(),*np.quantile(vals,[0.025,0.5,0.975]),vals.min(),vals.max()))
regrets=[]
for k in range(0,S,10):
    def ob(z):
        ww=np.exp(z-z.max()); ww/=ww.sum(); return comp_member(k,dict(zip(DOM,ww)))
    z,v=nelder_mead(ob,best[0],step=0.3,iters=800)
    regrets.append(comp_member(k,MIX)-v)
print('regret of MIX across members: max %.4f mean %.4f'%(max(regrets),np.mean(regrets)))
for m in [{'web':0.5,'code':0.35,'math':0.03,'papers':0.12},{'web':0.45,'code':0.4,'math':0.03,'papers':0.12},{'web':0.55,'code':0.3,'math':0.03,'papers':0.12},{'web':0.5,'code':0.35,'math':0.05,'papers':0.10},{'web':0.5,'code':0.35,'math':0.015,'papers':0.135}]:
    print(m,'mean comp %.4f'%comp_mean(m))
json.dump({'mix':MIX,'vals':vals.tolist()},open('final_out.json','w'))
