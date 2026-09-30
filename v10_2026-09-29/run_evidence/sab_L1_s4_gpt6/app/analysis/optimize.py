import numpy as np,json
from pathlib import Path
from queue_model import erlang_a,hyperexp
ROOT=Path('/app'); p=json.loads((ROOT/'analysis/parameters.json').read_text());dd=np.load(ROOT/'analysis/demand.npz');rates=dd['rates'];weights=dd['weights']
sa,sb,ra,rb,th=[p[x] for x in ['service_A','service_B','rating_A','rating_B','theta']]
def scores(lam,f,exact=False):
    mean=sa+(sb-sa)*f;c=np.clip(np.ceil(lam*mean/.75),4,16).astype(int)
    if exact:
        flat=[hyperexp(float(l),float(f),int(cc),th,sa,sb) for l,cc in zip(np.ravel(lam),np.ravel(c))]
        ab=np.array([x[0] for x in flat]).reshape(lam.shape);w=np.array([x[1] for x in flat]).reshape(lam.shape)
    else:ab,w=erlang_a(lam,mean,c,th)
    return (1-ab)*(ra+(rb-ra)*f)-.008*w

def hour_score(hour,f,exact=False):
    l=rates[hour];return np.sum(weights*l*scores(l,f,exact))/np.sum(weights*l)
def value(sched,exact=False):
    hs=np.array([hour_score(h,f,exact) for h,f in enumerate(sched)]);tw=rates@weights
    return float(np.average(hs,weights=tw))
if __name__=='__main__':
    grid=np.linspace(0,1,1001);fs=[];maxs=[]
    for h in range(24):
        l=rates[h,None,:];ss=scores(l,grid[:,None]);v=(ss*l*weights).sum(axis=1)/(l*weights).sum()
        i=int(np.argmax(v));fs.append(float(grid[i]));maxs.append(float(v[i]));print(h,fs[-1],v[i],flush=True)
    base=value([0]*24);q1=value([1]*24)-base;q2=value(fs)-base
    print('base',base,'Q1',q1,'Q2 proxy',q2,'fracB',np.average(fs,weights=rates@weights))
    (ROOT/'analysis/proxy_result.json').write_text(json.dumps({'schedule':fs,'base':base,'Q1':q1,'Q2_proxy':q2},indent=2)+'\n')
