"""Fit finished-run scaling separately from schedule-dependent checkpoint loss."""
import csv,json
import numpy as np

def minimize(fun,x,step,iterations=1500):
    x=np.array(x,dtype=float); p=[x]+[x+np.eye(len(x))[i]*s for i,s in enumerate(step)]
    vals=[fun(v) for v in p]
    for _ in range(iterations):
        ii=np.argsort(vals); p=[p[i] for i in ii]; vals=[vals[i] for i in ii]
        if np.max(np.abs(np.array(p)-p[0]))<1e-10: break
        c=np.mean(p[:-1],axis=0); r=2*c-p[-1]; fr=fun(r)
        if fr<vals[0]:
            e=c+2*(r-c); fe=fun(e)
            p[-1],vals[-1]=(e,fe) if fe<fr else (r,fr)
        elif fr<vals[-2]: p[-1],vals[-1]=r,fr
        else:
            co=c+.5*((r if fr<vals[-1] else p[-1])-c); fc=fun(co)
            if fc<min(fr,vals[-1]): p[-1],vals[-1]=co,fc
            else:
                p=[p[0]]+[p[0]+.5*(v-p[0]) for v in p[1:]]; vals=[fun(v) for v in p]
    return p[np.argmin(vals)]

nbck=json.load(open('/app/notebook/checkpoints.json'))
runs=[]
for row,cks in zip(csv.DictReader(open('/app/notebook/runs.csv')),nbck):
    runs.append(dict(config={'N':float(row['N']),'D':float(row['D']),'sched':row['sched']},loss=float(row['loss']),checkpoints=cks['checkpoints']))
runs += [json.loads(s) for s in open('/app/lab_runs.jsonl')]
finished=[]; ck=[]
for r in runs:
    n=r['config']['N']; d=r['config']['D']; sched=r['config']['sched']
    finished.append([n,d,r['loss']])
    for c in r.get('cooldown_branches') or []: finished.append([n,c['tokens'],c['loss']])
    for c in r.get('checkpoints') or []:
        if c['frac']==1: finished.append([n,c['tokens'],c['loss']])
        else: ck.append([n,c['tokens'],c['loss'],c['frac'],sched])
finished=np.array(finished)
n,d,y=finished.T

def basefit(exps):
    a,b=exps
    X=np.array([np.ones(len(n)),(n/1e8)**-a,(d/1e9)**-b]).T
    co=np.linalg.lstsq(X,y,rcond=None)[0]
    return np.mean((X@co-y)**2),co
ab=minimize(lambda ex:basefit(ex)[0],[.4,.33],[.03,.03])
err,co=basefit(ab)
E,A,B=co;a,b=ab

def L(n,d):return E+A*(np.array(n)/1e8)**-a+B*(np.array(d)/1e9)**-b

if __name__=='__main__':
    print('FINISHED ONLY',dict(E=E,A=A,B=B,alpha=a,beta=b,rmse=err**.5))
    for r in runs:
        n=r['config']['N'];d=r['config']['D'];s=r['config']['sched']
        print('run',n,d,s,'final residual',r['loss']-L(n,d))
        print('checkpoint residuals',[(c['frac'],round(c['loss']-L(n,c['tokens']),5)) for c in r.get('checkpoints') or []])
    print('Production final:',L(3e9,6e10))
    print('Candidates',[(n,L(n,1.08e21/(6*n))) for n in [8.9e8,1.8e9,3.6e9,7.1e9]])
    for s in ['wsd','cosine']:
        rows=np.array([[f,l-L(n,d)] for n,d,l,f,sched in ck if sched==s]); f,delta=rows.T
        phase=np.minimum(1,(1-f)/.2) if s=='wsd' else .5*(1+np.cos(np.pi*f))
        k=np.dot(phase,delta)/np.dot(phase,phase)
        print(s,'penalty coefficient',k,'rmse',np.mean((delta-k*phase)**2)**.5)
