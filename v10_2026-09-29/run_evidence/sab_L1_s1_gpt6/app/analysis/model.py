import numpy as np
import pandas as pd
from pathlib import Path

ETA = .0062

def load():
    hist = pd.read_csv('/app/data/history.csv')
    tests = pd.concat([pd.read_csv(p) for p in sorted(Path('/app/data').glob('abtest_*.csv'))],ignore_index=True)
    hh = hist.copy()
    hh['fraction_B']=0.
    hh['rating_A']=hh.mean_rating
    hh['rating_B']=np.nan
    hh['gen_time_A_s']=hh.mean_gen_time_s
    hh['gen_time_B_s']=np.nan
    hh['requests_A']=hh.requests
    hh['requests_B']=0
    tests['requests']=tests.requests_A+tests.requests_B
    return pd.concat([hh,tests],ignore_index=True)

def parameters(data):
    p={}
    for arm in ['A','B']:
        z=data[data['requests_'+arm]>0]
        w=z['requests_'+arm]*(1-z.abandon_rate)
        p['rating_'+arm]=np.average(z['rating_'+arm],weights=w)
        p['gen_'+arm]=np.average(z['gen_time_'+arm+'_s'],weights=w)
    # exponential patience identity P(abandon) = theta E[Wq] holds for mixtures too
    p['theta']=data.abandon_rate.sum()/data.mean_queue_wait_s.sum()
    return p

def erlang(lam,service,replicas,theta):
    lam,service,replicas=np.broadcast_arrays(lam,service,replicas)
    shape=lam.shape
    lam=lam.ravel(); service=service.ravel(); replicas=replicas.ravel()
    logp=np.zeros((200,len(lam)))
    for n in range(1,len(logp)):
        logp[n]=logp[n-1]+np.log(lam)-np.log(np.minimum(n,replicas)/service+np.maximum(n-replicas,0)*theta)
    prob=np.exp(logp-logp.max(axis=0));prob/=prob.sum(axis=0)
    q=np.maximum(np.arange(len(logp))[:,None]-replicas,0)
    w=(prob*q).sum(axis=0)/lam
    return (theta*w).reshape(shape),w.reshape(shape)

def score(lam,f,p):
    service=p['gen_A']+(p['gen_B']-p['gen_A'])*f
    c=np.clip(np.ceil(lam*service/.75),4,24)
    a,w=erlang(lam,service,c,p['theta'])
    rating=p['rating_A']+(p['rating_B']-p['rating_A'])*f
    return rating*(1-a)-ETA*w

if __name__=='__main__':
    data=load();p=parameters(data)
    print(p)
    a,w=erlang(data.requests.to_numpy()/3600,p['gen_A']+(p['gen_B']-p['gen_A'])*data.fraction_B.to_numpy(),data.replicas.to_numpy(),p['theta'])
    for f in [0,.3,1]:
        z=data.fraction_B==f
        print('f',f,'abandon actual,pred',data.loc[z,'abandon_rate'].mean(),a[z].mean(),'wait actual,pred',data.loc[z,'mean_queue_wait_s'].mean(),w[z].mean())
    grid=np.linspace(0,1,1001)
    schedule=[]
    for hour in range(24):
        d=data[data.hour==hour]
        l=d.requests.to_numpy()/3600
        sc=score(l[:,None],grid[None,:],p)
        avg=np.average(sc,axis=0,weights=l)
        best=grid[np.argmax(avg)]
        schedule.append(best)
        print(f'{hour:02d} {best:.3f} gain {avg.max()-avg[0]:.6f}')
    for name,s in [('allA',np.zeros(24)),('allB',np.ones(24)),('schedule',np.array(schedule))]:
        scores=score(data.requests.to_numpy()/3600,s[data.hour.to_numpy()],p)
        print(name,np.average(scores,weights=data.requests))
    np.savez('/app/analysis/fitted.npz',schedule=schedule,**p)
