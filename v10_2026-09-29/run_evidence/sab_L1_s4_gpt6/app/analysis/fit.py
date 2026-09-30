from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path('/app')
hist=pd.read_csv(ROOT/'data/history.csv')
exper=pd.concat([pd.read_csv(p) for p in sorted((ROOT/'data').glob('abtest_*.csv'))],ignore_index=True)
# Arm generation times and ratings weighted by estimated served counts.
params={}
for arm in ['A','B']:
    d=exper[exper['requests_'+arm]>0]
    counts=d['requests_'+arm].to_numpy()*(1-d.abandon_rate.to_numpy())
    ratings=d['rating_'+arm].to_numpy(); service=d['gen_time_'+arm+'_s'].to_numpy()
    if arm=='A':
        counts=np.r_[counts,hist.requests*(1-hist.abandon_rate)]
        ratings=np.r_[ratings,hist.mean_rating];service=np.r_[service,hist.mean_gen_time_s]
    for key,x in [('rating',ratings),('service',service)]:
        params[key+'_'+arm]=float(np.average(x,weights=counts))
        print(arm,key,params[key+'_'+arm], 'row-based SE',np.sqrt(np.sum(counts**2*(x-params[key+'_'+arm])**2))/counts.sum())
# a = theta W, which holds for any service distribution with exponential patience.
allq=pd.concat([hist[['abandon_rate','mean_queue_wait_s']],exper[['abandon_rate','mean_queue_wait_s']]])
ratios=allq.abandon_rate.to_numpy()/allq.mean_queue_wait_s.to_numpy()
# geometric ratio reduces finite denominator-noise bias
params['theta']=float(np.exp(np.log(ratios).mean()))
print('theta',params['theta'],'patience',1/params['theta'])
# Balanced week log demand model. Poisson log-count bias is tiny, corrected.
e=exper.copy();e['requests']=e.requests_A+e.requests_B
traffic=pd.concat([hist[['day','dow','hour','requests']],e[['day','dow','hour','requests']]],ignore_index=True)
dows=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];di=np.array([dows.index(x) for x in traffic.dow]);hh=traffic.hour.to_numpy()
X=np.zeros((len(traffic),30));X[np.arange(len(traffic)),hh]=1
for j in range(1,7): X[:,23+j]=(di==j)
y=np.log(traffic.requests/3600)+.5/traffic.requests
# WLS treats the daily demand variation equally across hours.
beta=np.linalg.lstsq(X,y,rcond=None)[0];resid=y-X@beta
traffic['resid']=resid
# daily multiplicative variation; aggregate Poisson contribution is subtracted
byday=traffic.groupby('day').agg(resid=('resid','mean'),n=('requests','sum'))
sigma2=max(0,np.var(byday.resid,ddof=1)-np.mean(1/byday.n))
print('residual daily sd',sigma2**.5,'hour sd',np.std(resid),'resid day',byday.to_string())
logrates=beta[:24,None]+np.r_[0,beta[24:]][None,:]
params['logrates']=logrates.tolist();params['sigma_day']=float(sigma2**.5)
# Quadrature over per-day variation and all weekdays, each weekday equally frequent.
z,w=np.polynomial.hermite.hermgauss(15);w=w/np.sqrt(np.pi)
rates=np.exp(logrates[:,:,None]+np.sqrt(2*sigma2)*z[None,None,:])
weights=np.broadcast_to(w[None,:]/7,(7,15)).ravel()
np.savez(ROOT/'analysis/demand.npz',rates=rates.reshape(24,-1),weights=weights)
(ROOT/'analysis/parameters.json').write_text(json.dumps(params,indent=2)+'\n')
traffic.to_csv(ROOT/'analysis/traffic.csv',index=False)
print('daily expected requests',sum(np.sum(rates.reshape(24,-1)*weights,axis=1))*3600)
print('mean rates',np.sum(rates.reshape(24,-1)*weights,axis=1))
