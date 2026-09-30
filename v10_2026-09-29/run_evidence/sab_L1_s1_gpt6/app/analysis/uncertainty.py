import numpy as np,pandas as pd,json
from model import load,parameters,score,erlang
from mixture import mixture
D=load();P=parameters(D)
# Round tiny sample-specific replica-boundary gains at light load to full B.
S=np.array([1,1,1,1,1,1,1,1,1,1,.86,.69,.57,.50,.49,.51,.60,.76,1,1,1,1,1,1])
lam=D.requests.to_numpy()/3600;hour=D.hour.to_numpy()
base=score(lam,0,P)
full=score(lam,1,P)
sched=score(lam,S[hour],P)
print('central',np.average(full-base,weights=lam),np.average(sched-base,weights=lam))
# Measurement calibration check using utilization (a separate queue statistic).
a,w=erlang(lam,P['gen_A']+(P['gen_B']-P['gen_A'])*D.fraction_B.to_numpy(),D.replicas.to_numpy(),P['theta'])
us=lam*(P['gen_A']+(P['gen_B']-P['gen_A'])*D.fraction_B.to_numpy())*(1-a)/D.replicas
for f in [0,.3,1]:
 z=D.fraction_B==f
 print('util f',f,'actual-pred',np.mean(D.utilization[z]-us[z]),'sd',np.std(D.utilization[z]-us[z]))
# Day bootstrap resamples paired parameter observations and arrival patterns together.
rng=np.random.default_rng(8193)
days=np.sort(D.day.unique()); blocks={day:D[D.day==day] for day in days}
boot=[]
for i in range(600):
 d=pd.concat([blocks[day] for day in rng.choice(days,len(days),replace=True)],ignore_index=True)
 p=parameters(d); l=d.requests.to_numpy()/3600;hh=d.hour.to_numpy()
 b=score(l,0,p)
 boot.append([np.average(score(l,1,p)-b,weights=l),np.average(score(l,S[hh],p)-b,weights=l)])
boot=np.array(boot)
print('bootstrap quantiles',np.quantile(boot,[.005,.025,.5,.975,.995],axis=0),'std',boot.std(axis=0))
# Account for exact service-mixture behavior, using actual per-day arrival rates at mixed hours.
corrections=[]
for hh in range(10,18):
 sub=D[D.hour==hh];ls=sub.requests.to_numpy()/3600;f=S[hh]
 # evaluate five quantiles and interpolate a smooth correction (all 24 replicas here).
 sample=np.quantile(ls,[0,.25,.5,.75,1]); dd=[]
 for l in sample:
  service=P['gen_A']*(1-f)+P['gen_B']*f;c=int(np.clip(np.ceil(l*service/.75),4,24))
  aex,wex,it=mixture(l,f,P['gen_A'],P['gen_B'],c,P['theta'])
  aa,ww=erlang(np.array(l),np.array(service),np.array(c),P['theta'])
  rating=P['rating_A']*(1-f)+P['rating_B']*f
  dd.append(-rating*(aex-aa)-.0062*(wex-ww))
 corr=np.interp(ls,sample,dd)
 corrections.append((hh,float(np.average(corr,weights=ls)),float(sub.requests.sum()/D.requests.sum())))
 print('exact mixture correction',corrections[-1],flush=True)
correction=sum(x[1]*x[2] for x in corrections)
print('total exact correction',correction)
# Sensitivity to expected traffic and service parameters, using fixed launch schedule.
for key,mults in [('arrival',[.98,.99,1,1.01,1.02]),('theta',[.95,1,1.05]),('gen_B',[.995,1,1.005])]:
 for m in mults:
  p=P.copy();l=lam.copy()
  if key=='arrival':l*=m
  else:p[key]*=m
  b=score(l,0,p)
  print('sensitivity',key,m,np.average(score(l,1,p)-b,weights=l),np.average(score(l,S[hour],p)-b,weights=l))
np.save('/app/analysis/bootstrap.npy',boot)
with open('/app/analysis/results.json','w') as f:
 json.dump(dict(parameters=P,schedule=S.tolist(),Q1=float(np.average(full-base,weights=lam)),Q2=float(np.average(sched-base,weights=lam)),mixture_correction=correction,bootstrap_quantiles=np.quantile(boot,[.025,.975],axis=0).tolist()),f,indent=2)
