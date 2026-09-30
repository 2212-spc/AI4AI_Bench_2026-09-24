import numpy as np,pandas as pd,json
H=pd.read_csv('/app/data/history.csv'); E=pd.concat([pd.read_csv('/app/data/abtest_000.csv'),pd.read_csv('/app/data/abtest_001.csv')],ignore_index=True)
E['requests']=E.requests_A+E.requests_B
D=pd.concat([H[['day','dow','hour','requests']],E[['day','dow','hour','requests']]])
th=(pd.concat([H,E]).eval('abandon_rate*requests').sum()/pd.concat([H,E]).eval('mean_queue_wait_s*requests').sum())
def queue(lam,m,c,theta=th):
 lam,m,c=np.broadcast_arrays(lam,m,c); z=np.ones(lam.shape); total=z.copy(); q=np.zeros(lam.shape)
 for n in range(1,140):
  z=z*lam/(np.minimum(n,c)/m+np.maximum(n-c,0)*theta)
  total+=z; q+=np.maximum(n-c,0)*z
 return q/total/lam
# pooled estimates within empirically evident service/rating regimes
A=(np.sum(H.mean_rating*H.requests*(1-H.abandon_rate))+np.sum(E.rating_A.fillna(0)*E.requests_A*(1-E.abandon_rate)))/(np.sum(H.requests*(1-H.abandon_rate))+np.sum(E.requests_A*(1-E.abandon_rate)))
Ag=(np.sum(H.mean_gen_time_s*H.requests)+np.sum(E.gen_time_A_s.fillna(0)*E.requests_A))/(H.requests.sum()+E.requests_A.sum())
params=[]
for peak in [False,True]:
 e=E[E.hour.between(9,17)==peak]; w=e.requests_B*(1-e.abandon_rate)
 params.append((np.average(e.rating_B,weights=w),np.average(e.gen_time_B_s,weights=w)))
print('params',th,A,Ag,params)
# validation known replica and observed arrival count
for data in [H,E]:
 if 'fraction_B' in data:
  f=data.fraction_B.to_numpy(); b=np.where(data.hour.between(9,17),params[1][1],params[0][1]); m=Ag*(1-f)+b*f
 else: m=Ag
 pred=queue(data.requests.to_numpy()/3600,m,data.replicas.to_numpy())
 print('validation mean predicted,observed wait',np.average(pred,weights=data.requests),np.average(data.mean_queue_wait_s,weights=data.requests),'logratio',np.mean(np.log(data.mean_queue_wait_s/pred)))
# resample arrivals with equal weekday weight; slight smoothing to remove sample-induced steps
rng=np.random.default_rng(893)
fractions=np.linspace(0,1,501); output=[]; allbase=[]; allfull=[]; allopt=[]; weights=[]
for h in range(24):
 d=D[D.hour==h]; rates=[]
 for dow in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']:
  r=d[d.dow==dow].requests.to_numpy()/3600
  rates.extend(rng.choice(r,200)*np.exp(rng.normal(-.5*.015**2,.015,200)))
 lam=np.array(rates); B,Bg=params[int(9<=h<=17)]
 m=Ag+(Bg-Ag)*fractions[:,None]; l=lam[None,:]; c=np.clip(np.ceil(l*m/.75),4,16)
 wait=queue(l,m,c); rating=A+(B-A)*fractions[:,None]
 val=np.sum(l*(rating*(1-th*wait)-.008*wait),axis=1)/np.sum(l)
 i=np.argmax(val); output.append(float(fractions[i])); weights.append(lam.mean()); allbase.append(val[0]);allfull.append(val[-1]);allopt.append(val[i])
 print(h,round(lam.mean(),4),round(fractions[i],3),round(val[i]-val[0],6),round(val[-1]-val[0],6))
w=np.array(weights); base=np.average(allbase,weights=w); full=np.average(allfull,weights=w); opt=np.average(allopt,weights=w)
print('baseline',base,'Q1',full-base,'Q2',opt-base)
with open('/app/rollout.conf','w') as f:
 for h,v in enumerate(output):f.write(f'{h:02d} {v:.3f}\n')
json.dump(dict(theta=th,A=A,Ag=Ag,B=params,schedule=output,baseline=base,Q1=full-base,Q2=opt-base),open('/app/estimates.json','w'),indent=2)
