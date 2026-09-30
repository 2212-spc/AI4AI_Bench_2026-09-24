import pandas as pd,numpy as np
H=pd.read_csv('/app/data/history.csv'); E=pd.concat([pd.read_csv('/app/data/abtest_000.csv'),pd.read_csv('/app/data/abtest_001.csv')]);E['requests']=E.requests_A+E.requests_B
th=.017 # likely exact
# hour params
ra=[];rb=[]
for h in range(24):
 x=H[H.hour==h]; y=E[E.hour==h]
 # means weighted by served approximate
 wa=x.requests*(1-x.abandon_rate); wb=y.requests_A*(1-y.abandon_rate)
 ra.append(np.sum(x.mean_rating*wa)+np.sum(y.rating_A.fillna(0)*wb))
 den=wa.sum()+wb.sum();ra[-1]/=den
 wb2=y.requests_B*(1-y.abandon_rate)
 rb.append(np.sum(y.rating_B*wb2)/wb2.sum())
print('ra',np.round(ra,6));print('rb',np.round(rb,6));
ga=3.2; gb=np.array([6 if h<=8 or h>=18 else 8 for h in range(24)])
# use all traffic observations. baseline model rows each H
base=[]; rows=[]
for h in range(24):
 x=pd.concat([H[H.hour==h][['requests','replicas']],E[E.hour==h][['requests','replicas']]],ignore_index=True)
 lam=x.requests.to_numpy()/3600; basec=x.replicas.to_numpy()
 # use actual all A baseline c based lambda*3.2/.75; calculate model
 c=np.clip(np.ceil(lam*ga/.75),4,16)
 # queue vector each row
 def qwait(m,c):
  z=np.ones(len(lam));tot=z.copy();q=np.zeros(len(lam))
  for n in range(1,100):
   z*=lam/(np.minimum(n,c)/m+np.maximum(n-c,0)*th);tot+=z;q+=np.maximum(n-c,0)*z
  return q/tot/lam
 wb=qwait(ga,c); sb=ra[h]*(1-th*wb)-.008*wb
 base.append(np.sum(lam*sb))
 rows.append(lam)
# aggregate grid loop
fs=np.linspace(0,1,1001); results=[]; opts=[]
for h in range(24):
 lam=rows[h]; w=np.zeros_like(fs); # score total weighted by row then / overall later
 for i,f in enumerate(fs):
  m=ga+(gb[h]-ga)*f; c=np.clip(np.ceil(lam*m/.75),4,16)
  z=np.ones(len(lam));tot=z.copy();q=np.zeros(len(lam))
  for n in range(1,100):
   z*=lam/(np.minimum(n,c)/m+np.maximum(n-c,0)*th);tot+=z;q+=np.maximum(n-c,0)*z
  wait=q/tot/lam
  score=(ra[h]*(1-f)+rb[h]*f)*(1-th*wait)-.008*wait
  w[i]=np.sum(lam*score)
 i=np.argmax(w);opts.append(fs[i]);results.append(w)
 print(h,fs[i],w[i]/np.sum(lam), 'gain', (w[i]-w[0])/np.sum(lam), 'full', (w[-1]-w[0])/np.sum(lam))
# all traffic aggregate weighting by each hour total mean lambda
hoursum=np.array([r.sum() for r in rows]); b=sum(base)/sum(hoursum)
q1=sum(results[h][-1] for h in range(24))/sum(hoursum)-b
q2=sum(results[h][int(round(opts[h]*1000))] for h in range(24))/sum(hoursum)-b
print('total base',b,'q1',q1,'q2',q2)
with open('/app/rollout.conf','w') as f:
 for h,v in enumerate(opts):f.write(f'{h:02d} {v:.3f}\n')
