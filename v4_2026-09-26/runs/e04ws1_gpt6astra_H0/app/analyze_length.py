import json,numpy as np
r=json.load(open('/app/length_grid_results.json'))['results']
r=[v for v in r if v['config']['n']>1]
nvals=[2,4,13,41,256]
x=np.array([np.sqrt(v['kl'])/(1+v['config']['len_pen']) for v in r]); y=np.array([v['proxy_reward'] for v in r])
X=np.array([[int(v['config']['n']==n) for n in nvals] for v in r])
for name,fn in [('tanh',np.tanh),('exp',lambda t:1-np.exp(-t)),('rational',lambda t:t/(1+t)),('log',np.log1p),('clip',lambda t:np.minimum(t,1)),('gaussian',lambda t:1-np.exp(-t*t))]:
 best=None
 for k in np.geomspace(.01,100,4000):
  A=np.column_stack([X,fn(k*x)])
  coef=np.linalg.lstsq(A,y,rcond=None)[0]
  err=np.mean((A@coef-y)**2)
  if best is None or err<best[0]:best=(err,k,coef)
 print(name,'RMSE',np.sqrt(best[0]),'k',best[1],'coef',best[2], 'fraction',best[2][-1]*fn(best[1]*np.sqrt(2.737962))/(best[2][3]+best[2][-1]*fn(best[1]*np.sqrt(2.737962))))
