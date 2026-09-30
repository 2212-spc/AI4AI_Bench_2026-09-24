import numpy as np
from model import load,parameters,score
D=load();p0=parameters(D)
S=np.array([1,1,1,1,1,1,1,1,1,1,.86,.69,.57,.50,.49,.51,.60,.76,1,1,1,1,1,1])
grid=np.linspace(0,1,201)
for scale in [.98,1,1.02]:
 for rdelta in [-.001,0,.001]:
  p=p0.copy();p['rating_B']+=rdelta
  regret=0
  for h in range(24):
   d=D[D.hour==h];l=d.requests.to_numpy()/3600*scale
   curve=np.average(score(l[:,None],grid[None,:],p),weights=l,axis=0)
   actual=np.average(score(l,S[h],p),weights=l)
   regret+=(curve.max()-actual)*d.requests.sum()/D.requests.sum()
  print('arrival scale',scale,'B rating change',rdelta,'regret',regret)
