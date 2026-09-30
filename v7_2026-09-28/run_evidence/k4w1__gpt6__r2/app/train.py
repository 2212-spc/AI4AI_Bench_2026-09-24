import json,numpy as np,collections
from model import *
qs=sum([json.load(open('/app/'+t+f+'.json'))['questions'] for t in ['algebra','geometry','combinatorics','number_theory'] for f in (['','2'] if t in ('algebra','geometry') else [''])],[])
X=[]
for q in qs:
 obs=[(s['variant'],s['answer'],s['score']) for s in q['samples']]
 X.append(likelihoods(groups(obs,q['topic']),q['correct_answer']))
X=np.array(X);X/=X.sum(axis=1,keepdims=True)
w=np.ones(len(COMPS))/len(COMPS)
for it in range(300):
 r=X*w;r/=r.sum(axis=1,keepdims=True)
 w=(r.sum(axis=0)+.2)/(len(qs)+len(COMPS)*.2)
# prompt exchangeability
for start in [17,22,27]:w[start:start+5]=w[start:start+5].mean()
print([(mode,rate[0],round(x,4)) for (mode,rate),x in zip(COMPS,w)])
json.dump(w.tolist(),open('/app/weights.json','w'))
