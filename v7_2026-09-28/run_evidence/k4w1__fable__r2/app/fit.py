import json, glob, math
import numpy as np
bins=[0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
edges=[-9]+bins+[9]
for t in ['algebra','geometry','combinatorics','number_theory']:
    qs=[q for f in glob.glob('data/dev*_%s.json'%t) for q in json.load(open(f))['questions']]
    sc=[s['score'] for q in qs for s in q['samples'] if s['correct']]
    sw=[s['score'] for q in qs for s in q['samples'] if not s['correct']]
    c=np.histogram(sc,edges)[0]; w=np.histogram(sw,edges)[0]
    prior=math.log(len(sc)/len(sw))
    llr=[round(math.log((ci+1)/(wi+1))-prior,2) for ci,wi in zip(c,w)]
    print(f'    "{t}": {llr},   # c={list(c)} w={list(w)}')
