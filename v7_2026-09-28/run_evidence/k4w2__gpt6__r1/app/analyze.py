import json, glob, collections, numpy as np
qs=sum([json.load(open(f))['questions'] for f in glob.glob('/app/dev*.json')],[])
for t in sorted(set(q['topic'] for q in qs)):
 print('\n',t)
 for v in range(5):
  ss=[s for q in qs if q['topic']==t for s in q['samples'] if s['variant']==v]
  print(v,len(ss),'accuracy',round(np.mean([s['correct'] for s in ss]),3), 'scores',[(c,round(np.mean([s['score'] for s in ss if s['correct']==c]),3),round(np.std([s['score'] for s in ss if s['correct']==c]),3)) for c in [True,False]])
 for lo,hi in zip([-2,-.2,.2,.5,.8,1.1,1.4,1.7,2,2.4],[-.2,.2,.5,.8,1.1,1.4,1.7,2,2.4,4]):
  ss=[s for q in qs if q['topic']==t for s in q['samples'] if lo<=s['score']<hi]
  print((lo,hi),len(ss),round(np.mean([s['correct'] for s in ss]),3) if ss else None)
