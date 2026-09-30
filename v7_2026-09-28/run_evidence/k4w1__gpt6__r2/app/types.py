import json, numpy as np,collections
qs=sum([json.load(open('/app/'+t+f+'.json'))['questions'] for t in ['algebra','geometry'] for f in ['', '2']],[])
by=collections.defaultdict(list)
for q in qs:
 s=q['samples'];wrong=[x for x in s if not x['correct']];high=[x for x in wrong if x['score']>.6];cnt=collections.Counter(x['answer'] for x in high)
 if not high: typ='noise'
 elif len(high)<3:typ='uncertain'
 elif max(cnt.values())>=len(high)*.7:typ='shared'
 else:typ='local'
 cc=np.array([[sum(x['correct'] for x in s if x['variant']==v),sum(not x['correct'] and x['score']>.6 for x in s if x['variant']==v)] for v in range(5)])
 by[typ].append(cc)
for typ,cc in by.items():
 a=np.array(cc);print(typ,len(a),'mean',a.mean(axis=(0,1))/4,'question C', np.sort(a[:,:,0].sum(axis=1)), 'variant C histogram',np.bincount(a[:,:,0].ravel(),minlength=5),'variant trap histogram',np.bincount(a[:,:,1].ravel(),minlength=5))
 if typ=='shared':print(a[:,:,0])
