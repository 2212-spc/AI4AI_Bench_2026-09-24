import json,collections,random
rows={str(x['id']):x for x in json.load(open('dev.json'))['prompts']}
g=collections.defaultdict(lambda:[0,0])
for i,y in json.load(open('labels1.json'))['labels'].items():
 x=rows[i];k=(x['topic'],x['length'],x['judge_win']);g[k][0]+=1;g[k][1]+=y!=x['judge_win']
pool=collections.defaultdict(list)
for x in json.load(open('dev2.json'))['prompts']:pool[(x['topic'],x['length'],x['judge_win'])].append(x['id'])
m={k:2 for k in pool}
for _ in range(80-sum(m.values())):
 def score(k):
  n,e=g[k];N=len(pool[k]);nt=N+len(pool[(k[0],k[1],not k[2])]);w=N/nt
  a=e+.35;b=n-e+10;t=a+b
  return w*w*a*b/t/(t+1)/(t+m[k])/(t+m[k]+1) if m[k]<N else -1
 k=max(sorted(pool),key=score);m[k]+=1
rng=random.Random(455);ids=[]
for k in sorted(pool):
 print(k,m[k]);ids+=rng.sample(pool[k],m[k])
json.dump({'arena':'dev2','ids':ids},open('labels_args2.json','w'))
