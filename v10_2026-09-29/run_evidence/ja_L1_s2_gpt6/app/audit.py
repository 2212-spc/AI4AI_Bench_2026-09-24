import csv,json,numpy as np,subprocess
from pathlib import Path
P=Path('/app')
pool=list(csv.DictReader(open(P/'data/pool.csv')))
rng=np.random.default_rng(20260929)
design={}
for m,call,alloc in [('A',0,[6,2,2,6,7,7]),('B',1,[8,4,4,10,11,13])]:
 rev={int(r['item']):int(r['review_verdict']) for r in csv.DictReader(open(P/f'data/review_{m}_{call:03}.csv'))}
 groups={f'{j}{r}':[int(x['item']) for x in pool if int(x[f'judge_{m}'])==j and rev[int(x['item'])]==r] for j,r in [(0,0),(0,1),(1,0),(1,1)]}
 agreed=sorted(groups.pop('11'),key=lambda i:(int(pool[i][f'len_{m}']),i))
 for k,ids in enumerate(np.array_split(agreed,3)): groups[f'11_len{k}']=list(map(int,ids))
 design[m]={}
 chosen=[]
 for (s,ids),n in zip(groups.items(),alloc):
  sample=list(map(int,rng.choice(ids,n,replace=False)))
  design[m][s]={'population':ids,'pilot':sample}
  chosen.extend(sample)
 print(m,{s:(len(x['population']),len(x['pilot'])) for s,x in design[m].items()},flush=True)
 result=subprocess.run(['lab','verify',f'model={m}','items='+','.join(map(str,chosen))],capture_output=True,text=True,check=True)
 print(result.stdout)
(P/'data/audit_design.json').write_text(json.dumps(design,indent=2))
