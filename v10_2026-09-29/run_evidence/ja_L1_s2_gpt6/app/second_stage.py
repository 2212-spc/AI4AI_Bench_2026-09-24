import csv,json,numpy as np,subprocess
from pathlib import Path
P=Path('/app'); p=list(csv.DictReader(open(P/'data/pool.csv'))); d=json.load(open(P/'data/audit_design.json'))
rng=np.random.default_rng(271828)
stage={}
for m in 'AB':
 groups={}
 if m=='A':
  for (s,g),n in zip(d[m].items(),[8,2,5,5,5,5]):
   groups[s]={'population':sorted(set(g['population'])-set(g['pilot'])),'n':n,'source':s}
 else:
  for s,n in [('00',4),('10',2),('11_len0',5),('11_len1',5)]:
   g=d[m][s];groups[s]={'population':sorted(set(g['population'])-set(g['pilot'])),'n':n,'source':s}
  for name,s,lo,hi,n in [('01_short','01',0,900,3),('01_long','01',900,10000,4),('11_900_1200','11_len2',0,1200,7),('11_1200_1500','11_len2',1200,1500,22),('11_1500_plus','11_len2',1500,10000,28)]:
   g=d[m][s]; ids=[i for i in g['population'] if i not in g['pilot'] and lo<=int(p[i][f'len_{m}'])<hi]
   groups[name]={'population':ids,'n':n,'source':s}
 chosen=[]
 for s,g in groups.items():
  g['sample']=list(map(int,rng.choice(g['population'],g['n'],replace=False)))
  chosen.extend(g['sample'])
 assert len(chosen)==(30 if m=='A' else 80)
 stage[m]=groups
 print(m,{s:(len(g['population']),g['n']) for s,g in groups.items()},flush=True)
 result=subprocess.run(['lab','verify',f'model={m}','items='+','.join(map(str,chosen))],capture_output=True,text=True,check=True)
 print(result.stdout)
(P/'data/second_stage_design.json').write_text(json.dumps(stage,indent=2))
