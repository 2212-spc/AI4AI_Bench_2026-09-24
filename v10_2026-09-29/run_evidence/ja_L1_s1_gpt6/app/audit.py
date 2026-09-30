import csv, json, numpy as np, subprocess
from pathlib import Path
ROOT=Path('/app')
rows=list(csv.DictReader(open(ROOT/'data/pool.csv')))
reviews={m:{int(r['item']):int(r['review_verdict']) for r in csv.DictReader(open(ROOT/f'data/review_{m}_{i:03d}.csv'))} for m,i in [('A',0),('B',1)]}
def make_strata():
 strata={}
 for m in ['A','B']:
  for j in [0,1]:
   for v in [0,1]:
    ids=[int(r['item']) for r in rows if int(r['judge_'+m])==j and reviews[m][int(r['item'])]==v]
    ids.sort(key=lambda i:(int(rows[i]['len_'+m]),i))
    groups=[ids] if m=='A' else [ids[:len(ids)//2],ids[len(ids)//2:]]
    for k,g in enumerate(groups):strata[f'{m}_{j}{v}_{k}']=g
 return strata
if __name__=='__main__':
 strata=make_strata(); rng=np.random.default_rng(92741)
 counts={'A_00_0':5,'A_01_0':3,'A_10_0':3,'A_11_0':13,'B_00_0':4,'B_00_1':4,'B_01_0':2,'B_01_1':2,'B_10_0':4,'B_10_1':4,'B_11_0':10,'B_11_1':10}
 selected={h:sorted(map(int,rng.choice(ids,counts[h],replace=False))) for h,ids in strata.items()}
 json.dump({'strata':strata,'pilot':selected,'seed':92741},open(ROOT/'data/design.json','w'),indent=2)
 for m in ['A','B']:
  ids=sorted(i for h,ii in selected.items() if h.startswith(m) for i in ii)
  p=subprocess.run(['lab','verify',f'model={m}','items='+','.join(map(str,ids))],capture_output=True,text=True,check=True)
  print(p.stdout)
