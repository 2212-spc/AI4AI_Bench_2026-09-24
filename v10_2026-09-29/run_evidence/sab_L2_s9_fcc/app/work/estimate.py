import csv,glob,numpy as np
from model import PROF, M as HM
rows=[]
for fn in sorted(glob.glob('/app/data/abtest_*.csv')):
    for r in csv.DictReader(open(fn)):
        if 'max_replicas' in fn: pass
        rows.append((fn,r))
# per-hour B stats (exclude drills 002/003 for gen? they are fine for rating/gen; include)
nB=np.zeros(24); sB=np.zeros(24); gBs=np.zeros(24); nA=np.zeros(24); sA=np.zeros(24); gAs=np.zeros(24)
for fn,r in rows:
    h=int(r['hour']); b=float(r['requests_B']); a=float(r['requests_A'])
    ab=float(r['abandon_rate'])
    if b>0: nB[h]+=b*(1-ab); sB[h]+=b*(1-ab)*float(r['rating_B']); gBs[h]+=b*(1-ab)*float(r['gen_time_B_s'])
    if a>0: nA[h]+=a*(1-ab); sA[h]+=a*(1-ab)*float(r['rating_A']); gAs[h]+=a*(1-ab)*float(r['gen_time_A_s'])
rB=sB/nB; gB=gBs/nB; rAx=sA/np.maximum(nA,1); gA=gAs/np.maximum(nA,1)
SIG=0.23
for h in range(24):
    print(f'{h:02d} nB {nB[h]:7.0f} rB {rB[h]:.4f}±{SIG/np.sqrt(nB[h]):.4f} gB {gB[h]:.3f}±{gB[h]/np.sqrt(nB[h]):.3f}  nA {nA[h]:7.0f} rA_exp {rAx[h]:.4f} gA {gA[h]:.3f}')
peak=[h for h in range(24) if 9<=h<=17]; off=[h for h in range(24) if h not in peak]
print('pooled peak rB',sB[peak].sum()/nB[peak].sum(),'gB',gBs[peak].sum()/nB[peak].sum(),'n',nB[peak].sum())
print('pooled off  rB',sB[off].sum()/nB[off].sum(),'gB',gBs[off].sum()/nB[off].sum(),'n',nB[off].sum())
# chi2 test for constant within blocks
for blk in [peak,off]:
    z=(rB[blk]-sB[blk].sum()/nB[blk].sum())/(SIG/np.sqrt(nB[blk])); print('chi2',(z**2).sum(),'df',len(blk)-1)
# day multipliers from experiments (full days only)
days={}
for fn,r in rows:
    d=int(r['day']); h=int(r['hour']); tot=float(r['requests_A'])+float(r['requests_B'])
    days.setdefault(d,{})[h]=tot/PROF[h]
for d in sorted(days):
    v=np.array(list(days[d].values())); print(d, len(v), v.mean().round(3), v.std().round(3), rows[[i for i,(f,r) in enumerate(rows) if int(r['day'])==d][0]][1]['dow'])
np.save('params.npy',dict(rB=rB,gB=gB,nB=nB,rAx=rAx,nA=nA,daymult_exp={d:np.mean(list(v.values())) for d,v in days.items()}))
