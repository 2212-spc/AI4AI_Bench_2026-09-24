import csv, numpy as np, collections, glob
from erlanga import erlang_a, replicas
ETA=0.0076
hist=list(csv.DictReader(open('/app/data/history.csv')))
exp=[]
for f in sorted(glob.glob('/app/data/abtest_*.csv')): exp+=list(csv.DictReader(open(f)))
# per-hour A rating from history + experiment A arm
hA=collections.defaultdict(lambda:[0,0]); hB=collections.defaultdict(lambda:[0,0]); gB=collections.defaultdict(lambda:[0,0])
for r in hist:
    n=float(r['requests'])*(1-float(r['abandon_rate'])); hA[int(r['hour'])][0]+=float(r['mean_rating'])*n; hA[int(r['hour'])][1]+=n
for r in exp:
    h=int(r['hour'])
    if r['rating_A']:
        n=float(r['requests_A'])*(1-float(r['abandon_rate'])); hA[h][0]+=float(r['rating_A'])*n; hA[h][1]+=n
    n=float(r['requests_B'])*(1-float(r['abandon_rate'])); hB[h][0]+=float(r['rating_B'])*n; hB[h][1]+=n
    gB[h][0]+=float(r['gen_time_B_s'])*n; gB[h][1]+=n
W={h:sum(float(r['requests']) for r in hist if int(r['hour'])==h) for h in range(24)}
rA={h:hA[h][0]/hA[h][1] for h in range(24)}; rB={h:hB[h][0]/hB[h][1] for h in range(24)}
print('hour  rA      rB      diff    nB')
for h in range(24): print(f'{h:02d} {rA[h]:.4f} {rB[h]:.4f} {rB[h]-rA[h]:+.4f} {hB[h][1]:.0f}')
totW=sum(W.values())
print('traffic-weighted diff', sum((rB[h]-rA[h])*W[h] for h in range(24))/totW)
print('pooled rA', sum(hA[h][0] for h in range(24))/sum(hA[h][1] for h in range(24)), 'pooled rB', sum(hB[h][0] for h in range(24))/sum(hB[h][1] for h in range(24)))
print('pooled gen B', sum(gB[h][0] for h in range(24))/sum(gB[h][1] for h in range(24)))
# rating vs daily variation: per-day B rating
byd=collections.defaultdict(lambda:[0,0])
for r in exp:
    n=float(r['requests_B']); byd[int(r['day'])][0]+=float(r['rating_B'])*n; byd[int(r['day'])][1]+=n
print('daily B rating', [round(v[0]/v[1],4) for v in byd.values()])
# B rating by fraction (night 1.0 vs day 0.3)
for f in ['1.0','0.3']:
    rs=[r for r in exp if r['fraction_B']==f]; n=np.array([float(r['requests_B']) for r in rs]); rb=np.array([float(r['rating_B']) for r in rs])
    print('fB',f,'rating B',np.average(rb,weights=n))
# per-request sd of B ratings
rb=np.array([float(r['rating_B']) for r in exp]); nb=np.array([float(r['requests_B']) for r in exp]); hh=np.array([int(r['hour']) for r in exp])
res=rb-np.array([rB[h] for h in hh]); print('B per-request sd', np.sqrt(np.mean(res**2*nb)))
