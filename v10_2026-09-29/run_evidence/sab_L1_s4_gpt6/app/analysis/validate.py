from optimize import *
import pandas as pd
# Round schedule for straightforward operations.
sched=[1.]*24
for path in (ROOT/'analysis').glob('refined_*.json'):
    for h,r in json.loads(path.read_text()).items():sched[int(h)]=round(r['fraction'],2)
base=value([0]*24);q1=value([1]*24)-base;q2=value(sched,True)-base
print('schedule',sched,'base',base,'Q1',q1,'Q2',q2,flush=True)
(ROOT/'rollout.conf').write_text(''.join(f'{h:02d} {f:.2f}\n' for h,f in enumerate(sched)))
# Parameter sensitivity; baseline is recomputed in every scenario.
def quick(lams,fs,sa_=sa,sb_=sb,theta=th,ratinga=ra,ratingb=rb):
    f=np.array(fs)[:,None];mean=sa_+(sb_-sa_)*f;c=np.clip(np.ceil(lams*mean/.75),4,16).astype(int)
    a,w=erlang_a(lams,mean,c,theta)
    s=(1-a)*(ratinga+(ratingb-ratinga)*f)-.008*w
    return (s*lams*weights).sum()/(lams*weights).sum()
sens=[]
for demand in [.99,1.,1.01]:
    l=rates*demand;b=quick(l,[0]*24)
    z={'demand_scale':demand,'Q1':quick(l,[1]*24)-b,'Q2_proxy':quick(l,sched)-b};sens.append(z);print(z,flush=True)
for delta in [-.015,.015]:
    b=quick(rates,[0]*24,sb_=sb+delta)
    z={'B_service_change_s':delta,'Q1':quick(rates,[1]*24,sb_=sb+delta)-b,'Q2_proxy':quick(rates,sched,sb_=sb+delta)-b};sens.append(z);print(z,flush=True)
# Validate pure-B congested drill; ratios identify patience independently of capacity approximation.
d=pd.read_csv(ROOT/'data/abtest_002.csv');preda=[];predw=[]
for r in d.itertuples():
    a,w=erlang_a((r.requests_A+r.requests_B)/3600,sb,r.replicas,th);preda.append(float(a));predw.append(float(w))
print('drill observed/predicted abandon',d.abandon_rate.mean(),np.mean(preda),'wait',d.mean_queue_wait_s.mean(),np.mean(predw),flush=True)
# Demand-model sensitivity: request-rate empirical distributions from the observed complete days.
t=pd.read_csv(ROOT/'analysis/traffic.csv');t=t[t.day<=33]
emp=[]
for fs in [[0]*24,[1]*24,sched]:
    req=t.requests.to_numpy();lam=req/3600;f=np.array(fs)[t.hour];mean=sa+(sb-sa)*f;c=np.clip(np.ceil(lam*mean/.75),4,16).astype(int);a,w=erlang_a(lam,mean,c,th)
    emp.append(float(np.average((1-a)*(ra+(rb-ra)*f)-.008*w,weights=req)))
print('empirical Q1/Q2proxy',emp[1]-emp[0],emp[2]-emp[0],flush=True)
result={'schedule':sched,'base_score':base,'Q1':q1,'Q2':q2,'overall_B_fraction':float(np.average(sched,weights=rates@weights)),'sensitivity':sens,'empirical_Q1':emp[1]-emp[0],'empirical_Q2_proxy':emp[2]-emp[0]}
(ROOT/'analysis/result.json').write_text(json.dumps(result,indent=2)+'\n')
