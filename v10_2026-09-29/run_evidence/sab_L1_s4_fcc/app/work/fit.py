import csv, numpy as np, glob
from model import erlang_a, replicas
rows=[]
for fn in sorted(glob.glob('/app/data/abtest_*.csv')):
    rows+=list(csv.DictReader(open(fn)))
hist=list(csv.DictReader(open('/app/data/history.csv')))
def wmean(vals,ws):
    vals=np.array(vals);ws=np.array(ws); return (vals*ws).sum()/ws.sum()
# B parameters: rating and gen time weighted by served requests
sB=[float(r['requests_B'])*(1-float(r['abandon_rate'])) for r in rows if r['requests_B']!='0']
rB=wmean([float(r['rating_B']) for r in rows if r['requests_B']!='0'],sB)
gB=wmean([float(r['gen_time_B_s']) for r in rows if r['requests_B']!='0'],sB)
# rating variance: per-request SD estimate from per-hour spread
resid=[(float(r['rating_B'])-rB)*np.sqrt(w) for r,w in zip([r for r in rows if r['requests_B']!='0'],sB)]
sdB=np.std(resid); print('per-request rating SD (B):',sdB, 'nB served',sum(sB),'se rB',sdB/np.sqrt(sum(sB)))
sA=[float(r['requests_A'])*(1-float(r['abandon_rate'])) for r in rows if r['requests_A']!='0']
rA_exp=wmean([float(r['rating_A']) for r in rows if r['requests_A']!='0'],sA)
sAh=[float(r['requests'])*(1-float(r['abandon_rate'])) for r in hist]
rA_h=wmean([float(r['mean_rating']) for r in hist],sAh); gA_h=wmean([float(r['mean_gen_time_s']) for r in hist],sAh)
residA=[(float(r['mean_rating'])-rA_h)*np.sqrt(w) for r,w in zip(hist,sAh)]
print('per-request rating SD (A):',np.std(residA))
nA=sum(sA)+sum(sAh)
rA=(rA_exp*sum(sA)+rA_h*sum(sAh))/nA
gA=(wmean([float(r['gen_time_A_s']) for r in rows if r['requests_A']!='0'],sA)*sum(sA)+gA_h*sum(sAh))/nA
print('rB',rB,'gB',gB,'rA',rA,'gA',gA,'rA_exp',rA_exp,'rA_hist',rA_h,'nA',nA,'se rA',np.std(residA)/np.sqrt(nA))
# gen-time SE: exponential so SD=mean
print('se gB',gB/np.sqrt(sum(sB)),'se gA',gA/np.sqrt(nA))
# theta: ratio estimator pooled
allr=rows+hist
th=np.array([float(r['abandon_rate'])/float(r['mean_queue_wait_s']) for r in allr])
print('theta',th.mean(),'se',th.std()/np.sqrt(len(th)),'n',len(th))
