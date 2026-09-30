import csv, glob, numpy as np, json
H=list(csv.DictReader(open('/app/data/history.csv')))
E=[r for p in glob.glob('/app/data/abtest_*.csv') for r in csv.DictReader(open(p))]
D=[]
for r in H:
 D.append(dict(day=int(r['day']),hour=int(r['hour']),dow=r['dow'],n=int(r['requests']),f=0,c=int(r['replicas']),ab=float(r['abandon_rate']),w=float(r['mean_queue_wait_s']),ra=float(r['mean_rating']),rb=np.nan,sa=float(r['mean_score']),sb=np.nan,ga=float(r['mean_gen_time_s']),gb=np.nan,na=int(r['requests']),nb=0))
for r in E:
 d={k:float(r[v]) if r[v] else np.nan for k,v in dict(f='fraction_B',c='replicas',ab='abandon_rate',w='mean_queue_wait_s',ra='rating_A',rb='rating_B',sa='score_A',sb='score_B',ga='gen_time_A_s',gb='gen_time_B_s',na='requests_A',nb='requests_B').items()}
 d.update(day=int(r['day']),hour=int(r['hour']),dow=r['dow'],n=d['na']+d['nb']);D.append(d)
x={k:np.array([r[k] for r in D]) for k in D[0]}
for arm in ['a','b']:
 for key in ['r','g']:
  w=x['n'+arm]*(1-x['ab']);v=x[key+arm];sel=np.isfinite(v)
  print(key+arm, np.average(v[sel],weights=w[sel]))
print('theta pooled',x['ab'].sum()/x['w'].sum(), 'ratio median',np.median(x['ab']/x['w']))
for h in range(24):
 sel=x['hour']==h;v=x['n'][sel];print(h,round(v.mean(),2),round(v.std(),2),round(v.min()),round(v.max()))
np.savez('/app/data/analysis_arrays.npz',**x)

def queue(lam,svc,c,th):
 lam,svc,c,th=np.broadcast_arrays(lam,svc,c,th)
 p=np.ones(lam.shape);z=p.copy();q=np.zeros(lam.shape)
 for n in range(1,250):
  p=p*lam/(np.minimum(n,c)/svc+np.maximum(n-c,0)*th)
  z+=p;q+=p*np.maximum(n-c,0)
 return q/z/lam
if __name__=='__main__':
 svc=1.698*(1-x['f'])+3.92*x['f']
 lam=x['n']/3600
 for th in [.015,.015625,.016,.017]:
  pred=queue(lam,svc,x['c'],th)
  for f in [0,.3,1]:
   s=x['f']==f
   print('fit',th,f,'wait',x['w'][s].mean(),pred[s].mean(),'ab',x['ab'][s].mean(),(th*pred)[s].mean(),'rms relative',np.sqrt(np.mean((x['w'][s]/pred[s]-1)**2)))
