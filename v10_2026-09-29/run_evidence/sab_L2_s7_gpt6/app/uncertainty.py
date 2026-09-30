exec(open('/app/analyze.py').read().split('# resample arrivals')[0])
T=pd.read_csv('/app/data/inferred.csv')
schedule=np.array([float(s.split()[1]) for s in open('/app/rollout.conf')])
# adopt full B where sampled autoscaler steps suggest immaterial tiny reductions
schedule[schedule>.98]=1.
# Use observed request rates for main fit; utilization inversion is a sensitivity check.
weights=T.groupby('dow').day.transform('nunique').to_numpy()
lam=T.requests.to_numpy()/3600; hh=T.hour.to_numpy();fs=schedule[hh];peak=(hh>=9)&(hh<=17)
def estimates(lam,Ag=Ag,Bg=None,theta=th,ratings=None):
 if Bg is None:Bg=np.where(peak,params[1][1],params[0][1])
 if ratings is None:ratings=(A,np.where(peak,params[1][0],params[0][0]))
 scores=[]
 for f in [np.zeros(len(T)),np.ones(len(T)),fs]:
  m=Ag+(Bg-Ag)*f;c=np.clip(np.ceil(lam*m/.75),4,16)
  wait=queue(lam,m,c,theta)
  r=ratings[0]+(ratings[1]-ratings[0])*f
  scores.append(r*(1-theta*wait)-.008*wait)
 return np.array(scores)
scores=estimates(lam)
mean=(scores*lam/weights).sum(axis=1)/(lam/weights).sum()
print('Empirical weekdays equally weighted:',mean, 'delta',mean[1:]-mean[0])
for name,l in [('utilization-inferred',T.inferred_rate.to_numpy()),('arrival +1%',lam*1.01),('arrival -1%',lam*.99)]:
 sc=estimates(l); avg=(sc*l/weights).sum(axis=1)/(l/weights).sum();print(name,avg[1:]-avg[0])
# resampling calendar days independently within weekday strata, same draw for all hours
rng=np.random.default_rng(188); nd=int(T.day.max()); nboot=20000
sums=np.zeros((nd,4))
for d in range(1,nd+1):
 ix=T.day.to_numpy()==d;sums[d-1,:3]=(scores[:,ix]*lam[ix]).sum(axis=1);sums[d-1,3]=lam[ix].sum()
boot=np.zeros((nboot,4))
for dow in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']:
 days=T.loc[T.dow==dow,'day'].unique()-1
 chosen=rng.choice(days,(nboot,len(days)))
 boot+=sums[chosen].mean(axis=1)
vals=boot[:,:3]/boot[:,3,None];delta=vals[:,1:]-vals[:,0,None]
# conservative independent Bernoulli variance for the noisy served ratings
wa=H.requests.mul(1-H.abandon_rate).sum()+E.requests_A.mul(1-E.abandon_rate).sum()
wb=np.array([E.loc[E.hour.between(9,17)==p,'requests_B'].mul(1-E.loc[E.hour.between(9,17)==p,'abandon_rate']).sum() for p in [False,True]])
print('rating SE A/B:',np.sqrt(A*(1-A)/wa),[np.sqrt(params[i][0]*(1-params[i][0])/wb[i]) for i in [0,1]])
print('day bootstrap 5,50,95:',np.quantile(delta,[.05,.5,.95],axis=0))
print('day bootstrap sd:',np.std(delta,axis=0))
# selected fractions vs empirical optimal: fine grid, equal weekday weights
regret=0.; total=(lam/weights).sum()
for h in range(24):
 ix=hh==h;l=lam[ix];wg=1/weights[ix];b,bg=params[int(9<=h<=17)]
 f=np.linspace(0,1,501)[:,None];m=Ag+(bg-Ag)*f;c=np.clip(np.ceil(l[None,:]*m/.75),4,16);w=queue(l[None,:],m,c);r=A+(b-A)*f
 v=((r*(1-th*w)-.008*w)*l*wg).sum(axis=1)
 actual=(scores[2,ix]*l*wg).sum();regret+=v.max()-actual
print('empirical optimization regret:',regret/total)
with open('/app/rollout.conf','w') as out:
 for h,f in enumerate(schedule):out.write(f'{h:02d} {f:.3f}\n')
json.dump({'Q1':float(mean[1]-mean[0]),'Q2':float(mean[2]-mean[0]),'bootstrap_sd':np.std(delta,axis=0).tolist(),'regret':regret/total},open('/app/final_estimates.json','w'),indent=2)
