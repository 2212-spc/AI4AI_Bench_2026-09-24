from fit import *
n,d,lr,y,groups,idx=load()
def prediction(p):
 x=np.log(lr)-p[0]-p[1]*np.log(n/1e8)-p[2]*np.log(d/1e9)
 base=p[5]+p[6]*(n/1e8)**(-p[8])+p[7]*(d/1e9)**(-p[9])
 return base+np.where(x<0,p[3],p[4])*x*x
# Size-dependent noise is visible in residuals; moderate weighting keeps the
# long-horizon small-model measurements influential while improving precision.
p,r=ls(lambda p:(prediction(p)-y)*(n/1e8)**.3,
       [-5.35,-.14,-.19,.063,.115,1.7,1.23,1.67,.326,.373])
def logopt(N,D):return p[0]+p[1]*np.log(N/1e8)+p[2]*np.log(D/1e9)
def point(v):return {'lo':round(float(v),6),'hi':round(float(v),6)}
a=logopt(1e9,2e10)/np.log(10)
b=logopt(1e9,1e12)/np.log(10)
x=np.log(.001453)-logopt(1e9,1e12)
excess=(p[4] if x>0 else p[3])*x*x
optloss=p[5]+p[6]*10**(-p[8])+p[7]*1000**(-p[9])
options=np.array([.000542,.00108,.00217,.00434])
xopt=np.log(options)-logopt(1e9,2e11)
penalties=np.where(xopt<0,p[3],p[4])*xopt*xopt
answers={'q1':point(a),'q2':point(b),'q3':point(excess),
 'q4':{'lo':round(float(b),6),'hi':round(float(b+.5*np.log10(8)),6)},
 'q5':{'verdict':'refuted'},'q6':{'verdict':'undetermined'},
 'q7':{'choice':'ABCD'[np.argmin(penalties)]},'q8':point(optloss)}
Path('/app/answers.json').write_text(json.dumps(answers,indent=2)+'\n')
Path('/app/analysis/results.json').write_text(json.dumps({
 'parameters':p.tolist(),'rmse':float(np.sqrt(np.mean((prediction(p)-y)**2))),
 'model_size_multiplier_for_10x':float(10**p[1]),
 'production_lr_range':[float(10**b),float(10**b*np.sqrt(8))],
 'q7_option_excess_losses':dict(zip('ABCD',penalties.tolist()))},indent=2)+'\n')
assert set(answers)=={f'q{i}' for i in range(1,9)}
for k,v in answers.items():
 if 'lo' in v:assert np.isfinite(v['lo']) and np.isfinite(v['hi']) and v['lo']<=v['hi']
print(json.dumps(answers,indent=2))
