from fit import *
rat=np.array([0 if row[3]=='end' else 1 if row[3]=='wsd_stable' else (1-float(row[3].split('_')[1]))/.2 if row[3].startswith('wsd') else .55+.45*np.cos(np.pi*float(row[3].split('_')[1])) for row in rows])
def f(p):
 E,A,al,B,be,K,ga=p
 return E+A*(n/1e8)**-al+B*(d/1e9)**-be+K*rat**ga
for exp in [0,.25,.5]:
 w=(n/1e8)**exp
 p=np.array([1.5,.7,.35,.94,.35,.11,.7]);lam=1e-6
 for it in range(1000):
  z=f(p);r=(z-y)*w
  J=np.column_stack([(f(p+np.eye(7)[j]*1e-6)-z)/1e-6 for j in range(7)])*w[:,None]
  st=np.linalg.solve(J.T@J+lam*np.eye(7),-J.T@r)
  if np.sum(((f(p+st)-y)*w)**2)<np.sum(r*r):p+=st;lam*=.8
  else:lam*=2
  if np.linalg.norm(st)<1e-10:break
 E,A,al,B,be,K,ga=p
 print('exp',exp,'params',p,'rmse',np.sqrt(np.mean((f(p)-y)**2)))
 base=lambda N,D:E+A*(N/1e8)**-al+B*(D/1e9)**-be
 pen=lambda f:K*(.55+.45*np.cos(np.pi*f))**ga
 print('q2',base(3e9,6e10),'q3',pen(.4),'q4',[(f,base(3e9,6e10*f)-base(3e9,6e10)+pen(f)-pen(1)) for f in [.9,.5]],'q5',base(3e9,3e10)+pen(.5)-base(3e9,2.1e10))
