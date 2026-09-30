exec(open('/app/analyze.py').read().split("if __name__")[0])
# load group opts from repeated copy and baseline p output by replicate fitting? use function above? redo groups from rows and grouped p saved
p=np.load('/app/fit.npz')['grouped']
# Need grouped p overwritten latest saved yes
# groups unavailable after exec? yes variables
base=p[5:]
x=np.log(groups[:,0]/1e8); z=np.log(groups[:,1]/1e10); yb=base

def fun(q): return q[0]+np.exp(q[1]-q[2]*x)+np.exp(q[3]-q[4]*z)
q,s=fit(fun,[1.5,.2,.35,-.7,.35],yb)
print('basefit',q,'rmse',np.sqrt(s/len(yb)))
for G,B in zip(groups,yb): print(G,B,fun(q)[list(map(tuple,groups)).index(tuple(G))])
for n,d in [(1e9,2e10),(1e9,1e12),(1e9,2e11)]:
 xx=np.log(n/1e8);zz=np.log(d/1e10); print(n,d,q[0]+np.exp(q[1]-q[2]*xx)+np.exp(q[3]-q[4]*zz))
