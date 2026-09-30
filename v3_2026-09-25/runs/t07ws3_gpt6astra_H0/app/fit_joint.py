from fit_scaling import *
# fit all points with separate schedule gap amplitudes and a WSD decay exponent.
rows=[(n,d,l,0,0,0) for n,d,l in finished]
for n,d,l,f,s in ck:
    phase=.5*(1+np.cos(np.pi*f)) if s=='cosine' else min(1,(1-f)/.2)
    rows.append((n,d,l,phase if s=='cosine' else 0,phase if s=='wsd' else 0,f))
x=np.array(rows);n,d,y,pc,pw,f=x.T

def jointfit(par):
    a,b,g=par
    X=np.array([np.ones(len(n)),(n/1e8)**-a,(d/1e9)**-b,pc,pw**g]).T
    co=np.linalg.lstsq(X,y,rcond=None)[0]
    return np.mean((X@co-y)**2),co
p=minimize(lambda p:jointfit(p)[0],[.4,.33,1.],[.03,.03,.1]);err,co=jointfit(p)
E,A,B,kc,kw=co;a,b,g=p
print('joint',dict(E=E,A=A,B=B,alpha=a,beta=b,cosine=kc,wsd=kw,wsd_decay=g,rmse=err**.5))
def L(n,d):return E+A*(np.array(n)/1e8)**-a+B*(np.array(d)/1e9)**-b
print('prod',L(3e9,6e10))
print('q3',kc*.5*(1+np.cos(.4*np.pi)))
print('q4',[L(3e9,6e10*f)+kc*.5*(1+np.cos(f*np.pi))-L(3e9,6e10) for f in [.9,.5]])
print('q5 difference',L(3e9,6e10*.8)+kc*.5*(1+np.cos(.8*np.pi))-L(3e9,6e10*.6))
