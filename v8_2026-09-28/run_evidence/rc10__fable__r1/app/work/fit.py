import runs, numpy as np, itertools
rows=runs.load()
for r in rows: r['mean']=float(r['res']['final_val_loss']); r['steps']=r['res']['steps']
lr0,wd0=0.000754,0.0318
Ts=[8,16,32,64,128]
data=[r for r in rows if r['tokens'] in Ts]
T=np.array([r['tokens'] for r in data],float); y=np.array([r['mean'] for r in data])
llr=np.log(np.array([r['lr']/lr0 for r in data])); lp=np.log(np.array([r['lr']*r['wd']*r['steps'] for r in data]))
t=np.log(T/32)
def design(theta):
    alpha,gamma,lp0,a,b=theta
    cols=[]
    for Ti in Ts: cols.append((T==Ti).astype(float))
    u=llr-alpha*t; v=lp-lp0-gamma*t
    cols.append(np.exp(a*t)*u*u); cols.append(np.exp(b*t)*v*v)
    # asymmetric low-p term: exp(-v)-1+v  (linex)
    cols.append(np.exp(b*t)*(np.exp(-v)-1+v))
    return np.stack(cols,1)
def loss(theta):
    A=design(theta); c,*_=np.linalg.lstsq(A,y,rcond=None)
    if c[-3]<0 or c[-2]<0 or c[-1]<0: return 1e9,c
    return np.sum((y-A@c)**2),c
rng=np.random.default_rng(0)
best=(1e9,None,None)
# coarse random search then local refine
for it in range(20000):
    th=np.array([rng.uniform(-0.9,-0.3), rng.uniform(0,1.2), rng.uniform(-1.5,1.5), rng.uniform(-0.5,1.5), rng.uniform(-0.5,2)])
    L,c=loss(th)
    if L<best[0]: best=(L,th,c)
L,th,c=best
for it in range(20000):
    th2=th+rng.normal(0,0.03,5)
    L2,c2=loss(th2)
    if L2<L: L,th,c=L2,th2,c2
print("theta alpha,gamma,lp0,a,b =",np.round(th,3)); print("coef",np.round(c,4)); print("resid_sd",np.sqrt(L/len(y)))
A=design(th); pred=A@c
for r,p in sorted(zip(data,pred),key=lambda z:(z[0]['tokens'],z[0]['lr'],z[0]['wd'])):
    print(f"  {r['tokens']}B lr={r['lr']/lr0:.2f}x wd={r['wd']/wd0:.2f}x y={r['mean']:.4f} pred={p:.4f} d={r['mean']-p:+.4f}")
alpha,gamma,lp0,a,b=th
for TT in [128,256,512]:
    tt=np.log(TT/32); steps=TT*1e9/(256*4096)
    lro=lr0*np.exp(alpha*tt); po=np.exp(lp0+gamma*tt); wdo=po/(lro*steps)
    # with linex the p-optimum shifts: minimize c_b*v^2 + c_l*(e^-v -1+v)
    vs=np.linspace(-2,2,4001); g=c[-2]*vs**2+c[-1]*(np.exp(-vs)-1+vs); vmin=vs[g.argmin()]
    po2=po*np.exp(vmin); wdo2=po2/(lro*steps)
    print(f"{TT}B: lr_opt={lro:.6f} ({lro/lr0:.3f}x) p_opt={po2:.2f} wd_opt={wdo2:.4f} ({wdo2/wd0:.2f}x)  curv_lr={c[-3]*np.exp(a*tt):.4f} curv_p={c[-2]*np.exp(b*tt):.4f} linex={c[-1]*np.exp(b*tt):.4f}")
np.save('theta.npy',th); np.save('coef.npy',c)
