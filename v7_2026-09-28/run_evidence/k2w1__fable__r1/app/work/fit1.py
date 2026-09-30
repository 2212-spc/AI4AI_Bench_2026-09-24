import numpy as np; from nm import fit; from load import load
rows=[r for r in load() if r['pool'] is None and abs(r['mix']['web']-0.3)<1e-9 and abs(r['mix']['math']-0.3)<1e-9]
print(len(rows))
for ev in ['general','code','math']:
    N=np.array([r['N'] for r in rows]); D=np.array([r['D'] for r in rows]); L=np.array([r['loss'][ev] for r in rows])
    def model(p,N,D):
        E,lA,al,lB,be=p; return E+np.exp(lA)*N**(-al)+np.exp(lB)*D**(-be)
    def obj(p): return np.sum((model(p,N,D)-L)**2)
    p,v=fit(obj,[1.5,np.log(2e2),0.3,np.log(1e3),0.3],restarts=8,step=0.3)
    print(ev,'E=%.3f A=%.3g al=%.3f B=%.3g be=%.3f rmse=%.4f'%(p[0],np.exp(p[1]),p[2],np.exp(p[3]),p[4],np.sqrt(v/len(L))))
    print('  target pred', model(p,2.5e9,5e11))
    for r in rows: print('   ',r['N'],r['D'],r['loss'][ev],round(float(model(p,r['N'],r['D'])),4))
