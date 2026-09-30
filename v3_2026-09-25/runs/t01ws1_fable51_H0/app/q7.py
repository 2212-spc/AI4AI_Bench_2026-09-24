import numpy as np
exec(open('/app/boot.py').read().split("print(\"param means\"")[0])
def lnlr(th,N,D): return th[0]+th[1]*np.log(N/1e8)+th[2]*np.log(D/2e9)
def excess(t,k,N,D,lr):
    u=np.log(lr)-lnlr(t,N,D); return k*shape(u,t[3])
d=np.array([excess(t[:4],t[4],1e9,2e11,0.00033)-excess(t[:4],t[4],1e9,2e11,0.000661) for t in out])
print("q7 excess(B)-excess(C): mean %.4f sd %.4f  frac B better %.2f"%(d.mean(),d.std(),(d<0).mean()))
# alt: free-model coefficients
for th in [np.array([-6.378,-0.107,-0.222,0.466])]:
    print("alt lr* at D=2e11: %.6f ; B/C midpoint %.6f"%(np.exp(lnlr(th,1e9,2e11)),np.sqrt(0.00033*0.000661)))
    for lr in [0.00033,0.000661]: print(lr, excess(th,0.163,1e9,2e11,lr))
