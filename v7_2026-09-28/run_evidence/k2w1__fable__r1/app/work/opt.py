import numpy as np, json; from gfit import *; from nm import nelder_mead
def optimize(P,N=2.5e9,D=5e11):
    def obj(z):
        w=np.exp(z-z.max()); w/=w.sum(); mix=dict(zip(DOM,w))
        return predict(P,N,D,mix)['composite']
    best=None
    for s in range(6):
        z0=np.random.randn(4)*0.5
        z,v=nelder_mead(obj,z0,step=0.5,iters=3000)
        if best is None or v<best[1]: best=(z,v)
    w=np.exp(best[0]-best[0].max()); w/=w.sum()
    return dict(zip(DOM,np.round(w,4))),best[1]
if __name__=="__main__":
    p=np.array(json.load(open('params.json'))); P=unpack(p)
    mix,v=optimize(P); print('opt mix',mix,'comp',round(v,4))
    pr=predict(P,2.5e9,5e11,mix); print(pr)
    print('epochs', {d:5e11*mix[d]/U[d] for d in DOM})
    for m in [{'web':0.5,'code':0.2,'math':0.1,'papers':0.2},{'web':0.6,'code':0.2,'math':0.1,'papers':0.1},{'web':0.4,'code':0.3,'math':0.1,'papers':0.2}]:
        print(m, round(predict(P,2.5e9,5e11,m)['composite'],4))
