"""Train the nightly classifier with numpy only."""
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def init_params(d, h, k, rng):
    return {"W1": (rng.standard_normal((d,h))*np.sqrt(2.0/d)).astype(np.float32), "b1": np.zeros(h,np.float32),
            "W2": (rng.standard_normal((h,h))*np.sqrt(2.0/h)).astype(np.float32), "b2": np.zeros(h,np.float32),
            "W3": (rng.standard_normal((h,k))*np.sqrt(1.0/h)).astype(np.float32), "b3": np.zeros(k,np.float32)}

def forward(P,x):
    a1=x@P["W1"]+P["b1"]; h1=np.maximum(a1,0)
    a2=h1@P["W2"]+P["b2"]; h2=np.maximum(a2,0)
    return h2@P["W3"]+P["b3"], (x,a1,h1,a2,h2)

def backward(P,c,dz):
    x,a1,h1,a2,h2=c; G={"W3":h2.T@dz,"b3":dz.sum(0)}
    da2=(dz@P["W3"].T)*(a2>0); G["W2"]=h1.T@da2;G["b2"]=da2.sum(0)
    da1=(da2@P["W2"].T)*(a1>0);G["W1"]=x.T@da1;G["b1"]=da1.sum(0)
    return G

class AdamW:
    def __init__(self,P,lr,wd):
        self.lr=lr;self.wd=wd;self.t=0
        self.m={n:np.zeros_like(v) for n,v in P.items()};self.v={n:np.zeros_like(v) for n,v in P.items()}
    def step(self,P,G,mult):
        self.t+=1;lr=self.lr*mult
        for n in P:
            self.m[n]=.9*self.m[n]+.1*G[n];self.v[n]=.999*self.v[n]+.001*G[n]*G[n]
            u=(self.m[n]/(1-.9**self.t))/(np.sqrt(self.v[n]/(1-.999**self.t))+1e-8)
            if n[0]=='W':u=u+self.wd*P[n]
            P[n]-=(lr*u).astype(np.float32)

def lr_multiplier(t,steps,warm):
    w=max(1,int(warm*steps))
    return (t+1)/w if t<w else .5*(1+np.cos(np.pi*(t-w)/max(1,steps-w)))

def make_transform(X,ridge):
    # Standardisation followed by a small-ridge ZCA transform.  This makes the
    # optimizer see comparable directions when the raw features are correlated.
    mu=X.mean(0,dtype=np.float64).astype(np.float32);sd=(X.std(0,dtype=np.float64)+1e-6).astype(np.float32)
    Z=np.ascontiguousarray((X-mu)/sd)
    eig,U=np.linalg.eigh((Z.T@Z)/len(Z))
    A=((U/np.sqrt(np.maximum(eig,0)+ridge))@U.T).astype(np.float32)
    def transform(V): return np.ascontiguousarray(((V-mu)/sd)@A)
    return transform(X),transform

def train(X,y,steps,cfg,seed=0):
    rng=np.random.default_rng(seed);k=int(cfg.get('num_classes',int(y.max())+1))
    Xs,norm=make_transform(X,float(cfg.get('whiten_ridge',0.01)))
    P=init_params(Xs.shape[1],int(cfg['hidden']),k,rng);E={n:v.copy() for n,v in P.items()}
    opt=AdamW(P,float(cfg['lr']),float(cfg['weight_decay']));bs=int(cfg['batch_size']);q=float(cfg.get('robust_q',0))
    ema=float(cfg.get('ema',0)); aug=float(cfg.get('aug_sigma',0)); smooth=float(cfg.get('label_smoothing',0))
    for t in range(steps):
        ii=rng.integers(0,len(Xs),bs);xb=Xs[ii];yb=y[ii]
        if aug: xb=xb+aug*rng.standard_normal(xb.shape).astype(np.float32)
        z,c=forward(P,xb);z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        py=p[np.arange(bs),yb].copy(); target=np.zeros_like(p);target[np.arange(bs),yb]=1
        if smooth: target=target*(1-smooth)+smooth/k
        dz=p-target
        if q:
            # Generalized cross entropy downweights examples whose crowd label
            # has become implausible, while retaining a useful gradient early.
            r=q*min(1.0,(t+1)/max(1.0,.12*steps));dz*=np.maximum(py,1e-8)[:,None]**r
        G=backward(P,c,dz/bs);opt.step(P,G,lr_multiplier(t,steps,float(cfg['warmup_frac'])))
        if ema:
            rate=min(ema,1-1/(t+1))
            for n in E:E[n]=rate*E[n]+(1-rate)*P[n]
    return E if ema else P,norm

def predict(P,norm,X):
    out=[]
    for i in range(0,len(X),4096):out.append(forward(P,norm(X[i:i+4096]))[0].argmax(1))
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--predict',required=True);ap.add_argument('--out',required=True);ap.add_argument('--config',default=os.path.join(HERE,'config.json'));ap.add_argument('--seed',type=int,default=0);a=ap.parse_args()
    cfg=json.load(open(a.config));d=np.load(a.data);t=time.time();P,n=train(d['X'].astype(np.float32),d['y'].astype(np.int64),a.steps,cfg,a.seed)
    np.save(a.out,predict(P,n,np.load(a.predict).astype(np.float32)).astype(np.int64));print('trained %d steps in %.1fs'%(a.steps,time.time()-t))
if __name__=='__main__':main()
