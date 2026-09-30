from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('tn','/app/work/train_new.py'); tn=importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)
cfg=json.load(open('/app/work/config_new.json')); cfg['num_classes']=10
Xu, first = np.unique(X, axis=0, return_index=True); yu=y[first]

def loss_grad(z, yb, k, kind, par):
    n=len(yb); zc=z-z.max(1,keepdims=True); logp=zc-np.log(np.exp(zc).sum(1,keepdims=True)); p=np.exp(logp)
    if kind=='ce':
        t=np.eye(k,dtype=np.float32)[yb]; return (p-t)/n
    if kind=='forward':   # q = (1-e) p + e/K ; loss = -log q_y
        e=par; q=(1-e)*p+e/k; qy=q[np.arange(n),yb]
        dq=np.zeros_like(p); dq[np.arange(n),yb]=-1.0/qy          # dL/dq
        dp=(1-e)*dq                                                # dL/dp
        # softmax backprop: dz = p*(dp - sum(dp*p))
        return p*(dp-(dp*p).sum(1,keepdims=True))/n
    if kind=='gce':       # loss = (1 - p_y^q)/q ; dL/dp_y = -p_y^(q-1)
        q=par; py=p[np.arange(n),yb]
        dp=np.zeros_like(p); dp[np.arange(n),yb]=-(py**(q-1))
        return p*(dp-(dp*p).sum(1,keepdims=True))/n
    if kind=='sce':       # symmetric CE: CE + a * RCE(with clip -log(1e-4))
        a=par; t=np.eye(k,dtype=np.float32)[yb]
        dce=(p-t)
        # RCE = -sum_j p_j log t_j, log t_j = A for j!=y, 0 for j=y  => RCE = -A (1-p_y); dRCE/dp_y = A ... -> dp_y = A? derivative of -A(1-p_y) wrt p_y = A (A=-log(1e-4)=9.2 negative sign): RCE = A(1-p_y)
        A=-np.log(1e-4); dp=np.zeros_like(p); dp[np.arange(n),yb]=-A
        drce=p*(dp-(dp*p).sum(1,keepdims=True))
        return (dce+a*drce)/n
    raise ValueError

def train_loss(Xs, y, steps, wd, kind='ce', par=0.0, seed=0, hidden=256, lr=0.003, bs=128):
    rng=np.random.default_rng(seed); k=10
    P=tn.init_params(Xs.shape[1],hidden,k,rng); opt=tn.AdamW(P,lr,wd); n=len(Xs)
    for t in range(steps):
        idx=rng.integers(0,n,bs); z,cache=tn.forward(P,Xs[idx])
        dz=loss_grad(z,y[idx],k,kind,par); G=tn.backward(P,cache,dz)
        opt.step(P,G,tn.lr_multiplier(t,steps,0.02))
    return P

def lda_fit(A,t,shrink=0.02,k=10):
    means=np.array([A[t==c].mean(0) for c in range(k)]); pri=(np.bincount(t,minlength=k)+1)/(len(t)+k)
    R=A-means[t]; S=R.T@R/len(A); S=(1-shrink)*S+shrink*np.eye(A.shape[1])*np.trace(S)/A.shape[1]; Si=np.linalg.inv(S)
    def f(Q):
        sc=np.array([-0.5*np.einsum('ij,jk,ik->i',Q-m,Si,Q-m)+np.log(p) for m,p in zip(means,pri)]).T
        return softmax(sc)
    return f
