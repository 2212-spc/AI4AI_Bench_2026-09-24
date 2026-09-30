import sys, json, numpy as np, train2 as T
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'].astype(np.float32),s['y'].astype(np.int64); Xd,yd=d['X'].astype(np.float32),d['y'].astype(np.int64)
base=json.load(open('/app/repo/config.json'))
n=int(sys.argv[1]); val=int(sys.argv[2]); cands=json.loads(sys.argv[3]); seeds=int(sys.argv[4])
res=[]
for seed in range(seeds):
    rng=np.random.default_rng(100+seed); perm=rng.permutation(len(X)); idx=perm[:n]; vidx=perm[n:n+val]
    steps=int(32*n/128); probs=[]; vacc=[]; dacc=[]
    for wd in cands:
        cfg=dict(base); cfg['weight_decay']=wd
        M,norm=T.train(X[idx],y[idx],steps,cfg,seed=seed)
        pv=T.predict_proba(M,norm,X[vidx]); pd_=T.predict_proba(M,norm,Xd)
        vacc.append((pv.argmax(1)==y[vidx]).mean()); dacc.append((pd_.argmax(1)==yd).mean()); probs.append(pd_)
    order=np.argsort(vacc)[::-1]
    top1=dacc[order[0]]; oracle=max(dacc)
    top2=((probs[order[0]]+probs[order[1]]).argmax(1)==yd).mean()
    all3=(sum(probs).argmax(1)==yd).mean()
    res.append([top1,top2,all3,oracle]); print(seed,'val',np.round(vacc,3),'dev',np.round(dacc,3),'top1 %.3f top2 %.3f all %.3f oracle %.3f'%tuple(res[-1]),flush=True)
print('mean top1 top2 all oracle', np.round(np.mean(res,0),4))
