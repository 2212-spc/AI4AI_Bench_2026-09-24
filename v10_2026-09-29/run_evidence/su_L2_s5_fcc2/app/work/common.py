import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np, json, sys, time
sys.path.insert(0,'/app/repo')
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'].astype(np.float32),s['y'].astype(np.int64)
Xd,yd=d['X'].astype(np.float32),d['y'].astype(np.int64)
K=10
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
