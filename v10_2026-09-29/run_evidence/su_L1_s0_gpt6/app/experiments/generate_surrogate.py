import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
rng=np.random.default_rng(84652)
# Independent synthetic examples; used only for runtime and scale sensitivity.
x=rng.standard_normal((100000,32)).astype(np.float32)
w1=rng.standard_normal((32,96)).astype(np.float32)/np.sqrt(32)
w2=rng.standard_normal((96,96)).astype(np.float32)*np.sqrt(2/96)
w3=rng.standard_normal((96,10)).astype(np.float32)*np.sqrt(2/96)
z=np.maximum(np.maximum(x@w1,0)@w2,0)@w3
# Keep class imbalance moderate rather than letting one random bias dominate.
z=z-z.mean(0)+rng.standard_normal(10)*.2
gold=(z+.1*rng.standard_normal(z.shape)).argmax(1)
y=gold.copy(); mask=rng.random(len(x))<.2;y[mask]=rng.integers(0,10,mask.sum())
# Raw-feature transform provides the same order of conditioning as the real data.
u,_,v=np.linalg.svd(rng.standard_normal((32,32)));a=(u*np.geomspace(.2,2,32))@v
x=(x@a).astype(np.float32)
np.savez('/app/experiments/synthetic_corpus.npz',X=x[:80000],y=y[:80000])
np.save('/app/experiments/synthetic_test_X.npy',x[80000:])
np.save('/app/experiments/synthetic_gold.npy',gold[80000:])
