import sys, numpy as np, json
from exp import *
part=int(sys.argv[1]); nparts=int(sys.argv[2])
cands = {
 'base': {},
 'ls0.1': dict(label_smoothing=0.1), 'ls0.3': dict(label_smoothing=0.3),
 'aug0.1': dict(aug_sigma=0.1), 'aug0.3': dict(aug_sigma=0.3), 'aug0.5': dict(aug_sigma=0.5),
 'lr0.001': dict(lr=0.001), 'lr0.01': dict(lr=0.01), 'lr0.03': dict(lr=0.03),
 'h64': dict(hidden=64), 'h128': dict(hidden=128), 'h512': dict(hidden=512),
 'bs32': dict(batch_size=32), 'bs512': dict(batch_size=512),
 'wd2': dict(weight_decay=2.0), 'wd1_lr0.01': dict(weight_decay=1.0, lr=0.01), 'wd1_aug0.3': dict(weight_decay=1.0, aug_sigma=0.3),
 'wd0.3_aug0.5': dict(weight_decay=0.3, aug_sigma=0.5),
}
jobs=[(k,s) for k in cands for s in range(3)]
res=[]
for i,(k,s) in enumerate(jobs):
    if i%nparts!=part: continue
    a,t=run(cands[k], seed=s); res.append((k,s,a,t))
json.dump(res,open(f'grid2_{part}.json','w'))
