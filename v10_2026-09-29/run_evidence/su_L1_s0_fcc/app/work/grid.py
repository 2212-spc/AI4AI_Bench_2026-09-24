import sys, json, numpy as np, itertools
from exp import run
n=int(sys.argv[1]); seeds=int(sys.argv[2])
for arg in sys.argv[3:]:
    kw=json.loads(arg)
    accs=[run(n,seed=s,**kw)[0] for s in range(seeds)]
    print(n, kw, 'dev %.3f +- %.3f'%(np.mean(accs), np.std(accs)/np.sqrt(seeds)), flush=True)
