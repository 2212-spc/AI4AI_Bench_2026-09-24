import json, subprocess, numpy as np
from pathlib import Path
P=Path('/app/runs.json')
def run(n,d,m,pool=None):
    cfg={'mix':dict(zip(['web','code','math','papers'],map(float,m)))}
    if pool is not None: cfg['pool']=dict(zip(['web','code','math','papers'],map(float,pool)))
    out=subprocess.check_output(['/app/bin/lab','train',f'N={n}',f'D={d}','--json',json.dumps(cfg)],text=True)
    r=json.loads(out); data=json.loads(P.read_text()) if P.exists() else []
    data.append(dict(N=n,D=d,**cfg,**r)); P.write_text(json.dumps(data,indent=2))
    print(len(data),n,d,np.round(m,3),r.get('eval_loss'),r.get('budget_left'),flush=True)
    return r
if __name__=='__main__':
    mixes=np.r_[np.eye(4), [[.25]*4],[[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7]],np.ones((6,4))*.05]
    k=9
    for i in range(4):
      for j in range(i):
        mixes[k,i]=mixes[k,j]=.45; k+=1
    for m in mixes: run(5e7,1e9,m)
    for m in np.eye(4): run(5e7,4e9,m)
    for n in [1e8,2e8,4e8]: run(n,1e9,[.25]*4)
    for d in [2e9,4e9,8e9]: run(5e7,d,[.25]*4)
