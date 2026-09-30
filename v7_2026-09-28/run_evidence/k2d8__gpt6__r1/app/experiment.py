import json, subprocess, numpy as np
DOM=['web','code','math','papers']
def run(m,N=5e7,D=1e9,pool=None):
    args={'mix':dict(zip(DOM,map(float,m)))}
    if pool is not None: args['pool']=dict(zip(DOM,map(float,pool)))
    p=subprocess.run(['/app/bin/lab','train',f'N={N}',f'D={D}','--json',json.dumps(args)],capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    r=json.loads(p.stdout)
    with open('/app/runs.jsonl','a') as f:f.write(json.dumps(dict(N=N,D=D,**args,result=r))+'\n')
    print(r['call_id'],m, pool, r['eval_loss'],r['budget_left'],flush=True)
    return r
if __name__=='__main__':
    for m in np.eye(4):run(m)
    for m in [[.25]*4,[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7]]:run(m)
    for factor in [1,.3,.1,.03,.01]:run([.25]*4,pool=np.ones(4)*2.5e8*factor)
