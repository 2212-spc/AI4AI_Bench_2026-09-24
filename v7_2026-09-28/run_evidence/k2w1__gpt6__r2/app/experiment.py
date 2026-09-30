import json,subprocess,numpy as np
DOM=['web','code','math','papers']
def run(mix,N=5e7,D=1e9,pool=None):
    mix=dict(zip(DOM,map(float,mix))) if not isinstance(mix,dict) else mix
    arg={'mix':mix}
    if pool is not None:arg['pool']=dict(zip(DOM,map(float,pool))) if not isinstance(pool,dict) else pool
    p=subprocess.run(['/app/bin/lab','train',f'N={N}',f'D={D}','--json',json.dumps(arg)],capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    out=json.loads(p.stdout)
    rec={'N':N,'D':D,**arg,**out}
    with open('/app/runs.jsonl','a') as f:f.write(json.dumps(rec)+'\n')
    print(out['call_id'],list(mix.values()),N,D,pool,out['eval_loss'],out['budget_left'],flush=True)
    return rec
if __name__=='__main__':
    for m in np.eye(4):run(m)
    for i in range(4):
        for j in range(i+1,4):
            m=np.zeros(4);m[i]=m[j]=.5;run(m)
    for n,d in [(1e8,1e9),(2e8,1e9),(4e8,1e9),(5e7,2e9),(5e7,4e9),(5e7,1e10)]:run([.25]*4,n,d)
    for epoch in [1,2,4,8,16,32]:run([.25]*4,pool=[.25e9/epoch]*4)
