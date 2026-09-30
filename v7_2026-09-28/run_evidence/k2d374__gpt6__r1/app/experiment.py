import subprocess,json,numpy as np
DOM=['web','code','math','papers']
def run(m,D=1e9,N=5e7,pool=None):
    p={'mix':dict(zip(DOM,map(float,m)))}
    if pool is not None:p['pool']=dict(zip(DOM,map(float,pool)))
    r=subprocess.run(['/app/bin/lab','train',f'N={N}',f'D={D}','--json',json.dumps(p)],capture_output=True,text=True)
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    y=json.loads(r.stdout)
    with open('/app/data.jsonl','a') as f:f.write(json.dumps(dict(N=N,D=D,**p,**y))+'\n')
    print(y['call_id'],m,D,N,pool,y['eval_loss'],flush=True)
    return y
if __name__=='__main__':
 for i in range(4):
  m=np.eye(4)[i]
  for D in [1e9,4e9]:run(m,D)
 for N in [1e8,2e8,4e8]:run([.25]*4,N=N)
 for D in [2e9,4e9,8e9]:run([.25]*4,D=D)
 for p in [1e9,1e8,1e7]:run([.25]*4,pool=[p]*4)
