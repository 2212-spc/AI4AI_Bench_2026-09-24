import json,subprocess,numpy as np
DOM=['web','code','math','papers']
U=np.array([3e12,15625823631.892616,20747911539.64751,50433006481.05815])
def run(w,N=5e7,D=1e9,pool=None,tag=''):
 conf={'mix':dict(zip(DOM,map(float,w)))}
 if pool is not None:conf['pool']=dict(zip(DOM,map(float,pool)))
 p=subprocess.run(['/app/bin/lab','train',f'N={N}',f'D={D}','--json',json.dumps(conf)],capture_output=True,text=True)
 try:r=json.loads(p.stdout)
 except:raise RuntimeError(p.stdout+p.stderr)
 if 'eval_loss' not in r:raise RuntimeError(r)
 row=dict(N=N,D=D,w=list(map(float,w)),pool=None if pool is None else list(map(float,pool)),tag=tag,result=r)
 with open('/app/runs.jsonl','a') as f:f.write(json.dumps(row)+'\n')
 print(tag,w,r['eval_loss'],r['budget_left'],flush=True)
 return r
if __name__=='__main__':
 for i in range(4):run(np.eye(4)[i],tag='pure')
 for i in range(4):
  for e in [1,3,10,30,100]:
   pool=U.copy();pool[i]=1e9/e
   run(np.eye(4)[i],pool=pool,tag=f'epoch{e}')
 for w in [[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7],[.25]*4]:
  run(w,tag='mixfull')
  run(w,pool=U/500,tag='mixtargetepochs')
