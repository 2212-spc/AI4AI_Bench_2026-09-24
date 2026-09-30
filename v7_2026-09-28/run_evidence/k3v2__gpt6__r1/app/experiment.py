import subprocess,json
from pathlib import Path
p=Path('/app/observations.jsonl')
def run(n,steps,eta,seed=0):
 r=subprocess.run(['/app/bin/lab','train',f'n={n}',f'steps={steps}',f'eta={eta}',f'seed={seed}'],capture_output=True,text=True)
 try: d=json.loads(r.stdout)
 except: print(r.stdout,r.stderr);raise
 with p.open('a') as f:f.write(json.dumps(d)+'\n')
 print(json.dumps(d),flush=True)
 return d
if __name__=='__main__':
 for n,s,etas in [(256,1000,[.003,.01,.02,.03,.04,.06,.08]),(512,1000,[.01,.015,.02,.03,.04]),(1024,500,[.003,.007,.01,.015,.02]),(2048,200,[.003,.005,.008,.012])]:
  for e in etas:run(n,s,e)
