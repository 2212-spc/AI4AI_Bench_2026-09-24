import sys
from optimize import *
result=json.loads((ROOT/'analysis/proxy_result.json').read_text());fs=result['schedule']
# Use all-B at quiet hours; sub-percent artifacts at autoscaler steps are not material.
start,end=map(int,sys.argv[1:3]);out={}
for h in range(start,end):
    if h<10 or h>20:
        f=1.;best=hour_score(h,f,True)
    else:
        cand=np.round(np.arange(fs[h]-.045,fs[h]+.026,.005),3)
        vals=[hour_score(h,float(f),True) for f in cand]
        i=np.argmax(vals);f=float(cand[i]);best=float(vals[i])
    out[h]={'fraction':f,'score':best}
    print(h,out[h],flush=True)
(ROOT/f'analysis/refined_{start}_{end}.json').write_text(json.dumps(out,indent=2)+'\n')
