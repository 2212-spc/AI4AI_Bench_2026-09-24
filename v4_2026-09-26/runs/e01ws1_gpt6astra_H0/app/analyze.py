import json
import numpy as np

runs=[json.loads(line) for line in open('/app/lab_runs.jsonl')]
occ={r['item']:r['occurrences'] for r in json.load(open('/app/corpus_scan.json'))['items']}
results={}
for model in ['M_new','M_ref']:
    ds=[r for r in runs if r.get('svc')=='score' and r.get('model')==model]
    fullacc=np.concatenate([r['reps_acc'] for r in ds])
    firstacc=np.array([r['reps_acc'][0] for r in ds])
    # Every detailed response is the first repetition. Use its whole-bank
    # accuracy as a control variate; all repetitions estimate its expectation.
    rates=np.array([[np.mean([bool(x['correct']) for x in r['detail'] if (s is None or x['slice']==s) and (not clean or occ[x['item']]==0)]) for s in [None,0,1,2] for clean in [False,True]] for r in ds])
    centered=firstacc-firstacc.mean()
    slopes=centered@rates/(centered@centered)
    estimates=rates.mean(axis=0)+slopes*(fullacc.mean()-firstacc.mean())
    residuals=rates-rates.mean(axis=0)-centered[:,None]*slopes
    print(model,'n_detail',len(ds),'n_reps',len(fullacc),'mean_acc',fullacc.mean(),'sd_acc',fullacc.std(ddof=1))
    print('raw',rates.mean(axis=0).reshape(4,2))
    print('adjusted',estimates.reshape(4,2))
    print('slopes',slopes.reshape(4,2))
    print('residual SE',(residuals.std(axis=0,ddof=2)/len(ds)**.5).reshape(4,2))
    # A quadratic check assesses curvature in the control-variate relationship.
    xx=np.column_stack([np.ones(len(ds)),firstacc,firstacc**2])
    quadratic=np.array([1,fullacc.mean(),np.mean(fullacc**2)])@np.linalg.lstsq(xx,rates,rcond=None)[0]
    print('quadratic',quadratic.reshape(4,2))
    results[model]={'estimates':estimates,'acc':fullacc}

new=results['M_new']['estimates']
ref=results['M_ref']['estimates']
weighted=[]
for w in [.15,.85]:
    weights=np.array([w,(1-w)*2/3,(1-w)/3])
    weighted.append(weights@new[[2,4,6]])
    print('w',w,'whole',weighted[-1],'clean_new',weights@new[[3,5,7]],'clean_ref',weights@ref[[3,5,7]],'gap',weights@(new-ref)[[3,5,7]])
answers={
 'q1':dict(lo=float(new[1]),hi=float(new[1])),
 'q2':dict(lo=float(new[0]-new[1]),hi=float(new[0]-new[1])),
 'q3':dict(lo=float(min(weighted)),hi=float(max(weighted))),
 'q4':dict(lo=float(results['M_new']['acc'].std(ddof=1)),hi=float(results['M_new']['acc'].std(ddof=1))),
 'q5':dict(verdict='undetermined'),
}
with open('/app/answers.json','w') as f:
    json.dump(answers,f,indent=2)
    f.write('\n')
print(json.dumps(answers,indent=2))
