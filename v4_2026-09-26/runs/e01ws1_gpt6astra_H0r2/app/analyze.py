import json
import numpy as np
from pathlib import Path
root=Path('/app')
rows=[json.loads(x) for x in (root/'lab_runs.jsonl').read_text().splitlines()]
prior=[json.loads(x) for x in (root/'notebook/runs.jsonl').read_text().splitlines()]
corpus={x['item']:x['occurrences'] for x in json.load(open(root/'corpus_scan.json'))['items']}
measurements={}
for model in ['M_new','M_ref']:
    runs=[r for r in rows+prior if r.get('model')==model and r.get('fmt')=='mc_letter' and r.get('split')=='all']
    acc=np.concatenate([r['reps_acc'] for r in runs])
    release=np.concatenate([np.array(r['reps_acc'])*r['n_scored']/r['n_items'] for r in runs])
    detailed=[r for r in runs if 'detail' in r]
    a=[]
    for r in detailed:
        d=r['detail']
        all_accuracy=np.mean([bool(x['correct']) for x in d])
        clean_accuracy=np.mean([bool(x['correct']) for x in d if corpus[x['item']]==0])
        all_slice=[np.mean([bool(x['correct']) for x in d if x['slice']==s]) for s in range(3)]
        clean_slice=[np.mean([bool(x['correct']) for x in d if x['slice']==s and corpus[x['item']]==0]) for s in range(3)]
        a.append([all_accuracy,clean_accuracy]+all_slice+clean_slice)
    a=np.array(a)
    # The full-bank score is a control variate for call-level fluctuations.
    # Repeated calls estimate its expectation much more precisely than the
    # limited first-repetition item-level tables.
    slope=np.cov(a,rowvar=False,ddof=1)[0,:]/np.var(a[:,0],ddof=1)
    corrected=a.mean(axis=0)+slope*(release.mean()-a[:,0].mean())
    residual=a-a[:,0,None]*slope
    residual_se=residual.std(axis=0,ddof=1)/len(a)**.5
    print(model,'repetitions',len(acc),'acc mean,SD',acc.mean(),acc.std(ddof=1),'release mean',release.mean())
    print('raw',a.mean(axis=0))
    print('slopes',slope)
    print('control-variate estimates',corrected,'residual SE',residual_se)
    measurements[model]={'repetitions':len(acc),'reported_acc_sd':float(acc.std(ddof=1)),'release_accuracy':float(release.mean()),'corrected':corrected.tolist(),'residual_se':residual_se.tolist()}
    if model=='M_new':
        new_acc=acc
        new_release=release.mean()
        clean=corrected[1]
        # Estimate each slice directly using the dedicated runs.
        estimates=[];variances=[]
        for s in range(3):
            sr=[r for r in rows if r.get('model')==model and r.get('split')==f's{s}'][0]
            v=np.array(sr['reps_acc'])*sr['n_scored']/sr['n_items']
            estimates.append(v.mean());variances.append(v.var(ddof=1)/len(v))
            print('slice',s,'direct estimate',v.mean(),'SE',v.std(ddof=1)/len(v)**.5)
        # Combine slice estimates from the item-level calls (adjusted for
        # their call-level fluctuation) with independent dedicated slice runs.
        estimates=np.array(estimates);variances=np.array(variances)
        control_cov=np.cov(residual[:,2:5],rowvar=False,ddof=1)/len(a)
        control_cov+=np.outer(slope[2:5],slope[2:5])*release.var(ddof=1)/len(release)
        precision_control=np.linalg.inv(control_cov)
        precision_direct=np.diag(1/variances)
        combined_cov=np.linalg.inv(precision_control+precision_direct)
        combined=combined_cov@(precision_control@corrected[2:5]+precision_direct@estimates)
        weights=np.array([471,507,522])/1500
        updated_release=weights@combined
        clean+=slope[1]*(updated_release-new_release)
        new_release=updated_release
        print('combined slice estimates',combined,'combined whole bank',new_release)
        q3=sorted([w*combined[0]+(1-w)*(2*combined[1]+combined[2])/3 for w in [.15,.85]])
        measurements['new_slice_accuracy']=combined.tolist()
        measurements['q3_endpoints']=q3

diff=np.array(measurements['M_new']['corrected'][5:])-np.array(measurements['M_ref']['corrected'][5:])
gap=[w*diff[0]+(1-w)*(2*diff[1]+diff[2])/3 for w in [.15,.85]]
print('Clean slice differences',diff,'release endpoint gaps',gap)
measurements['clean_slice_differences']=diff.tolist()
measurements['clean_release_gap_endpoints']=gap
answers={
 'q1':{'lo':float(clean),'hi':float(clean)},
 'q2':{'lo':float(new_release-clean),'hi':float(new_release-clean)},
 'q3':{'lo':float(q3[0]),'hi':float(q3[1])},
 'q4':{'lo':float(new_acc.std(ddof=1)),'hi':float(new_acc.std(ddof=1))},
 'q5':{'verdict':'supported' if min(gap)>0 else 'refuted' if max(gap)<=0 else 'undetermined'},
}
(root/'analysis_results.json').write_text(json.dumps(measurements,indent=2)+'\n')
(root/'answers.json').write_text(json.dumps(answers,indent=2)+'\n')
print(json.dumps(answers,indent=2))
