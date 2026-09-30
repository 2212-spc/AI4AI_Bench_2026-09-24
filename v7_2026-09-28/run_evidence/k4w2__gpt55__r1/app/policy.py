def run(env):
    qs=env.questions
    info={}
    for q in qs:
        z=[]
        for _ in range(2):
            a,s=env.sample(q['id'],0)
            z.append((a,s))
        conf=(z[0][1]+z[1][1])/2.0
        if z[0][0]!=z[1][0]: conf-=0.25
        info[q['id']]={'samples':z,'conf':conf}
    # Spend the shared reserve on the questions whose initial verifier evidence
    # is weakest.  Ten additional draws gives a useful consensus on difficult
    # instances, while retaining two draws for the easy majority.
    extra=env.budget-2*len(qs)
    order=sorted(qs,key=lambda q:info[q['id']]['conf'])
    for q in order:
        if extra<10: break
        z=info[q['id']]['samples']
        for _ in range(10):
            a,s=env.sample(q['id'],0);z.append((a,s))
        extra-=10
    i=0
    while extra>0:
        q=order[i%len(order)]
        a,s=env.sample(q['id'],0);info[q['id']]['samples'].append((a,s))
        extra-=1;i+=1
    out={}
    for q in qs:
        scores={}
        for a,s in info[q['id']]['samples']:
            scores[a]=scores.get(a,0.0)+min(s,0.5)
        out[q['id']]=max(scores,key=scores.get)
    env.submit(out)
