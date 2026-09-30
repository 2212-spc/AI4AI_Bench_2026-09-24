"""Production poststratification with sequential, finite-population auditing.

Calibration counts are human labels purchased in the two development arenas;
no development prompt IDs or production weights are retained.  Conditional
judge-error probabilities have a partially pooled beta-binomial model.
"""
import math
import random
import bisect
from statistics import NormalDist

# (human labels, disagreements), keyed by topic, length, judge verdict.
_CAL = {
 ('code','long',False):(10,0), ('code','long',True):(16,1),
 ('code','short',False):(10,0), ('code','short',True):(10,1),
 ('factual','long',False):(10,0), ('factual','long',True):(10,0),
 ('factual','short',False):(16,1), ('factual','short',True):(10,0),
 ('math','long',False):(10,0), ('math','long',True):(10,2),
 ('math','short',False):(23,2), ('math','short',True):(10,0),
 ('writing','long',False):(10,0), ('writing','long',True):(10,0),
 ('writing','short',False):(33,2), ('writing','short',True):(10,0),
}


def estimate(env):
    rng = random.Random(731905)
    pools = {}
    sizes = {}
    for row in env.rows:
        s = row['topic'] + '|' + row['length']
        key = (row['topic'], row['length'], bool(row['judge_win']))
        pools.setdefault(key, []).append(row['id'])
        sizes[s] = sizes.get(s, 0) + 1
    keys = sorted(pools)
    d = len(keys)
    for pool in pools.values():
        rng.shuffle(pool)
    N = [len(pools[k]) for k in keys]
    c = [(1 if not k[2] else -1) * env.production_mix[k[0]+'|'+k[1]] /
         sizes[k[0]+'|'+k[1]] for k in keys]
    base = sum(-c[j]*N[j] for j,k in enumerate(keys) if k[2])
    n = [0]*d
    e = [0]*d
    # Integrate over both the mean error rate and pooling strength instead
    # of treating empirically fitted hyperparameters as known constants.
    states = []
    for i in range(26):
        m = math.exp(math.log(.001)+(i+.5)/26*math.log(400))
        for j in range(20):
            strength = math.exp(math.log(2)+(j+.5)/20*math.log(75))
            a, b = m*strength, (1-m)*strength
            lp = math.log(m)+18*math.log1p(-m)  # Beta(1,19), log-grid Jacobian
            aa, tt = [], []
            for key in keys:
                nn, ee = _CAL.get(key, (0,0))
                lp += (math.lgamma(a+ee)-math.lgamma(a)
                       +math.lgamma(b+nn-ee)-math.lgamma(b)
                       +math.lgamma(strength)-math.lgamma(strength+nn))
                aa.append(a+ee)
                tt.append(strength+nn)
            states.append([lp,aa,tt])

    def posterior():
        peak = max(h[0] for h in states)
        weights = [math.exp(h[0]-peak) for h in states]
        z = sum(weights)
        return [w/z for w in weights]

    # Optimal one-step reduction of posterior variance of the *weighted*
    # target.  This includes cross-cell uncertainty from partial pooling.
    budget = min(int(env.label_budget), sum(N))
    for step in range(budget):
        weights = posterior()
        U = [N[j]-n[j] for j in range(d)]
        means = [0.0]*d
        cov = [0.0]*d
        grand = 0.0
        for w,h in zip(weights,states):
            pp = [(h[1][j]+e[j])/(h[2][j]+n[j]) for j in range(d)]
            mu = sum(c[j]*U[j]*pp[j] for j in range(d))
            grand += w*mu
            for j in range(d):
                if U[j]:
                    p = pp[j]
                    t = h[2][j]+n[j]
                    means[j] += w*p
                    cov[j] += w*(mu*p+c[j]*p*(1-p)*(t+U[j])/(t+1))
        scores = [((cov[j]-grand*means[j])**2 /
                   max(1e-14,means[j]*(1-means[j]))) if U[j] else -1
                  for j in range(d)]
        j = max(range(d), key=lambda j:scores[j])
        ident = pools[keys[j]][n[j]]
        got = env.label([ident])
        human = got[str(ident)] if str(ident) in got else got[ident]
        error = int(bool(human) != keys[j][2])
        for h in states:
            p = (h[1][j]+e[j])/(h[2][j]+n[j])
            h[0] += math.log(p if error else 1-p)
        n[j] += 1
        e[j] += error

    weights = posterior()
    U = [N[j]-n[j] for j in range(d)]
    known = base+sum(c[j]*e[j] for j in range(d))
    point = known + sum(w*sum(c[j]*U[j]*(h[1][j]+e[j]) /
                                      (h[2][j]+n[j]) for j in range(d))
                        for w,h in zip(weights,states))
    cumulative=[]
    z=0.0
    for w in weights:
        z+=w
        cumulative.append(z)
    cumulative[-1]=1.0
    # Beta-binomial prediction for the as-yet unlabelled finite population:
    # audited labels are known exactly, not another random future sample.
    draws=[]
    for _ in range(7000):
        h=states[bisect.bisect_left(cumulative,rng.random())]
        value=known
        for j in range(d):
            if U[j] and c[j]:
                a=h[1][j]+e[j]
                b=h[2][j]+n[j]-a
                p=rng.betavariate(a,b)
                errors=sum(rng.random()<p for _ in range(U[j]))
                value+=c[j]*errors
        draws.append(value)
    draws.sort()
    coverage=float(env.target_coverage)
    tail=(1-coverage)/2
    lo=draws[max(0,int(tail*len(draws)))]
    hi=draws[min(len(draws)-1,int((1-tail)*len(draws)))]
    env.submit(max(0.0,min(1.0,point)),max(0.0,lo),min(1.0,hi))
