import itertools, numpy as np, policy
from sweep import run
grid=dict(MAX_PER_Q=[14,20], PHASE1_HARD=[3,5])
for combo in itertools.product(*grid.values()):
    p=dict(zip(grid.keys(),combo)); r=run(p, reps=8)
    print(round(np.mean([r[s]['all'] for s in r]),4), p, {s:(r[s]['all'], r[s]['algebra'], r[s]['geometry'], r[s]['number_theory']) for s in r}, flush=True)
