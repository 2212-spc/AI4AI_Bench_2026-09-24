import json
import numpy as np

history = json.load(open('/app/full_history.json'))

model = {
    'general': {
        'E': 1.3658, 'A': 0.9445, 'alpha': 0.2250, 'beta': 0.2650, 'B_prim': 0.8896,
        'c': np.array([1.0000, 0.1914, 0.1030, 0.7812]) # [web, code, math, papers]
    },
    'code': {
        'E': 0.5648, 'A': 0.3888, 'alpha': 0.3500, 'beta': 0.2450, 'B_prim': 1.1654,
        'c': np.array([0.0515, 1.0000, 0.0869, 0.0433]) # [web, code, math, papers]
    },
    'math': {
        'E': 0.7367, 'A': 0.5283, 'alpha': 0.3500, 'beta': 0.3150, 'B_prim': 1.1738,
        'c': np.array([0.0524, 0.2359, 1.0000, 0.3444]) # [web, code, math, papers]
    }
}

# Repeat functions f(ep) = 1 + alpha * (1 - ep^(-beta))
repeat_params = {
    'web':    (0.0, 1.0),
    'code':   (4.960, 0.300),
    'math':   (1.010, 1.450),
    'papers': (2.330, 0.600)
}

def get_effective_tokens(d_seen, u, dom):
    if d_seen <= u:
        return d_seen
    ep = d_seen / u
    alpha, beta = repeat_params[dom]
    return u * (1.0 + alpha * (1.0 - ep**(-beta)))

def simulate(N, D, mix, pool=None):
    if pool is None:
        pool = {'web': 3e12, 'code': 4e10, 'math': 4e9, 'papers': 2e10}
    
    doms = ['web', 'code', 'math', 'papers']
    d_eff = []
    for dom in doms:
        d_seen = mix.get(dom, 0.0) * D
        u = pool.get(dom, 3e12 if dom == 'web' else 4e10 if dom == 'code' else 4e9 if dom == 'math' else 2e10)
        d_eff.append(get_effective_tokens(d_seen, u, dom))
    d_eff = np.array(d_eff) / 1e9 # in units of 1e9 tokens
    
    losses = {}
    for ev in ['general', 'code', 'math']:
        mod = model[ev]
        s = np.dot(d_eff, mod['c'])
        losses[ev] = mod['E'] + mod['A'] * (N / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
    
    comp = 0.34 * losses['general'] + 0.33 * losses['code'] + 0.33 * losses['math']
    return losses, comp

# Test simulate across ALL 69 runs in history
comp_diffs = []
for h in history:
    act_comp = h['result']['composite']
    losses, pred_comp = simulate(h['args']['N'], h['args']['D'], h['args']['mix'], h['args'].get('pool'))
    comp_diffs.append(act_comp - pred_comp)

comp_diffs = np.array(comp_diffs)
print(f"Across ALL 69 runs in history:")
print(f"Max abs error:  {np.max(np.abs(comp_diffs)):.5f}")
print(f"Mean abs error: {np.mean(np.abs(comp_diffs)):.5f}")
print(f"Std error:      {np.std(comp_diffs):.5f}")

