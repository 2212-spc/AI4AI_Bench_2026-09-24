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

# Collect all runs where epochs > 1 for some domain
repeat_runs = []
for h in history:
    args = h['args']
    res = h['result']
    m = args['mix']
    p = args.get('pool', {})
    
    d_w = m.get('web', 0.0) * args['D']
    d_c = m.get('code', 0.0) * args['D']
    d_m = m.get('math', 0.0) * args['D']
    d_p = m.get('papers', 0.0) * args['D']
    
    u_w = p.get('web', 3e12)
    u_c = p.get('code', 4e10)
    u_m = p.get('math', 4e9)
    u_p = p.get('papers', 2e10)
    
    if d_w > u_w or d_c > u_c or d_m > u_m or d_p > u_p:
        repeat_runs.append({
            'N': args['N'], 'D': args['D'],
            'r': np.array([m.get('web', 0.0), m.get('code', 0.0), m.get('math', 0.0), m.get('papers', 0.0)]),
            'u': np.array([u_w, u_c, u_m, u_p]),
            'd': np.array([d_w, d_c, d_m, d_p]),
            'ep': np.array([d_w/u_w, d_c/u_c, d_m/u_m, d_p/u_p]),
            'losses': res['eval_loss'],
            'comp': res['composite']
        })

print(f"Total repeat runs: {len(repeat_runs)}")

# Let's test functional forms for effective tokens D_eff(d_seen, u):
# If ep <= 1: d_eff = d_seen
# If ep > 1:
# Model A: d_eff = u * (1 + delta_d * log(ep))
# Model B: d_eff = u * (1 + c_d * (ep - 1)^gamma_d)
# Model C: d_eff = u * (1 + alpha_d * (1 - ep^(-beta_d)))

# Let's fit per-domain repeat functions:
# For domain d:
# In runs where only domain d has ep > 1:
# Let's inspect which runs have repeats for which domain:
for dom_idx, dom_name in enumerate(['web', 'code', 'math', 'papers']):
    dom_repeats = [r for r in repeat_runs if r['ep'][dom_idx] > 1.0]
    print(f"Domain {dom_name:8s}: {len(dom_repeats)} repeat runs")

