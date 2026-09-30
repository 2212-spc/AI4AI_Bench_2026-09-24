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

comp_diffs = []
for h in history:
    args = h['args']
    res = h['result']
    m = args['mix']
    p = args.get('pool', {})
    
    # Check if epochs <= 1
    d_w = m.get('web', 0.0) * args['D']
    d_c = m.get('code', 0.0) * args['D']
    d_m = m.get('math', 0.0) * args['D']
    d_p = m.get('papers', 0.0) * args['D']
    
    u_w = p.get('web', 3e12)
    u_c = p.get('code', 4e10)
    u_m = p.get('math', 4e9)
    u_p = p.get('papers', 2e10)
    
    if d_w <= u_w and d_c <= u_c and d_m <= u_m and d_p <= u_p:
        r = np.array([m.get('web', 0.0), m.get('code', 0.0), m.get('math', 0.0), m.get('papers', 0.0)])
        losses = {}
        for ev in ['general', 'code', 'math']:
            mod = model[ev]
            s = np.dot(r, mod['c'])
            d_eff = (args['D'] / 1e9) * s
            loss = mod['E'] + mod['A'] * (args['N'] / 1e8)**(-mod['alpha']) + mod['B_prim'] * (d_eff)**(-mod['beta'])
            losses[ev] = loss
        pred_comp = 0.34 * losses['general'] + 0.33 * losses['code'] + 0.33 * losses['math']
        act_comp = res['composite']
        comp_diffs.append(act_comp - pred_comp)

comp_diffs = np.array(comp_diffs)
print(f"Across all {len(comp_diffs)} valid runs without repetition:")
print(f"Max abs diff:  {np.max(np.abs(comp_diffs)):.5f}")
print(f"Mean abs diff: {np.mean(np.abs(comp_diffs)):.5f}")
print(f"Std diff:      {np.std(comp_diffs):.5f}")

