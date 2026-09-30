import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# We have:
# Runs at N in [5e7, 1e8, 2e8, 4e8]
# D in [1e9, 2e9, 4e9, 8e9]
# And mixtures at N=4e8, D=2e9 (runs 58-61) and N=1e8, D=2e9 (run 68).

# Let's inspect all runs where epochs <= 1 for all domains (or no pool):
valid_runs = []
for h in history:
    args = h['args']
    res = h['result']
    m = args['mix']
    p = args.get('pool', {})
    
    # Check epochs
    d_w = m.get('web', 0.0) * args['D']
    d_c = m.get('code', 0.0) * args['D']
    d_m = m.get('math', 0.0) * args['D']
    d_p = m.get('papers', 0.0) * args['D']
    
    u_w = p.get('web', 3e12)
    u_c = p.get('code', 4e10)
    u_m = p.get('math', 4e9)
    u_p = p.get('papers', 2e10)
    
    if d_w <= u_w and d_c <= u_c and d_m <= u_m and d_p <= u_p:
        valid_runs.append({
            'N': args['N'],
            'D': args['D'],
            'r_w': m.get('web', 0.0),
            'r_c': m.get('code', 0.0),
            'r_m': m.get('math', 0.0),
            'r_p': m.get('papers', 0.0),
            'gen': res['eval_loss']['general'],
            'code': res['eval_loss']['code'],
            'math': res['eval_loss']['math'],
            'comp': res['composite']
        })

print(f"Total valid runs with no repetition: {len(valid_runs)}")

# Let's check the diversity of (N, D) in valid_runs:
nd_pairs = sorted(list(set((r['N'], r['D']) for r in valid_runs)))
for n, d in nd_pairs:
    count = len([r for r in valid_runs if r['N'] == n and r['D'] == d])
    print(f"N={n:.1e}, D={d:.1e}: {count} runs")

