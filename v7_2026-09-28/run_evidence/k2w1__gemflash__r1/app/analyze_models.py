import json
import numpy as np

history = json.load(open('/app/full_history.json'))

dataset = []
for h in history:
    args = h['args']
    res = h['result']
    m = args['mix']
    p = args.get('pool', {})
    dataset.append({
        'N': args['N'],
        'D': args['D'],
        'r_web': m.get('web', 0.0),
        'r_code': m.get('code', 0.0),
        'r_math': m.get('math', 0.0),
        'r_papers': m.get('papers', 0.0),
        'u_web': p.get('web', 3e12),
        'u_code': p.get('code', 4e10),
        'u_math': p.get('math', 4e9),
        'u_papers': p.get('papers', 2e10),
        'loss_gen': res['eval_loss']['general'],
        'loss_code': res['eval_loss']['code'],
        'loss_math': res['eval_loss']['math'],
        'composite': res['composite']
    })

print(f"Total dataset size: {len(dataset)}")

# Let's inspect the math eval loss specifically across all runs where D=1e9, N=5e7:
sub = [d for d in dataset if d['N'] == 5e7 and d['D'] == 1e9]
print(f"Number of runs at N=5e7, D=1e9: {len(sub)}")

# Let's see: for math eval loss, can we find an empirical formula that predicts ALL math eval losses?
# Let's print out the data points:
for d in sub:
    ep_m = (d['r_math'] * d['D']) / d['u_math']
    ep_c = (d['r_code'] * d['D']) / d['u_code']
    ep_p = (d['r_papers'] * d['D']) / d['u_papers']
    print(f"web={d['r_web']:.2f} cod={d['r_code']:.2f} mat={d['r_math']:.2f} pap={d['r_papers']:.2f} | ep_m={ep_m:.2f} ep_c={ep_c:.2f} ep_p={ep_p:.2f} | gen={d['loss_gen']:.4f} cod={d['loss_code']:.4f} mat={d['loss_math']:.4f}")

