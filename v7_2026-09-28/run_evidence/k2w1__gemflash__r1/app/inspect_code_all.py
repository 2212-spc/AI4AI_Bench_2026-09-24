import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# Let's inspect code loss across all runs
runs_code = []
for h in history:
    args = h['args']
    m = args['mix']
    p = args.get('pool', {})
    runs_code.append((h['i'], args['N'], args['D'], m, p, h['result']['eval_loss']['code']))

for r in runs_code:
    if r[3].get('code', 0) > 0 or r[1] > 5e7 or r[2] > 1e9:
        print(f"run {r[0]:2d}: N={r[1]:.1e} D={r[2]:.1e} mix={r[3]} pool={r[4]} -> code={r[5]:.4f}")

