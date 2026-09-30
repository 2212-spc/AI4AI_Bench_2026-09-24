import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# Inspect code eval errors across valid runs
valid_runs = []
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
    if d_w <= u_w and d_c <= u_c and d_m <= u_m and d_p <= u_p:
        valid_runs.append(h)

c_code = np.array([0.1013, 1.0000, 0.1394, 0.0921])
# Check code loss on pure web vs D
print("Pure web code loss vs D (N=5e7):")
for h in valid_runs:
    if h['args']['N'] == 5e7 and h['args']['mix'].get('web', 0) == 1.0:
        print(f"D={h['args']['D']:.1e}: code_loss={h['result']['eval_loss']['code']:.4f}")

print("\nPure web code loss vs N (D=1e9):")
for h in valid_runs:
    if h['args']['D'] == 1e9 and h['args']['mix'].get('web', 0) == 1.0:
        print(f"N={h['args']['N']:.1e}: code_loss={h['result']['eval_loss']['code']:.4f}")

