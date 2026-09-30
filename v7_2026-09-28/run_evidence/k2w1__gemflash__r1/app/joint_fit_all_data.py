import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# We have 51 valid runs with NO repeats (all domains epochs <= 1).
# These runs span:
# N in [5e7, 1e8, 2e8, 4e8]
# D in [1e9, 2e9, 4e9, 8e9]
# mixtures:
# - pure web at multiple N and D
# - pure code at multiple N
# - pure math at multiple N
# - pure papers at multiple N
# - sweeps of web+math at multiple ratios
# - sweeps of web+papers at multiple ratios
# - web 0.8 + code 0.2
# - 4-domain mixes at N=5e7, D=1e9 and N=4e8, D=2e9!

# In all these 51 runs:
# d_eff_vec = D * [r_web, r_code, r_math, r_papers]
# For each eval set e:
# L_e = E_e + A_e * (N/1e8)^(-alpha_e) + B_prim,e * (s_e * D / 1e9)^(-beta_e)
# where s_e = sum_d c_{e, d} * r_d.
# Let's fit (E, A, alpha, beta, B_prim, c_web, c_code, c_math, c_papers)
# on ALL 51 runs simultaneously!

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
        valid_runs.append({
            'N': args['N'],
            'D': args['D'],
            'r': np.array([m.get('web', 0.0), m.get('code', 0.0), m.get('math', 0.0), m.get('papers', 0.0)]),
            'losses': res['eval_loss'],
            'comp': res['composite']
        })

print(f"Total valid runs: {len(valid_runs)}")

for ev_name in ['general', 'code', 'math']:
    y = np.array([r['losses'][ev_name] for r in valid_runs])
    N = np.array([r['N'] for r in valid_runs])
    D = np.array([r['D'] for r in valid_runs])
    R = np.array([r['r'] for r in valid_runs])
    
    # We want to find the parameter set that minimizes max abs error or RMS error across ALL 51 runs.
    # Notice: when we fit on all 51 runs, the parameters are MUCH more constrained than on just pure web!
    best_rms = 1e9
    best_params = None
    
    # Grid search for (alpha, beta)
    # The primary domain index:
    prim_idx = {'general': 0, 'code': 1, 'math': 2}[ev_name]
    
    for alpha in np.linspace(0.20, 0.45, 26):
        for beta in np.linspace(0.20, 0.45, 26):
            # For fixed alpha and beta, can we find E, A, B and transfer c?
            # From earlier:
            # Transfer c is given by c_d = (B_prim / B_d)^(1 / beta)
            # where B_d is the loss difference at N=5e7, D=1e9!
            # Since L_d(5e7, 1e9) = E + A * (0.5)^(-alpha) + B_d!
            # So B_d = L_d(5e7, 1e9) - E - A * 2^alpha !
            # That means B_d is directly determined by E and A!
            pass

print("Testing optimization...")
