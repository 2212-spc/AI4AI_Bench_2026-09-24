import json
import numpy as np

history = json.load(open('/app/full_history.json'))

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
            'r_w': m.get('web', 0.0),
            'r_c': m.get('code', 0.0),
            'r_m': m.get('math', 0.0),
            'r_p': m.get('papers', 0.0),
            'gen': res['eval_loss']['general'],
            'code': res['eval_loss']['code'],
            'math': res['eval_loss']['math'],
            'comp': res['composite']
        })

# For each eval set e:
# Fixed transfer coefficients c_e:
# General: c = [1.0000, 0.2175, 0.1299, 0.7509]
# Code:    c = [0.1013, 1.0000, 0.1394, 0.0921]
# Math:    c = [0.0506, 0.2335, 1.0000, 0.3389]

transfer = {
    'general': np.array([1.0000, 0.2175, 0.1299, 0.7509]),
    'code':    np.array([0.1013, 1.0000, 0.1394, 0.0921]),
    'math':    np.array([0.0506, 0.2335, 1.0000, 0.3389])
}

# In each run:
# effective token fraction: s_e = sum_d c_{e, d} * r_d
# effective total tokens: D_eff,e = s_e * D
# Then L_e(N, D) = E + A * (N / 1e8)^(-alpha) + B * (D_eff,e / 1e9)^(-beta)

for eval_name in ['general', 'code', 'math']:
    c = transfer[eval_name]
    y = np.array([r[eval_name if eval_name != 'general' else 'gen'] for r in valid_runs])
    N = np.array([r['N'] for r in valid_runs])
    D = np.array([r['D'] for r in valid_runs])
    R = np.array([[r['r_w'], r['r_c'], r['r_m'], r['r_p']] for r in valid_runs])
    s = R @ c
    D_eff = D * s
    
    best_err = 1e9
    best_fit = None
    
    # Grid search over alpha and beta
    for alpha in np.linspace(0.15, 0.60, 46):
        for beta in np.linspace(0.15, 0.60, 46):
            # Regression: y = E + A * (N/1e8)^(-alpha) + B * (D_eff/1e9)^(-beta)
            X = np.column_stack([np.ones_like(y), (N / 1e8)**(-alpha), (D_eff / 1e9)**(-beta)])
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if params[0] < 0 or params[1] < 0 or params[2] < 0:
                continue
            pred = X @ params
            err = np.max(np.abs(pred - y))
            if err < best_err:
                best_err = err
                best_fit = (params, alpha, beta)
                
    params, alpha, beta = best_fit
    E, A, B = params
    print(f"\n=================== EVAL: {eval_name} ===================")
    print(f"E={E:.4f}, A={A:.4f}, B={B:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"Max abs error across ALL 51 valid runs: {best_err:.5f}")
    
    X = np.column_stack([np.ones_like(y), (N / 1e8)**(-alpha), (D_eff / 1e9)**(-beta)])
    pred = X @ params
    resids = y - pred
    print(f"Mean abs error: {np.mean(np.abs(resids)):.5f}, Std error: {np.std(resids):.5f}")
    
    # Let's inspect predictions on the big runs (N=4e8, D=2e9)
    print("Predictions on N=4e8, D=2e9 runs:")
    for i, r in enumerate(valid_runs):
        if r['N'] == 4e8 and r['D'] == 2e9:
            print(f"w={r['r_w']:.2f} c={r['r_c']:.2f} m={r['r_m']:.2f} p={r['r_p']:.2f} | actual={y[i]:.4f} pred={pred[i]:.4f} diff={resids[i]:+.4f}")

