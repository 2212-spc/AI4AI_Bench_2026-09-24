import json
import numpy as np

history = json.load(open('/app/full_history.json'))

runs = []
for h in history:
    if h['args']['N'] == 5e7 and h['args']['D'] == 1e9:
        args = h['args']
        m = args['mix']
        p = args.get('pool', {})
        runs.append({
            'r_w': m.get('web', 0.0),
            'r_c': m.get('code', 0.0),
            'r_m': m.get('math', 0.0),
            'r_p': m.get('papers', 0.0),
            'u_w': p.get('web', 3e12),
            'u_c': p.get('code', 4e10),
            'u_m': p.get('math', 4e9),
            'u_p': p.get('papers', 2e10),
            'loss_gen': h['result']['eval_loss']['general'],
            'loss_code': h['result']['eval_loss']['code'],
            'loss_math': h['result']['eval_loss']['math']
        })

no_pool_runs = [r for r in runs if r['r_m']*1e9 <= r['u_m'] and r['r_c']*1e9 <= r['u_c'] and r['r_p']*1e9 <= r['u_p']]
R = np.array([[r['r_w'], r['r_c'], r['r_m'], r['r_p']] for r in no_pool_runs])

for eval_name in ['general', 'code', 'math']:
    y_losses = np.array([r['loss_' + (eval_name if eval_name != 'general' else 'gen')] for r in no_pool_runs])
    
    best_err = 1e9
    best_params = None
    
    # Grid search over L_inf and beta
    for L_inf in np.linspace(0.5, np.min(y_losses) - 0.05, 60):
        for beta in np.linspace(0.15, 0.60, 46):
            y_trans = (y_losses - L_inf)**(-1.0 / beta)
            k, _, _, _ = np.linalg.lstsq(R, y_trans, rcond=None)
            if np.any(k <= 0): continue
            pred_loss = L_inf + (R @ k)**(-beta)
            err = np.max(np.abs(pred_loss - y_losses))
            if err < best_err:
                best_err = err
                best_params = (L_inf, beta, k)
                
    L_inf, beta, k = best_params
    print(f"\n=================== EVAL: {eval_name} ===================")
    print(f"Best fit: L_inf={L_inf:.4f}, beta={beta:.4f}, max_err={best_err:.5f}")
    # Normalize by the dominant domain for this eval set:
    dom_idx = {'general': 0, 'code': 1, 'math': 2}[eval_name]
    c = k / k[dom_idx]
    print(f"Transfer coefficients [web, code, math, papers] (normalized to {['web','code','math'][dom_idx]}=1):")
    print(f"c_web:    {c[0]:.4f}")
    print(f"c_code:   {c[1]:.4f}")
    print(f"c_math:   {c[2]:.4f}")
    print(f"c_papers: {c[3]:.4f}")
    B = k[dom_idx]**(-beta)
    print(f"B={B:.4f}")
    
    pred_loss = L_inf + (R @ k)**(-beta)
    resids = y_losses - pred_loss
    print(f"Mean abs resid: {np.mean(np.abs(resids)):.5f}, Std resid: {np.std(resids):.5f}")

