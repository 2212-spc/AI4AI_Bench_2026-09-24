import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# Filter runs at N=5e7, D=1e9
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
            'loss': h['result']['eval_loss']['math']
        })

print(f"Total runs for math: {len(runs)}")

# First, consider runs with NO pool (epochs <= 1 for all domains)
no_pool_runs = [r for r in runs if r['r_m']*1e9 <= r['u_m'] and r['r_c']*1e9 <= r['u_c'] and r['r_p']*1e9 <= r['u_p']]
print(f"No-pool runs: {len(no_pool_runs)}")

# In no-pool runs:
# S_math = r_m + c_w * r_w + c_c * r_c + c_p * r_p
# loss = L_inf + B * S_math^(-beta)
# Or (loss - L_inf)^(-1/beta) = B^(-1/beta) * (r_m + c_w * r_w + c_c * r_c + c_p * r_p)
# That's a linear combination!
# Let y = (loss - L_inf)^(-1/beta)
# y = k_m * r_m + k_w * r_w + k_c * r_c + k_p * r_p !
# Where k_d = B^(-1/beta) * c_d !
# If this is true, then for fixed L_inf and beta:
# y is an EXACT linear function of [r_w, r_c, r_m, r_p] through the origin!
# Let's test this!

best_err = 1e9
best_params = None

y_losses = np.array([r['loss'] for r in no_pool_runs])
R = np.array([[r['r_w'], r['r_c'], r['r_m'], r['r_p']] for r in no_pool_runs])

for L_inf in np.linspace(1.0, 2.4, 71):
    if L_inf >= np.min(y_losses): continue
    for beta in np.linspace(0.15, 0.60, 46):
        y_trans = (y_losses - L_inf)**(-1.0 / beta)
        # fit y_trans = R @ k
        k, res, _, _ = np.linalg.lstsq(R, y_trans, rcond=None)
        if np.any(k <= 0): continue
        # predict loss back
        pred_y_trans = R @ k
        pred_loss = L_inf + pred_y_trans**(-beta)
        err = np.max(np.abs(pred_loss - y_losses))
        if err < best_err:
            best_err = err
            best_params = (L_inf, beta, k)

L_inf, beta, k = best_params
print(f"Best fit: L_inf={L_inf:.4f}, beta={beta:.4f}, max_err={best_err:.5f}")
# Normalized transfer coefficients c_d = k_d / k_m:
c = k / k[2]
print(f"Transfer coefficients [web, code, math, papers]:")
print(f"c_web:    {c[0]:.4f}")
print(f"c_code:   {c[1]:.4f}")
print(f"c_math:   {c[2]:.4f}")
print(f"c_papers: {c[3]:.4f}")
B = k[2]**(-beta)
print(f"B={B:.4f}")

# Print predictions vs actual for no-pool runs
print("\nPredictions on no-pool runs:")
pred_loss = L_inf + (R @ k)**(-beta)
for i, r in enumerate(no_pool_runs):
    diff = r['loss'] - pred_loss[i]
    print(f"w={r['r_w']:.2f} c={r['r_c']:.2f} m={r['r_m']:.2f} p={r['r_p']:.2f} | actual={r['loss']:.4f} pred={pred_loss[i]:.4f} diff={diff:+.4f}")
