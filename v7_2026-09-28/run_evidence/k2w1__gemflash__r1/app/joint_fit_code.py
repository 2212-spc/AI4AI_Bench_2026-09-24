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
            'code': res['eval_loss']['code']
        })

y = np.array([r['code'] for r in valid_runs])
N = np.array([r['N'] for r in valid_runs])
D = np.array([r['D'] for r in valid_runs])
R = np.array([[r['r_w'], r['r_c'], r['r_m'], r['r_p']] for r in valid_runs])

# Joint optimization of (E, A, B, alpha, beta, c_web, c_math, c_papers)
# Let's use scipy.optimize.minimize if available, or a simple gradient descent / coordinate descent in numpy!
# Let's implement coordinate descent / Nelder-Mead in pure numpy!

def loss_func(params):
    # params: [E, A, B, alpha, beta, c_w, c_m, c_p]
    E, A, B, alpha, beta, c_w, c_m, c_p = params
    if E < 0 or A < 0 or B < 0 or alpha <= 0 or beta <= 0: return 1e9
    if c_w < 0 or c_m < 0 or c_p < 0: return 1e9
    c = np.array([c_w, 1.0, c_m, c_p])
    s = R @ c
    D_eff = (D / 1e9) * s
    pred = E + A * (N / 1e8)**(-alpha) + B * D_eff**(-beta)
    return np.mean((pred - y)**2)

# Start from a reasonable point:
# At D_eff = 1 (pure code at 1e9):
# N=5e7: 2.2274
# N=1e8: 2.1226
# N=2e8: 2.0304
# Notice: 2.2274 - 2.1226 = 0.1048. 2.1226 - 2.0304 = 0.0922.
# Slope in N is ~ 0.10 / log(2) ~ 0.14.
# At N=1e8, A*(N/1e8)^(-alpha) ~ A.
# Pure web at D=1e9 (N=5e7): 3.47. Pure code: 2.23.
# Difference is 1.24!
# Let's grid search the best starting point:

best_mse = 1e9
best_p = None

for alpha in np.linspace(0.2, 0.45, 11):
    for beta in np.linspace(0.25, 0.55, 13):
        for c_w in [0.03, 0.05, 0.08, 0.10]:
            for c_m in [0.1, 0.15, 0.2]:
                for c_p in [0.05, 0.1, 0.15]:
                    c = np.array([c_w, 1.0, c_m, c_p])
                    s = R @ c
                    D_eff = (D / 1e9) * s
                    X = np.column_stack([np.ones_like(y), (N / 1e8)**(-alpha), D_eff**(-beta)])
                    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    if coef[0] < 0 or coef[1] < 0 or coef[2] < 0: continue
                    pred = X @ coef
                    mse = np.mean((pred - y)**2)
                    if mse < best_mse:
                        best_mse = mse
                        best_p = (coef[0], coef[1], coef[2], alpha, beta, c_w, c_m, c_p)

print("Grid search best MSE:", best_mse, "RMS:", np.sqrt(best_mse))
print("Params [E, A, B, alpha, beta, c_w, c_m, c_p]:", np.round(best_p, 4))

# Local Nelder-Mead or coordinate search around best_p:
p = list(best_p)
step = [0.05, 0.05, 0.05, 0.02, 0.02, 0.005, 0.01, 0.01]

for it in range(200):
    for idx in range(len(p)):
        for direction in [-1, 1]:
            cand = list(p)
            cand[idx] += direction * step[idx]
            if loss_func(cand) < loss_func(p):
                p = cand

print("After local optimization:")
print("MSE:", loss_func(p), "RMS:", np.sqrt(loss_func(p)))
E, A, B, alpha, beta, c_w, c_m, c_p = p
print(f"E={E:.4f}, A={A:.4f}, B={B:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")
print(f"c_web={c_w:.4f}, c_math={c_m:.4f}, c_papers={c_p:.4f}")

# Check predictions across all 51 runs
c = np.array([c_w, 1.0, c_m, c_p])
s = R @ c
D_eff = (D / 1e9) * s
pred = E + A * (N / 1e8)**(-alpha) + B * D_eff**(-beta)
errs = np.abs(pred - y)
print(f"Max abs error: {np.max(errs):.5f}, Mean abs error: {np.mean(errs):.5f}")

# Print on big runs:
print("\nBig runs (N=4e8, D=2e9):")
for i, r in enumerate(valid_runs):
    if r['N'] == 4e8 and r['D'] == 2e9:
        print(f"w={r['r_w']:.2f} c={r['r_c']:.2f} m={r['r_m']:.2f} p={r['r_p']:.2f} | actual={y[i]:.4f} pred={pred[i]:.4f} diff={y[i]-pred[i]:+.4f}")

