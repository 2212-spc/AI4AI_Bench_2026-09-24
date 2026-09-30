import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# Pure web data points:
data_web = [
    (5e7, 1e9, 3.3582, 3.4690, 4.3826),
    (1e8, 1e9, 3.1965, 3.3667, 4.2395),
    (2e8, 1e9, 3.0682, 3.2728, 4.1260),
    (4e8, 1e9, 2.9465, 3.2126, 4.0344),
    (5e7, 2e9, 3.2146, 3.0898, 3.8008),
    (5e7, 4e9, 3.0859, 2.7703, 3.3306),
    (5e7, 8e9, 2.9823, 2.5103, 2.9555),
    (1e8, 2e9, 3.0460, 2.9886, 3.6537),
]

# For each eval set, find ALL parameter sets that fit pure web data with max_err < 0.007
# (which is within simulation noise ptp = 0.006 - 0.009)
N_vals = np.array([d[0] for d in data_web])
D_vals = np.array([d[1] for d in data_web])

N_target = 2.5e9
D_target = 500e9

# Effective tokens at target for the chosen mix:
# web: 0.57, code: 0.32, math: 0.02, papers: 0.09
# d_eff_vec ~ [285, 108, 6.9, 38] in billions of tokens
# General: s_gen = 285 + 0.1914*108 + 0.103*6.9 + 0.7812*38 = 336.1B -> d_eff_gen = 336.1
# Code: s_code = 0.0515*285 + 108 + 0.0869*6.9 + 0.0433*38 = 124.9B -> d_eff_code = 124.9
# Math: s_math = 0.0524*285 + 0.2359*108 + 6.9 + 0.3444*38 = 60.5B -> d_eff_math = 60.5

d_eff_target = {
    'general': 336.1,
    'code': 124.9,
    'math': 60.5
}

B_ratio = {
    'general': 1.0, # primary is web
    'code': 1.1654 / 2.4080, # primary is code, so B_prim / B_web
    'math': 1.1738 / 2.9734  # primary is math, so B_prim / B_web
}

preds = []

for trial in range(100):
    pass

for col, name in [(2, 'general'), (3, 'code'), (4, 'math')]:
    y_vals = np.array([d[col] for d in data_web])
    target_preds = []
    
    for alpha in np.linspace(0.18, 0.45, 55):
        for beta in np.linspace(0.18, 0.45, 55):
            X = np.column_stack([np.ones_like(y_vals), (N_vals / 1e8)**(-alpha), (D_vals / 1e9)**(-beta)])
            params, _, _, _ = np.linalg.lstsq(X, y_vals, rcond=None)
            if params[0] < 0 or params[1] < 0 or params[2] < 0: continue
            pred = X @ params
            err = np.max(np.abs(pred - y_vals))
            if err < 0.006:
                # This is an acceptable parameter set!
                E, A, B_web = params
                B_prim = B_web * B_ratio[name]
                pred_target = E + A * (N_target / 1e8)**(-alpha) + B_prim * (d_eff_target[name])**(-beta)
                target_preds.append((err, pred_target, E, A, B_web, alpha, beta))
                
    target_preds.sort()
    vals = [p[1] for p in target_preds]
    print(f"\nEval {name}: {len(target_preds)} fits with max_err < 0.006")
    print(f"  Target loss range: min={np.min(vals):.4f}, mean={np.mean(vals):.4f}, max={np.max(vals):.4f}, std={np.std(vals):.4f}")
    best = target_preds[0]
    print(f"  Best fit (err={best[0]:.5f}): pred={best[1]:.4f}, E={best[2]:.4f}, A={best[3]:.4f}, B={best[4]:.4f}, alpha={best[5]:.4f}, beta={best[6]:.4f}")

