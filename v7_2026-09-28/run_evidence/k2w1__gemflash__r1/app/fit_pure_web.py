import numpy as np

# Pure web data points:
# (N, D, gen, code, math)
# (5e7, 1e9): gen=3.3582, code=3.4690, math=4.3826
# (1e8, 1e9): gen=3.1965, code=3.3667, math=4.2395
# (2e8, 1e9): gen=3.0682, code=3.2728, math=4.1260
# (4e8, 1e9): gen=2.9465, code=3.2126, math=4.0344
# (5e7, 2e9): gen=3.2146, code=3.0898, math=3.8008
# (5e7, 4e9): gen=3.0859, code=2.7703, math=3.3306
# (5e7, 8e9): gen=2.9823, code=2.5103, math=2.9555
# (1e8, 2e9): gen=3.0460, code=2.9886, math=3.6537

data = [
    (5e7, 1e9, 3.3582, 3.4690, 4.3826),
    (1e8, 1e9, 3.1965, 3.3667, 4.2395),
    (2e8, 1e9, 3.0682, 3.2728, 4.1260),
    (4e8, 1e9, 2.9465, 3.2126, 4.0344),
    (5e7, 2e9, 3.2146, 3.0898, 3.8008),
    (5e7, 4e9, 3.0859, 2.7703, 3.3306),
    (5e7, 8e9, 2.9823, 2.5103, 2.9555),
    (1e8, 2e9, 3.0460, 2.9886, 3.6537),
]

N = np.array([d[0] for d in data])
D = np.array([d[1] for d in data])

for col, name in [(2, 'general'), (3, 'code'), (4, 'math')]:
    y = np.array([d[col] for d in data])
    best_err = 1e9
    best_res = None
    
    for alpha in np.linspace(0.15, 0.55, 81):
        for beta in np.linspace(0.15, 0.55, 81):
            X = np.column_stack([np.ones_like(y), (N / 1e8)**(-alpha), (D / 1e9)**(-beta)])
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if params[0] < 0 or params[1] < 0 or params[2] < 0: continue
            pred = X @ params
            err = np.max(np.abs(pred - y))
            if err < best_err:
                best_err = err
                best_res = (params, alpha, beta)
                
    params, alpha, beta = best_res
    E, A, B_web = params
    print(f"\nPure Web {name}:")
    print(f"E={E:.4f}, A={A:.4f}, B_web={B_web:.4f}, alpha={alpha:.4f}, beta={beta:.4f}, max_err={best_err:.5f}")
    X = np.column_stack([np.ones_like(y), (N / 1e8)**(-alpha), (D / 1e9)**(-beta)])
    pred = X @ params
    for i, d in enumerate(data):
        print(f"N={d[0]:.1e}, D={d[1]:.1e}: actual={y[i]:.4f}, pred={pred[i]:.4f}, err={y[i]-pred[i]:+.4f}")
