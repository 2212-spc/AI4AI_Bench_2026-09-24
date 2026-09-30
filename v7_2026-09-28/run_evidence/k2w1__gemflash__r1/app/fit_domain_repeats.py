import json
import numpy as np

history = json.load(open('/app/full_history.json'))

model = {
    'general': {
        'E': 1.3658, 'A': 0.9445, 'alpha': 0.2250, 'beta': 0.2650, 'B_prim': 0.8896,
        'c': np.array([1.0000, 0.1914, 0.1030, 0.7812]) # [web, code, math, papers]
    },
    'code': {
        'E': 0.5648, 'A': 0.3888, 'alpha': 0.3500, 'beta': 0.2450, 'B_prim': 1.1654,
        'c': np.array([0.0515, 1.0000, 0.0869, 0.0433]) # [web, code, math, papers]
    },
    'math': {
        'E': 0.7367, 'A': 0.5283, 'alpha': 0.3500, 'beta': 0.3150, 'B_prim': 1.1738,
        'c': np.array([0.0524, 0.2359, 1.0000, 0.3444]) # [web, code, math, papers]
    }
}

# Math repeat runs (12 runs)
math_repeats = []
for h in history:
    args = h['args']
    m = args['mix']
    p = args.get('pool', {})
    d_m = m.get('math', 0.0) * args['D']
    u_m = p.get('math', 4e9)
    if d_m > u_m:
        math_repeats.append(h)

# Let's test functions for math: D_eff = U * f(ep)
# Candidate 1: f(ep) = 1 + c * (ep - 1)^gamma
# Candidate 2: f(ep) = 1 + alpha * (1 - ep^(-beta))
# Candidate 3: f(ep) = 1 + delta * log(ep)
# Candidate 4: f(ep) = 1 + c * (ep - 1) / (1 + d * (ep - 1))

candidates = {}

# Test Candidate 1:
best_err1 = 1e9
best_p1 = None
for c in np.linspace(0.4, 1.2, 81):
    for gamma in np.linspace(0.1, 0.8, 71):
        errs = []
        for h in math_repeats:
            args = h['args']
            m = args['mix']
            p = args.get('pool', {})
            ep = (m.get('math', 0.0) * args['D']) / p.get('math', 4e9)
            d_eff_m = p.get('math', 4e9) * (1.0 + c * (ep - 1.0)**gamma)
            
            # effective vector D_eff:
            d_eff_vec = np.array([
                m.get('web', 0.0) * args['D'],
                m.get('code', 0.0) * args['D'],
                d_eff_m,
                m.get('papers', 0.0) * args['D']
            ]) / 1e9
            
            mod = model['math']
            s = np.dot(d_eff_vec, mod['c'])
            pred = mod['E'] + mod['A'] * (args['N'] / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
            act = h['result']['eval_loss']['math']
            errs.append(abs(pred - act))
        max_err = max(errs)
        if max_err < best_err1:
            best_err1 = max_err
            best_p1 = (c, gamma)

print(f"Candidate 1 (power law): c={best_p1[0]:.3f}, gamma={best_p1[1]:.3f}, max_err={best_err1:.5f}")

# Test Candidate 2:
best_err2 = 1e9
best_p2 = None
for alpha in np.linspace(0.6, 1.5, 91):
    for beta in np.linspace(0.5, 3.0, 51):
        errs = []
        for h in math_repeats:
            args = h['args']
            m = args['mix']
            p = args.get('pool', {})
            ep = (m.get('math', 0.0) * args['D']) / p.get('math', 4e9)
            d_eff_m = p.get('math', 4e9) * (1.0 + alpha * (1.0 - ep**(-beta)))
            
            d_eff_vec = np.array([
                m.get('web', 0.0) * args['D'],
                m.get('code', 0.0) * args['D'],
                d_eff_m,
                m.get('papers', 0.0) * args['D']
            ]) / 1e9
            
            mod = model['math']
            s = np.dot(d_eff_vec, mod['c'])
            pred = mod['E'] + mod['A'] * (args['N'] / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
            act = h['result']['eval_loss']['math']
            errs.append(abs(pred - act))
        max_err = max(errs)
        if max_err < best_err2:
            best_err2 = max_err
            best_p2 = (alpha, beta)

print(f"Candidate 2 (power saturation): alpha={best_p2[0]:.3f}, beta={best_p2[1]:.3f}, max_err={best_err2:.5f}")

# Test Candidate 4 (rational / Langmuir):
best_err4 = 1e9
best_p4 = None
for c in np.linspace(0.5, 3.0, 101):
    for d in np.linspace(0.5, 3.0, 101):
        errs = []
        for h in math_repeats:
            args = h['args']
            m = args['mix']
            p = args.get('pool', {})
            ep = (m.get('math', 0.0) * args['D']) / p.get('math', 4e9)
            d_eff_m = p.get('math', 4e9) * (1.0 + c * (ep - 1.0) / (1.0 + d * (ep - 1.0)))
            
            d_eff_vec = np.array([
                m.get('web', 0.0) * args['D'],
                m.get('code', 0.0) * args['D'],
                d_eff_m,
                m.get('papers', 0.0) * args['D']
            ]) / 1e9
            
            mod = model['math']
            s = np.dot(d_eff_vec, mod['c'])
            pred = mod['E'] + mod['A'] * (args['N'] / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
            act = h['result']['eval_loss']['math']
            errs.append(abs(pred - act))
        max_err = max(errs)
        if max_err < best_err4:
            best_err4 = max_err
            best_p4 = (c, d)

print(f"Candidate 4 (rational): c={best_p4[0]:.3f}, d={best_p4[1]:.3f}, max_err={best_err4:.5f}")

