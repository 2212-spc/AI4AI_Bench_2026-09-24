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

# For Code repeats: runs 46, 47, 48 (pool code: 1e8 (ep=2), 5e7 (ep=4), 2e7 (ep=10))
# Let's fit alpha_code, beta_code for f(ep) = 1 + alpha * (1 - ep^(-beta)):
code_repeats = [history[46], history[47], history[48]]
best_err_c = 1e9
best_p_c = None
for alpha in np.linspace(0.8, 5.0, 421):
    for beta in np.linspace(0.1, 2.0, 191):
        errs = []
        for h in code_repeats:
            args = h['args']
            m = args['mix']
            p = args['pool']
            ep = (m['code'] * args['D']) / p['code']
            d_eff_c = p['code'] * (1.0 + alpha * (1.0 - ep**(-beta)))
            
            d_eff_vec = np.array([
                m['web'] * args['D'],
                d_eff_c,
                0.0,
                0.0
            ]) / 1e9
            
            mod = model['code']
            s = np.dot(d_eff_vec, mod['c'])
            pred = mod['E'] + mod['A'] * (args['N'] / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
            act = h['result']['eval_loss']['code']
            errs.append(abs(pred - act))
        if max(errs) < best_err_c:
            best_err_c = max(errs)
            best_p_c = (alpha, beta)

print(f"Code repeats (saturation): alpha={best_p_c[0]:.3f}, beta={best_p_c[1]:.3f}, max_err={best_err_c:.5f}")

# For Papers repeats: runs 50, 51, 52 (pool papers: 1e8 (ep=2), 5e7 (ep=4), 2e7 (ep=10))
papers_repeats = [history[50], history[51], history[52]]
best_err_p = 1e9
best_p_p = None
for alpha in np.linspace(0.8, 5.0, 421):
    for beta in np.linspace(0.1, 2.0, 191):
        errs = []
        for h in papers_repeats:
            args = h['args']
            m = args['mix']
            p = args['pool']
            ep = (m['papers'] * args['D']) / p['papers']
            d_eff_p = p['papers'] * (1.0 + alpha * (1.0 - ep**(-beta)))
            
            d_eff_vec = np.array([
                m['web'] * args['D'],
                0.0,
                0.0,
                d_eff_p
            ]) / 1e9
            
            mod = model['math']
            s = np.dot(d_eff_vec, mod['c'])
            pred = mod['E'] + mod['A'] * (args['N'] / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
            act = h['result']['eval_loss']['math']
            errs.append(abs(pred - act))
        if max(errs) < best_err_p:
            best_err_p = max(errs)
            best_p_p = (alpha, beta)

print(f"Papers repeats (saturation): alpha={best_p_p[0]:.3f}, beta={best_p_p[1]:.3f}, max_err={best_err_p:.5f}")

