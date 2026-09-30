import numpy as np

# Load simulation model
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

repeat_params = {
    'web':    (0.0, 1.0),
    'code':   (4.960, 0.300),
    'math':   (1.010, 1.450),
    'papers': (2.330, 0.600)
}

U = {
    'web': 3e12,
    'code': 4e10,
    'math': 4e9,
    'papers': 2e10
}

N_target = 2.5e9
D_target = 500e9

def get_effective_tokens(d_seen, u, dom):
    if d_seen <= u:
        return d_seen
    ep = d_seen / u
    alpha, beta = repeat_params[dom]
    return u * (1.0 + alpha * (1.0 - ep**(-beta)))

def evaluate_mix(r_w, r_c, r_m, r_p):
    mix = {'web': r_w, 'code': r_c, 'math': r_m, 'papers': r_p}
    doms = ['web', 'code', 'math', 'papers']
    d_eff = []
    for dom in doms:
        d_seen = mix[dom] * D_target
        u = U[dom]
        d_eff.append(get_effective_tokens(d_seen, u, dom))
    d_eff = np.array(d_eff) / 1e9
    
    losses = {}
    for ev in ['general', 'code', 'math']:
        mod = model[ev]
        s = np.dot(d_eff, mod['c'])
        losses[ev] = mod['E'] + mod['A'] * (N_target / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
    
    comp = 0.34 * losses['general'] + 0.33 * losses['code'] + 0.33 * losses['math']
    return losses, comp

# Grid search over 4-simplex with fine resolution
best_comp = 1e9
best_mix = None
best_losses = None

step = 0.005
steps = int(round(1.0 / step))

# Let's search over (r_w, r_c, r_m, r_p)
grid_results = []

for i_c in range(steps + 1):
    r_c = i_c * step
    for i_m in range(steps + 1 - i_c):
        r_m = i_m * step
        for i_p in range(steps + 1 - i_c - i_m):
            r_p = i_p * step
            r_w = round(1.0 - r_c - r_m - r_p, 4)
            if r_w < -1e-6: continue
            r_w = max(0.0, r_w)
            
            losses, comp = evaluate_mix(r_w, r_c, r_m, r_p)
            if comp < best_comp:
                best_comp = comp
                best_mix = (r_w, r_c, r_m, r_p)
                best_losses = losses

print("Optimization complete!")
r_w, r_c, r_m, r_p = best_mix
print(f"Optimal mixture:")
print(f"  web:    {r_w:.4f} ({r_w*D_target/1e9:.1f}B tokens, {r_w*D_target/U['web']:.2f} epochs)")
print(f"  code:   {r_c:.4f} ({r_c*D_target/1e9:.1f}B tokens, {r_c*D_target/U['code']:.2f} epochs)")
print(f"  math:   {r_m:.4f} ({r_m*D_target/1e9:.1f}B tokens, {r_m*D_target/U['math']:.2f} epochs)")
print(f"  papers: {r_p:.4f} ({r_p*D_target/1e9:.1f}B tokens, {r_p*D_target/U['papers']:.2f} epochs)")
print(f"Predicted losses:")
print(f"  general:   {best_losses['general']:.5f}")
print(f"  code:      {best_losses['code']:.5f}")
print(f"  math:      {best_losses['math']:.5f}")
print(f"  composite: {best_comp:.5f}")

