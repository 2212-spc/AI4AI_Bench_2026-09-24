import numpy as np
from optimize_target import model, N_target, D_target, U

# Calculate composite loss under both repeat models
r_w, r_c, r_m, r_p = 0.57, 0.32, 0.02, 0.09
mix = {'web': r_w, 'code': r_c, 'math': r_m, 'papers': r_p}

# Model 1: Saturation model
d_eff_sat = np.array([
    r_w * D_target,
    U['code'] * (1.0 + 4.96 * (1.0 - 4.0**(-0.30))),
    U['math'] * (1.0 + 1.01 * (1.0 - 2.5**(-1.45))),
    U['papers'] * (1.0 + 2.33 * (1.0 - 2.25**(-0.60)))
]) / 1e9

# Model 2: Power law / empirical interpolation
d_eff_pow = np.array([
    r_w * D_target,
    U['code'] * (1.0 + 0.938 * (4.0 - 1.0)**0.558),
    U['math'] * 1.728,
    U['papers'] * (1.0 + 0.807 * (2.25 - 1.0)**0.356)
]) / 1e9

# Model 3: Exact empirical table
d_eff_emp = np.array([
    r_w * D_target,
    U['code'] * 2.97,
    U['math'] * 1.728,
    U['papers'] * 1.88
]) / 1e9

def compute_loss(d_eff_vec):
    losses = {}
    for ev in ['general', 'code', 'math']:
        mod = model[ev]
        s = np.dot(d_eff_vec, mod['c'])
        losses[ev] = mod['E'] + mod['A'] * (N_target / 1e8)**(-mod['alpha']) + mod['B_prim'] * s**(-mod['beta'])
    comp = 0.34 * losses['general'] + 0.33 * losses['code'] + 0.33 * losses['math']
    return losses, comp

print("Model 1 (saturation):")
l1, c1 = compute_loss(d_eff_sat)
print(f"  gen={l1['general']:.5f}, code={l1['code']:.5f}, math={l1['math']:.5f} -> comp={c1:.5f}")

print("\nModel 2 (power law):")
l2, c2 = compute_loss(d_eff_pow)
print(f"  gen={l2['general']:.5f}, code={l2['code']:.5f}, math={l2['math']:.5f} -> comp={c2:.5f}")

print("\nModel 3 (exact empirical):")
l3, c3 = compute_loss(d_eff_emp)
print(f"  gen={l3['general']:.5f}, code={l3['code']:.5f}, math={l3['math']:.5f} -> comp={c3:.5f}")

