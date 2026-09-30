import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# For Code:
# Best fit parameters from fit_all_evals.py:
# L_inf = 1.7794, beta = 0.5800, B = 0.4478
# Transfer: c_web = 0.1013, c_code = 1.0000, c_math = 0.1394, c_papers = 0.0921
# In runs 45 to 48: N=5e7, D=1e9, mix: web 0.8, code 0.2
# run 45: pool=2e8 (ep=1.0) -> code loss = 2.70862
# run 46: pool=1e8 (ep=2.0) -> code loss = 2.73740
# run 47: pool=5e7 (ep=4.0) -> code loss = 2.83104
# run 48: pool=2e7 (ep=10.0)-> code loss = 3.07038

L_inf_c = 1.7794
beta_c = 0.5800
B_c = 0.4478
c_w_c = 0.1013

for r_id, ep, u in [(45, 1.0, 2e8), (46, 2.0, 1e8), (47, 4.0, 5e7), (48, 10.0, 2e7)]:
    loss = history[r_id]['result']['eval_loss']['code']
    s_c = ((loss - L_inf_c) / B_c)**(-1.0 / beta_c)
    d_eff_c = 1e9 * (s_c - c_w_c * 0.8)
    ratio = d_eff_c / u
    print(f"Code pool ep={ep:4.1f}: loss={loss:.4f}, S_c={s_c:.4f}, D_eff={d_eff_c:.2e}, D_eff/U={ratio:.3f}")

# For Papers:
# In runs 49 to 52: N=5e7, D=1e9, mix: web 0.8, papers 0.2
# Papers contributes primarily to General and Math.
# On Math eval:
# run 49: ep=1.0, loss=3.76022
# run 50: ep=2.0, loss=3.81786
# run 51: ep=4.0, loss=3.98183
# run 52: ep=10.0, loss=4.25636

L_inf_m = 1.3948
beta_m = 0.3100
B_m = 1.1860
c_w_m = 0.0506
c_p_m = 0.3389

print("\nPapers pool evaluated on Math loss:")
for r_id, ep, u in [(49, 1.0, 2e8), (50, 2.0, 1e8), (51, 4.0, 5e7), (52, 10.0, 2e7)]:
    loss = history[r_id]['result']['eval_loss']['math']
    s_m = ((loss - L_inf_m) / B_m)**(-1.0 / beta_m)
    # S_m = c_w_m * 0.8 + c_p_m * (D_eff_p / 1e9)
    d_eff_p = 1e9 * (s_m - c_w_m * 0.8) / c_p_m
    ratio = d_eff_p / u
    print(f"Papers pool ep={ep:4.1f}: loss={loss:.4f}, S_m={s_m:.4f}, D_eff={d_eff_p:.2e}, D_eff/U={ratio:.3f}")

