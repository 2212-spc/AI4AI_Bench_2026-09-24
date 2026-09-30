import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# We know for Math eval at N=5e7, D=1e9:
# L_inf = 1.3948 (or near 1.40), beta = 0.31, B = 1.1860
# c = [c_web=0.0506, c_code=0.2335, c_math=1.0000, c_papers=0.3389]
# S_math = c_web * D_eff,w + c_code * D_eff,c + c_math * D_eff,m + c_papers * D_eff,p  (in units of 1e9)
# From loss, we can invert S_math:
# S_math = ((loss - L_inf) / B)**(-1 / beta)

# Let's inspect the math pool runs!
# In each math pool run, D=1e9, web=1-r_m, math=r_m, code=0, papers=0.
# So D_eff,w = (1 - r_m) * 1e9 (since web has 3T tokens, epochs << 1).
# Therefore:
# S_math = c_web * (1 - r_m) + c_math * (D_eff,m / 1e9)
# => D_eff,m = 1e9 * (S_math - c_web * (1 - r_m))
# And epochs = (r_m * 1e9) / U_m !
# Ratio D_eff,m / U_m !

L_inf = 1.3948
beta = 0.3100
B = 1.1860
c_w = 0.0506

print(f"{'run':4s} | {'r_m':5s} | {'U_m':8s} | {'epochs':6s} | {'loss':7s} | {'S_math':7s} | {'D_eff,m':10s} | {'D_eff/U':7s}")

pool_math_runs = []
for h in history:
    if h['args']['N'] == 5e7 and h['args']['D'] == 1e9 and 'pool' in h['args'] and 'math' in h['args']['pool']:
        r_m = h['args']['mix'].get('math', 0.0)
        u_m = h['args']['pool']['math']
        loss = h['result']['eval_loss']['math']
        ep = (r_m * 1e9) / u_m
        s_math = ((loss - L_inf) / B)**(-1.0 / beta)
        d_eff_m = 1e9 * (s_math - c_w * (1.0 - r_m))
        ratio = d_eff_m / u_m
        pool_math_runs.append((ep, ratio, u_m, d_eff_m))
        print(f"{h['i']:4d} | {r_m:5.2f} | {u_m:8.1e} | {ep:6.2f} | {loss:7.4f} | {s_math:7.4f} | {d_eff_m:10.2e} | {ratio:7.3f}")

