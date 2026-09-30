import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# Pure web fits:
# general: E=1.3658, A=0.9445, B_web=0.8896, alpha=0.2250, beta=0.2650
# code:    E=0.5648, A=0.3888, B_web=2.4080, alpha=0.3500, beta=0.2450
# math:    E=0.7367, A=0.5283, B_web=2.9734, alpha=0.3500, beta=0.3150

# Now, for ANY pure domain run at D=1e9:
# L_e(N, 1e9) = E_e + A_e * (N/1e8)^(-alpha_e) + B_d,e
# So the difference between pure domain d and pure web at the same N and D=1e9 is:
# L_e(N, 1e9, domain d) - L_e(N, 1e9, web) = B_d,e - B_web,e !
# Let's test if this difference is CONSTANT across N = 5e7, 1e8, 2e8!

for eval_name, col in [('general', 'general'), ('code', 'code'), ('math', 'math')]:
    print(f"\n=== EVAL SET: {eval_name} ===")
    for dom in ['code', 'math', 'papers']:
        diffs = []
        for N in [5e7, 1e8, 2e8]:
            r_dom = [h['result']['eval_loss'][col] for h in history 
                     if h['args']['N'] == N and h['args']['D'] == 1e9 and h['args']['mix'].get(dom, 0) == 1.0 and not h['args'].get('pool')]
            r_web = [h['result']['eval_loss'][col] for h in history 
                     if h['args']['N'] == N and h['args']['D'] == 1e9 and h['args']['mix'].get('web', 0) == 1.0 and not h['args'].get('pool')]
            diffs.append(np.mean(r_dom) - np.mean(r_web))
        print(f"Domain {dom:7s} - web: N=5e7: {diffs[0]:+.4f}, N=1e8: {diffs[1]:+.4f}, N=2e8: {diffs[2]:+.4f} (std={np.std(diffs):.5f})")

