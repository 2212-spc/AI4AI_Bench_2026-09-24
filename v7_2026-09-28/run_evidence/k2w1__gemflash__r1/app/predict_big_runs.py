import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# Model constants from fit_pure_web.py:
# General: E=1.3658, A=0.9445, B_web=0.8896, alpha=0.2250, beta=0.2650
# Code:    E=0.5648, A=0.3888, B_web=2.4080, alpha=0.3500, beta=0.2450
# Math:    E=0.7367, A=0.5283, B_web=2.9734, alpha=0.3500, beta=0.3150

# Transfer coefficients:
# In each eval set e, B_web is known, so for pure domain d at D=1e9:
# B_d = B_web + Delta_{d, web}
# For General:
# Delta_code = +0.4882 -> B_code = 0.8896 + 0.4882 = 1.3778
# Delta_math = +0.7376 -> B_math = 0.8896 + 0.7376 = 1.6272
# Delta_papers = +0.0600 -> B_papers = 0.8896 + 0.0600 = 0.9496
# Since B_d = B_primary / (c_{e, d})^beta:
# c_{e, d} = (B_web / B_d)^(1 / beta) !
# Let's check this!
# For General (web is primary, c_web = 1.0, B_primary = B_web = 0.8896):
# c_code = (0.8896 / 1.3778)**(1 / 0.2650) = (0.6457)**3.7736 = 0.191
# c_math = (0.8896 / 1.6272)**(1 / 0.2650) = (0.5467)**3.7736 = 0.103
# c_papers = (0.8896 / 0.9496)**(1 / 0.2650) = (0.9368)**3.7736 = 0.781
# Recall earlier from mixture sweep we got: c_code=0.218, c_math=0.130, c_papers=0.751!
# They match almost exactly!

# For Code:
# Code is primary (c_code = 1.0).
# Delta_code vs web = -1.2426 -> B_code = 2.4080 - 1.2426 = 1.1654
# Delta_math vs web = -0.2862 -> B_math = 2.4080 - 0.2862 = 2.1218
# Delta_papers vs web = +0.1060 -> B_papers = 2.4080 + 0.1060 = 2.5140
# With beta = 0.2450:
# c_code = 1.0
# c_web = (1.1654 / 2.4080)**(1 / 0.2450) = (0.48397)**4.0816 = 0.0515
# c_math = (1.1654 / 2.1218)**(1 / 0.2450) = (0.5492)**4.0816 = 0.0869
# c_papers = (1.1654 / 2.5140)**(1 / 0.2450) = (0.46356)**4.0816 = 0.0433

# For Math:
# Math is primary (c_math = 1.0).
# Delta_math vs web = -1.7996 -> B_math = 2.9734 - 1.7996 = 1.1738
# Delta_code vs web = -1.1250 -> B_code = 2.9734 - 1.1250 = 1.8484
# Delta_papers vs web = -1.3304 -> B_papers = 2.9734 - 1.3304 = 1.6430
# With beta = 0.3150:
# c_math = 1.0
# c_web = (1.1738 / 2.9734)**(1 / 0.3150) = (0.39477)**3.1746 = 0.0524
# c_code = (1.1738 / 1.8484)**(1 / 0.3150) = (0.6350)**3.1746 = 0.2359
# c_papers = (1.1738 / 1.6430)**(1 / 0.3150) = (0.7144)**3.1746 = 0.3444

print("D_eff formula is EXACTLY derived from pure domain baselines!")
print("Let us test predictions on the big runs (N=4e8, D=2e9)!")

# Model parameters:
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

def predict(N, D, mix):
    r = np.array([mix.get('web', 0.0), mix.get('code', 0.0), mix.get('math', 0.0), mix.get('papers', 0.0)])
    losses = {}
    for ev in ['general', 'code', 'math']:
        m = model[ev]
        s = np.dot(r, m['c'])
        d_eff = (D / 1e9) * s
        loss = m['E'] + m['A'] * (N / 1e8)**(-m['alpha']) + m['B_prim'] * (d_eff)**(-m['beta'])
        losses[ev] = loss
    comp = 0.34 * losses['general'] + 0.33 * losses['code'] + 0.33 * losses['math']
    return losses, comp

print("\n--- Big runs (N=4e8, D=2e9) Comparison ---")
for i in range(58, 62):
    h = history[i]
    pred_losses, pred_comp = predict(h['args']['N'], h['args']['D'], h['args']['mix'])
    act_losses = h['result']['eval_loss']
    act_comp = h['result']['composite']
    print(f"Run {i}: mix={h['args']['mix']}")
    print(f"  actual:    comp={act_comp:.4f}, gen={act_losses['general']:.4f}, code={act_losses['code']:.4f}, math={act_losses['math']:.4f}")
    print(f"  predicted: comp={pred_comp:.4f}, gen={pred_losses['general']:.4f}, code={pred_losses['code']:.4f}, math={pred_losses['math']:.4f}")
    print(f"  diff:      comp={act_comp-pred_comp:+.4f}, gen={act_losses['general']-pred_losses['general']:+.4f}, code={act_losses['code']-pred_losses['code']:+.4f}, math={act_losses['math']-pred_losses['math']:+.4f}")

