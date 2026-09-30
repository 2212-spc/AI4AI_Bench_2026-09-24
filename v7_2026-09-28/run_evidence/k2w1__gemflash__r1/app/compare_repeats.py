import numpy as np

# Let's inspect the ratio D_eff / U for each domain:
# For math:
# ep = 1.0  -> 1.00
# ep = 2.0  -> 1.61
# ep = 2.5  -> 1.73
# ep = 4.0  -> 1.87
# ep = 5.0  -> 1.88
# ep = 10.0 -> 1.92

# For papers:
# ep = 1.0  -> 0.99
# ep = 2.0  -> 1.75
# ep = 4.0  -> 2.38
# ep = 10.0 -> 2.64

# For code:
# ep = 1.0  -> 1.01
# ep = 2.0  -> 1.88
# ep = 4.0  -> 2.97
# ep = 10.0 -> 4.00

# Notice: In all three domains, repeating data gives DIMINISHING RETURNS!
# Specifically:
# For code, 2 epochs gives 1.88x effective tokens (decay factor ~ 0.88 for 2nd epoch)
# For math, 2 epochs gives 1.61x effective tokens (decay factor ~ 0.61 for 2nd epoch)
# For papers, 2 epochs gives 1.75x effective tokens (decay factor ~ 0.75 for 2nd epoch)

# What is the formula in literature for epoch decay?
# In Muennighoff et al. (Equation 1):
# D_eff = U * (1 + delta * (1 - exp(-k * (ep - 1))))
# Or D_eff = U * (1 + c * (ep - 1)^gamma)
# Let's check D_eff / U = 1 + c * (ep - 1)^gamma:

for name, eps, ratios in [
    ('Math', [1, 2, 2.5, 4, 5, 10], [1.0, 1.61, 1.73, 1.87, 1.88, 1.92]),
    ('Papers', [1, 2, 4, 10], [1.0, 1.75, 2.38, 2.64]),
    ('Code', [1, 2, 4, 10], [1.0, 1.88, 2.97, 4.00])
]:
    eps = np.array(eps[1:])
    y = np.array(ratios[1:]) - 1.0
    poly = np.polyfit(np.log(eps - 1), np.log(y), 1)
    gamma = poly[0]
    c = np.exp(poly[1])
    print(f"{name:8s}: D_eff = U * (1 + {c:.3f} * (ep - 1)^{gamma:.3f})")

