import numpy as np

# Model with saturation:
# Code:   f(ep) = 1 + 4.96 * (1 - ep^-0.30)
# Math:   f(ep) = 1 + 1.01 * (1 - ep^-1.45)
# Papers: f(ep) = 1 + 2.33 * (1 - ep^-0.60)

# Model with rational:
# Math: f(ep) = 1 + 1.675 * (ep - 1) / (1 + 1.60 * (ep - 1))
# Code: f(ep) = 1 + 1.5 * (ep - 1) / (1 + 0.3 * (ep - 1))?
# Or power law:
# Math: f(ep) = 1 + 0.658 * (ep - 1)^0.183
# Papers: f(ep) = 1 + 0.807 * (ep - 1)^0.356
# Code: f(ep) = 1 + 0.938 * (ep - 1)^0.558

# Let's compare predictions at the chosen mixture:
# mix = {'web': 0.57, 'code': 0.32, 'math': 0.02, 'papers': 0.09}
# D = 500e9:
# web: 285B (0.10 epochs -> 1.000)
# code: 160B (ep = 4.00)
# math: 10B (ep = 2.50)
# papers: 45B (ep = 2.25)

print("At chosen mixture:")
# Code (ep = 4.0):
# saturation: 1 + 4.96 * (1 - 4^-0.30) = 1 + 4.96 * (1 - 0.65975) = 1 + 4.96 * 0.34025 = 2.688
# empirical in runs: at ep=4, ratio was 2.97!
# power law: 1 + 0.938 * 3^0.558 = 1 + 0.938 * 1.847 = 2.73
print("Code D_eff/U at ep=4:")
print("  saturation: ", 1.0 + 4.96 * (1.0 - 4.0**(-0.30)))
print("  power law:  ", 1.0 + 0.938 * (4.0 - 1.0)**0.558)
print("  empirical:  ", 2.97)

# Math (ep = 2.5):
# saturation: 1 + 1.01 * (1 - 2.5^-1.45) = 1 + 1.01 * (1 - 0.264) = 1 + 1.01 * 0.736 = 1.743
# rational: 1 + 1.675 * 1.5 / (1 + 1.60 * 1.5) = 1 + 2.5125 / 3.4 = 1.739
# empirical in run 32: at ep=2.5, ratio was 1.728!
print("Math D_eff/U at ep=2.5:")
print("  saturation: ", 1.0 + 1.01 * (1.0 - 2.5**(-1.45)))
print("  rational:   ", 1.0 + 1.675 * 1.5 / (1.0 + 1.60 * 1.5))
print("  empirical:  ", 1.728)

# Papers (ep = 2.25):
# saturation: 1 + 2.33 * (1 - 2.25^-0.60) = 1 + 2.33 * (1 - 0.615) = 1 + 2.33 * 0.385 = 1.897
# power law: 1 + 0.807 * 1.25^0.356 = 1 + 0.807 * 1.083 = 1.874
# empirical: between ep=2 (1.75) and ep=4 (2.38), at 2.25 is ~ 1.85 - 1.90!
print("Papers D_eff/U at ep=2.25:")
print("  saturation: ", 1.0 + 2.33 * (1.0 - 2.25**(-0.60)))
print("  power law:  ", 1.0 + 0.807 * (2.25 - 1.0)**0.356)

