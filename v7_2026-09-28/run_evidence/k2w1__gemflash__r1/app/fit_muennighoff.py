import numpy as np

# Epochs and corresponding D_eff / U values:
# ep = 1.0  -> 1.000
# ep = 2.0  -> 1.61  (mean of 1.624, 1.613, 1.609, 1.578, 1.620)
# ep = 2.5  -> 1.728
# ep = 4.0  -> 1.87  (mean of 1.896, 1.873, 1.842)
# ep = 5.0  -> 1.884
# ep = 10.0 -> 1.92  (mean of 1.906, 1.930)

eps = np.array([1.0, 2.0, 2.5, 4.0, 5.0, 10.0])
y = np.array([1.000, 1.610, 1.728, 1.870, 1.884, 1.920])

# In Muennighoff et al. (2023), Equation 1:
# D_eff = U * (1 + delta * (1 - exp(-k * (ep - 1))))
# Or D_eff = U * (1 + c_r * log(ep))?
# Or D_eff = U * (1 + (1 - exp(-k * (ep - 1))) * c_max)?
# Or D_eff = U * (1 + delta * (ep - 1)^gamma)?

# Let's test D_eff / U = 1 + c_max * (1 - exp(-k * (ep - 1))):
best_err = 1e9
best_params = None
for c_max in np.linspace(0.8, 1.2, 41):
    for k in np.linspace(0.3, 1.5, 121):
        pred = 1.0 + c_max * (1.0 - np.exp(-k * (eps - 1.0)))
        err = np.max(np.abs(pred - y))
        if err < best_err:
            best_err = err
            best_params = (c_max, k)

print("Fit 1 (exponential saturation): c_max, k =", best_params, "max_err =", best_err)
c_max, k = best_params
print("Predictions:", 1.0 + c_max * (1.0 - np.exp(-k * (eps - 1.0))))

# Let's test Muennighoff's exact formula from the paper!
# What is the formula in Muennighoff et al. (Scaling Laws for Data-Constrained Language Models)?
# In Section 3 of Muennighoff et al.:
# "R_D = D / U. We find that the effective number of tokens D_eff can be modeled as:
# D_eff = U * (1 + c * (R_D - 1)^gamma) ? Or
# D_eff = U * (1 + alpha * log(R_D))? Or
# D_eff = U * (1 + alpha * (1 - R_D^(-beta)))?"

# Let's test D_eff / U = 1 + alpha * (1 - eps**(-beta)):
best_err2 = 1e9
best_params2 = None
for alpha in np.linspace(0.8, 1.5, 71):
    for beta in np.linspace(0.4, 2.0, 161):
        pred = 1.0 + alpha * (1.0 - eps**(-beta))
        err = np.max(np.abs(pred - y))
        if err < best_err2:
            best_err2 = err
            best_params2 = (alpha, beta)

print("Fit 2 (power saturation): alpha, beta =", best_params2, "max_err =", best_err2)
alpha, beta = best_params2
print("Predictions:", 1.0 + alpha * (1.0 - eps**(-beta)))

# Let's test D_eff / U = 1 + c * (ep - 1) / (1 + d * (ep - 1)):
best_err3 = 1e9
best_params3 = None
for c in np.linspace(0.5, 1.5, 101):
    for d in np.linspace(0.4, 1.5, 111):
        pred = 1.0 + c * (eps - 1.0) / (1.0 + d * (eps - 1.0))
        err = np.max(np.abs(pred - y))
        if err < best_err3:
            best_err3 = err
            best_params3 = (c, d)
print("Fit 3 (rational / Langmuir): c, d =", best_params3, "max_err =", best_err3)
c, d = best_params3
print("Predictions:", 1.0 + c * (eps - 1.0) / (1.0 + d * (eps - 1.0)))

