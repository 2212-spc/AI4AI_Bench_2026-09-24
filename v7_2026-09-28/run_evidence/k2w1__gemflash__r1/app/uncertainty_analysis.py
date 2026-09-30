import numpy as np

# We want an interval [lo, hi] of width at most 0.05.
# Let width = 0.048 (or 0.046).
# If the central estimate is ~ 1.435, an interval [1.412, 1.458] has width 0.046.
# Let's test the range across:
# 1. Parameter uncertainty in Chinchilla scaling (alpha, beta, E, A, B)
# 2. Parameter uncertainty in transfer matrix
# 3. Parameter uncertainty in repeat models

# Let's check how much the predicted target loss changes if:
# - alpha varies by +/- 0.02
# - beta varies by +/- 0.02
# - E varies by +/- 0.05
# Remember: In fitting pure web data:
# N goes from 5e7..4e8 (factor of 8) to 2.5e9 (factor of 6.25 from 4e8)
# D goes from 1e9..8e9 (factor of 8) to 500e9 (factor of 62.5 from 8e9)
# BUT effective D_eff for each eval set:
# General: D_eff ~ 320B (factor of 40 from 8e9)
# Code: D_eff ~ 125B (factor of 15 from 8e9)
# Math: D_eff ~ 60B (factor of 7.5 from 8e9)

