import json
import numpy as np

history = json.load(open('/app/full_history.json'))

# Focus on Math eval loss at N=5e7, D=1e9
# When epochs = 1, we have:
# web=1.0: loss = 4.3826
# web+math sweep (r_math from 0.005 to 0.8, and 1.0):
# web+papers sweep (r_papers from 0.05 to 0.4, and 1.0):
# web 0.8 + code 0.2: loss = 3.9233, pure code: loss = 3.2588
# balanced 0.25 all: loss = 2.9604

# Model:
# Effective tokens for domain d: D_eff,d = D_d if epochs <= 1 else U_d * f(epochs_d)
# Then total effective math tokens:
# S_math = c_web * D_eff,web + c_code * D_eff,code + c_math * D_eff,math + c_papers * D_eff,papers
# (with c_math = 1.0)
# Then loss_math = L_inf + B / (S_math / 1e9)^beta

# Let's test this model on ALL N=5e7, D=1e9 runs!
