import numpy as np
from lib import *
import mlp
for wd in (0.3, 1.0):
    for lr in (0.001, 0.003):
        for n in (1000, 4000):
            print('wd', wd, 'lr', lr, 'n', n, round(acc(dict(weight_decay=wd, lr=lr, noise_rate=0.25), n=n, seeds=(0,1,2,3), mod=mlp),4), flush=True)
