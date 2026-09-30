import numpy as np
from lib import *
import mlp
for wd in (0.3, 1.0, 3.0):
    for n in (1000, 2000, 4000):
        print('wd', wd, 'n', n, round(acc(dict(weight_decay=wd, noise_rate=0.25), n=n, seeds=(0,1,2,3,4), mod=mlp),4), flush=True)
