import numpy as np
from lib import *
import mlp
for nr in (0.1, 0.2, 0.3, 0.4):
    print('nr', nr, round(acc(dict(weight_decay=3.0, noise_rate=nr), n=4000, seeds=(0,1,2,3,4), mod=mlp),4), flush=True)
for nr in (0.0, 0.25):
  for wd in (2.0, 4.0):
    print('wd', wd, 'nr', nr, round(acc(dict(weight_decay=wd, noise_rate=nr), n=4000, seeds=(0,1,2,3,4), mod=mlp),4), flush=True)
