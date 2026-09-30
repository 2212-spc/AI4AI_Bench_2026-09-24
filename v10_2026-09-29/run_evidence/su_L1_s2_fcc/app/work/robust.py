# quick dev-scored check on the REAL sample: label smoothing, aug, lr, warmup, hidden
from harness import *
import numpy as np
tests = {
 "base": {},
 "ls0.1": {"label_smoothing":0.1},
 "ls0.3": {"label_smoothing":0.3},
 "aug0.1": {"aug_sigma":0.1},
 "aug0.3": {"aug_sigma":0.3},
 "lr0.002": {"lr":0.002},
 "lr0.005": {"lr":0.005},
 "bs256": {"batch_size":256},
 "wd4": {"weight_decay":4.0},
 "hid128": {"hidden":128},
}
for name,ov in tests.items():
    a=[run(ov, seed=s) for s in range(3)]
    print(f"{name:10s} {np.mean(a):.3f}  {np.round(a,3)}", flush=True)
