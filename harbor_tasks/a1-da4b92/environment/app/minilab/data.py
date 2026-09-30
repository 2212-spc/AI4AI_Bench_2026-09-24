import numpy as np


def load(folder):
    tr = np.load(folder + "/train.npz")
    va = np.load(folder + "/val.npz")
    return tr["x"], tr["y"], va["x"], va["y"]
