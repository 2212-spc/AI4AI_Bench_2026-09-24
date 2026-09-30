import numpy as np
def deff(D,U,Rs):
    Um=np.minimum(D,U); R=np.maximum(D/U-1,0)
    return Um+Um*Rs*(1-np.exp(-R/Rs))
