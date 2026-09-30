"""Reference: the rows are not independent.  Near-identical feature vectors carrying the identical label are
repeated records of one underlying entity; random folds leak each entity into train and validation, so CV
rewards memorisation.  Recover entities by linking rows whose nearest-neighbour distance is far below the
typical inter-point spacing (connected components), then choose by GroupKFold on the recovered groups."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from candidates import candidates


def recover_groups(X, y):
    Z = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(30, len(Z) - 1) + 1).fit(Z)
    D, I = nn.kneighbors(Z)
    D, I = D[:, 1:], I[:, 1:]
    # distance scale of genuinely different points: median distance to the 30th neighbour
    scale = np.median(D[:, -1])
    # gap in the log-distance histogram separates copies from distinct points
    ld = np.log(D[:, 0] + 1e-12)
    grid = np.linspace(ld.min(), np.log(scale), 200)
    hist = np.array([np.sum(np.abs(ld - g) < 0.1) for g in grid])
    lo = int(np.argmax(hist[:100]))                       # mode of the near-copy mode
    cut = np.exp(grid[lo + int(np.argmin(hist[lo:]))])     # first valley after it
    cut = min(cut, 0.25 * scale)
    r, c = np.nonzero(D < cut)
    from scipy.sparse import coo_matrix
    A = coo_matrix((np.ones(len(r)), (r, I[r, c])), shape=(len(Z), len(Z)))
    return connected_components(A, directed=False)[1]


def select(X, y):
    g = recover_groups(X, y)
    cv = GroupKFold(5)
    sc = {n: -cross_val_score(mk(), X, y, cv=cv, groups=g, scoring="neg_mean_squared_error").mean() for n, mk in candidates()}
    return min(sc, key=sc.get)
