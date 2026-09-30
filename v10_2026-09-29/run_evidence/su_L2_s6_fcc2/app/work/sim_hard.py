"""Harder synthetic corpus: same fitted Gaussians but class means shrunk toward the global mean (factor
SHRINK) and covariances inflated, so classes overlap more and the sample-size optimum of wd shifts."""
import numpy as np, sys
sys.path.insert(0, '/app/work'); import sim
SHRINK = float(sys.argv[1]); out = sys.argv[2]
pri, params = sim.fit()
gm = sum(p * m for p, (m, C) in zip(pri, params))
params = [(gm + SHRINK * (m - gm), C / SHRINK ** 2 * 0.7) for (m, C) in params]
Xs, yg, yc = sim.generate(100000, 1, pri=pri, params=params)
np.savez(out, X=Xs[:80000], yg=yg[:80000], yc=yc[:80000], Xt=Xs[80000:], yt=yg[80000:])
import numpy.linalg as la
Xt, yt = Xs[80000:], yg[80000:]; sc = []
for c in range(10):
    m, C = params[c]; Ci = la.inv(C); D = Xt - m
    sc.append(-0.5 * np.einsum('ij,jk,ik->i', D, Ci, D) - 0.5 * la.slogdet(C)[1] + np.log(pri[c] + 1e-12))
print('hard synthetic Bayes acc', (np.array(sc).T.argmax(1) == yt).mean())
