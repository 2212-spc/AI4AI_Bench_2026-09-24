"""Synthetic corpus generator mimicking the real data (32-d, 10 classes, roughly Gaussian per class).

Class means / pooled covariance are estimated from the gold dev set; per-class covariance is a shrunk
per-class estimate from the crowd sample (so classes have distinct shapes -> nonlinear Bayes boundary).
Crowd labels: gold w.p. (1-noise), otherwise drawn from a confusion distribution skewed towards frequent
classes.  Used only for extrapolating hyper-parameter trends to n=80000; nothing here ships.
"""
import numpy as np

s = np.load('/app/data/sample.npz'); d = np.load('/app/data/dev.npz')
Xs, ys = s['X'].astype(np.float64), s['y']
Xd, yd = d['X'].astype(np.float64), d['y']
K, D = 10, 32
prior = np.bincount(yd, minlength=K) / len(yd)


def fit():
    mu = np.zeros((K, D)); cov = np.zeros((K, D, D))
    pooled = np.cov(Xd.T)
    for c in range(K):
        m_d = Xd[yd == c]
        mu[c] = m_d.mean(0) if len(m_d) >= 5 else Xs[ys == c].mean(0)
        m_s = Xs[ys == c]
        cc = np.cov(m_s.T) if len(m_s) > D + 5 else pooled
        cov[c] = 0.5 * cc + 0.5 * pooled
    return mu, cov


def make(n, noise=0.45, seed=0):
    rng = np.random.default_rng(seed)
    mu, cov = fit()
    g = rng.choice(K, n, p=prior)
    X = np.empty((n, D))
    for c in range(K):
        m = g == c
        X[m] = rng.multivariate_normal(mu[c], cov[c], m.sum())
    # nonlinearity: mild feature warping so a linear model is not Bayes-optimal
    X[:, :8] = X[:, :8] + 0.15 * np.sign(X[:, 8:16]) * X[:, 8:16] ** 2 / (1 + np.abs(X[:, 8:16]))
    flip = rng.random(n) < noise
    conf = rng.choice(K, n, p=prior)
    y = np.where(flip, conf, g)
    return X.astype(np.float32), y.astype(np.int64), g.astype(np.int64)


if __name__ == '__main__':
    X, y, g = make(100000, seed=0)
    np.savez('/tmp/synth_corpus.npz', X=X[:80000], y=y[:80000])
    np.save('/tmp/synth_test_X.npy', X[80000:]); np.save('/tmp/synth_test_g.npy', g[80000:])
    np.savez('/tmp/synth_sample.npz', X=X[:4000], y=y[:4000])
    print('crowd acc', (y == g).mean())
