"""Does correcting the noisy posterior for prior-skewed label flips help on gold dev?  (real sample -> dev)"""
import numpy as np, sys, json
sys.path.insert(0, '/app/repo')
import train as T
s = np.load('/app/data/sample.npz'); d = np.load('/app/data/dev.npz')
X, y = s['X'].astype(np.float32), s['y'].astype(np.int64)
Xd, yd = d['X'].astype(np.float32), d['y'].astype(np.int64)
cfg = json.load(open('/app/repo/config.json'))
K = 10
pi = np.bincount(y, minlength=K) / len(y)          # crowd prior (available in production)


def probs(P, norm, A):
    z, _ = T.forward(P, norm(A)); z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


for wd in (2.0, 1.0):
    c = dict(cfg); c['weight_decay'] = wd
    Pd = 0; Ptr = 0
    for sd in (0, 1, 2):
        P, norm = T.train(X, y, 1000, c, seed=sd)
        Pd = Pd + probs(P, norm, Xd) / 3; Ptr = Ptr + probs(P, norm, X) / 3
    print('wd', wd, 'plain acc %.3f' % (Pd.argmax(1) == yd).mean())
    for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        print('  subtract q=%.1f acc %.3f' % (q, ((Pd - q * pi).argmax(1) == yd).mean()))
    for b in (0.25, 0.5, 0.75, 1.0):
        print('  divide beta=%.2f acc %.3f' % (b, ((Pd / pi ** b).argmax(1) == yd).mean()))
    # anchor-point estimate of q from training rows: p~(c|anchor) = 1-q+q*pi_c
    for pct in (99, 99.5, 99.9):
        top = np.percentile(Ptr, pct, axis=0)
        qc = (1 - top) / (1 - pi)
        print('  pct %.1f  q_c' % pct, np.round(qc, 2), ' mean over frequent classes %.2f' % qc[pi > 0.08].mean())
    print('  pred class dist plain', np.round(np.bincount(Pd.argmax(1), minlength=K) / len(yd), 3))
    print('  gold dist            ', np.round(np.bincount(yd, minlength=K) / len(yd), 3))
