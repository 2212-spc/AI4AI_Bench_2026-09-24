import numpy as np, sys, json, time
sys.path.insert(0, '/app/repo')
import train as T
c = np.load('/tmp/synth_corpus.npz'); X, y = c['X'], c['y']
Xt = np.load('/tmp/synth_test_X.npy'); gt = np.load('/tmp/synth_test_g.npy')
BASE = json.load(open('/app/repo/config.json'))
n = 80000; K = 10
pi = np.bincount(y, minlength=K) / len(y)

def probs(P, norm, A):
    out = []
    for i in range(0, len(A), 4096):
        z, _ = T.forward(P, norm(A[i:i + 4096])); z = z - z.max(1, keepdims=True); p = np.exp(z); out.append(p / p.sum(1, keepdims=True))
    return np.concatenate(out)

def go(tag, upd, seed=0):
    cfg = dict(BASE); cfg.update(upd)
    steps = int(32 * n / cfg['batch_size'])
    t0 = time.time()
    P, norm = T.train(X[:n], y[:n], steps, cfg, seed=seed)
    p = probs(P, norm, Xt)
    corr = ' '.join('q%.1f=%.4f' % (q, ((p - q * pi).argmax(1) == gt).mean()) for q in (0.2, 0.3, 0.45, 0.6))
    print('%-8s %s seed %d acc %.4f | %s (%.0fs)' % (tag, upd, seed, (p.argmax(1) == gt).mean(), corr, time.time() - t0), flush=True)

which = sys.argv[1]
if which == 'a':
    go('wd0.6', {'weight_decay': 0.6}); go('wd0.6', {'weight_decay': 0.6}, seed=1)
    go('wd0.3', {'weight_decay': 0.3}, seed=1); go('wd1.0', {'weight_decay': 1.0}, seed=1)
elif which == 'b':
    go('aug', {'weight_decay': 0.6, 'aug_sigma': 0.3}); go('ls', {'weight_decay': 0.6, 'label_smoothing': 0.3})
    go('bs256', {'weight_decay': 0.6, 'batch_size': 256}); go('lr1e-3', {'weight_decay': 0.6, 'lr': 0.001})
