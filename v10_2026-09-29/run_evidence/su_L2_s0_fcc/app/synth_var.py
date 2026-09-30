import numpy as np, sys, json, time
sys.path.insert(0, '/app/repo')
import train as T
c = np.load('/tmp/synth_corpus.npz'); X, y = c['X'], c['y']
Xt = np.load('/tmp/synth_test_X.npy'); gt = np.load('/tmp/synth_test_g.npy')
BASE = json.load(open('/app/repo/config.json'))
n = 80000; steps = 20000

def logits(P, norm, A):
    out = []
    for i in range(0, len(A), 4096):
        z, _ = T.forward(P, norm(A[i:i + 4096])); out.append(z)
    return np.concatenate(out)

def go(tag, upd, seeds=(0,), ytrain=y):
    cfg = dict(BASE); cfg.update(upd)
    t0 = time.time(); Z = 0
    for sd in seeds:
        P, norm = T.train(X[:n], ytrain[:n], steps, cfg, seed=sd)
        z = logits(P, norm, Xt); z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        Z = Z + p
        print('  %s seed %d acc %.4f' % (tag, sd, (p.argmax(1) == gt).mean()), flush=True)
    print('%s %s ens%d acc %.4f (%.0fs)' % (tag, upd, len(seeds), (Z.argmax(1) == gt).mean(), time.time() - t0), flush=True)

# gold-label headroom (using the true generator labels for the training rows)
import synth
_, _, g = synth.make(100000, seed=0)
#go('GOLD', {'weight_decay': 0.3}, ytrain=g)
go('ens3', {'weight_decay': 0.6}, seeds=(0, 1, 2))
go('h512', {'weight_decay': 0.6, 'hidden': 512})
go('aug', {'weight_decay': 0.6, 'aug_sigma': 0.3})
go('ls', {'weight_decay': 0.6, 'label_smoothing': 0.3})
go('wd0.6', {'weight_decay': 0.6})
