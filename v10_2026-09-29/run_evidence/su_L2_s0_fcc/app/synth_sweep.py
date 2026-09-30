import numpy as np, sys, json, time
sys.path.insert(0, '/app/repo')
import train as T
c = np.load('/tmp/synth_corpus.npz'); X, y = c['X'], c['y']
Xt = np.load('/tmp/synth_test_X.npy'); gt = np.load('/tmp/synth_test_g.npy')
BASE = json.load(open('/app/repo/config.json'))
ns = [int(a) for a in sys.argv[1].split(',')]
wds = [float(a) for a in sys.argv[2].split(',')]
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
for n in ns:
    for wd in wds:
        cfg = dict(BASE); cfg['weight_decay'] = wd; cfg.update(extra)
        steps = int(32 * n / cfg['batch_size'])
        t0 = time.time()
        P, norm = T.train(X[:n], y[:n], steps, cfg, seed=0)
        acc = (T.predict(P, norm, Xt) == gt).mean()
        print('n %d wd %.2f %s steps %d acc %.4f (%.0fs)' % (n, wd, extra, steps, acc, time.time() - t0), flush=True)
