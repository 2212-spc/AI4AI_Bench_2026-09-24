"""Run train_new.train on a synthetic corpus; args: corpus.npz n json-overrides seeds"""
import numpy as np, sys, json, time
sys.path.insert(0, '/app/work'); import train_new as T
S = np.load(sys.argv[1]); X, yc, Xt, yt = S['X'], S['yc'], S['Xt'], S['yt']
BASE = json.load(open('/app/repo/config.json'))
n = int(sys.argv[2]); over = json.loads(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(',')]
for sd in seeds:
    c = dict(BASE, **over); t0 = time.process_time()
    M, norm = T.train(X[:n], yc[:n], int(32 * n / c['batch_size']), c, seed=sd)
    pr = T.predict_proba(M, norm, Xt)
    acc = (pr.argmax(1) == yt).mean()
    members = [(T.softmax(T.forward(P, norm(Xt))[0]).argmax(1) == yt).mean() for P in M]
    print(n, over, 'seed', sd, 'test %.4f' % acc, 'members', ['%.4f' % m for m in members],
          'cpu %.0fs' % (time.process_time() - t0), flush=True)
