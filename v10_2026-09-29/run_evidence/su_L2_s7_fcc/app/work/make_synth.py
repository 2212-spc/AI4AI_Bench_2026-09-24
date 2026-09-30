import numpy as np
r = np.random.default_rng(123)
S = np.load('/app/data/sample.npz'); mu, sd = S['X'].mean(0), S['X'].std(0)
def gen(n):
    Z = r.standard_normal((n, 32)).astype(np.float32)
    return Z
W1 = r.standard_normal((32, 64)) / np.sqrt(32) * 1.5; b1 = r.standard_normal(64) * 0.5
W2 = r.standard_normal((64, 10)) / np.sqrt(64) * 3.0
bias = np.log(np.array([34,157,146,10,9,42,172,359,28,43]) / 1000.0)
def teacher(Z):
    h = np.tanh(Z @ W1 + b1)
    z = 2.2 * (h @ W2 + 0.8 * (Z[:, :10] * Z[:, 10:20])) + bias   # linear + nonlinear pieces
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p
def labels(Z, noise):
    p = teacher(Z); y = np.array([r.choice(10, p=pi) for pi in p])
    flip = r.random(len(y)) < noise
    y[flip] = r.integers(0, 10, flip.sum())
    return y, p.argmax(1)
Ztr = gen(80000); ytr, _ = labels(Ztr, 0.27)
Zte = gen(20000); _, yte_bayes = labels(Zte, 0.0); pte = teacher(Zte); yte = np.array([r.choice(10, p=pi) for pi in pte])
Xtr = (Ztr * sd + mu).astype(np.float32); Xte = (Zte * sd + mu).astype(np.float32)
np.savez('synth_corpus.npz', X=Xtr, y=ytr); np.save('synth_test_X.npy', Xte); np.save('synth_test_y.npy', yte)
np.savez('synth_sample.npz', X=Xtr[:4000], y=ytr[:4000])
print('bayes acc on test', (yte_bayes == yte).mean(), 'class dist', np.bincount(yte) / 20000)
