import sys, numpy as np
z = np.load(sys.argv[1]); V = z["V"]; D = z["D"]
mu = V.mean(1); se = V.std(1, ddof=1) / np.sqrt(V.shape[1])
lab = lambda m: "".join("1" if m >> i & 1 else "0" for i in range(6))
order = np.argsort(mu)
print("n seeds", V.shape[1], "base", round(mu[0], 5))
for m in order[:14]: print(lab(m), "%.5f ±%.5f div%d" % (mu[m], se[m], D[m].sum()))
print("...")
for nm, m in [("c1", 1), ("c2", 2), ("c3", 4), ("c4", 8), ("c5", 16), ("c6", 32)]: print("single", nm, "%.5f" % mu[m], "d=%.5f" % (mu[m] - mu[0]))
for m in [3, 5, 6, 7, 63, 3 | 8, 7 | 8, 7 | 16, 7 | 8 | 16]: print(lab(m), "%.5f div%d" % (mu[m], D[m].sum()))
# c6 effect across all cells (paired)
d6 = [(V[m | 32] - V[m]).mean() for m in range(32)]; print("c6 mean paired effect %.6f max|.| %.6f" % (np.mean(d6), np.max(np.abs(d6))))
