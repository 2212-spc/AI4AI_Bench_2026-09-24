"""Per-epoch visiting order with block locality (keeps reads sequential inside a block)."""
import numpy as np

BLOCK = 64


def epoch_order(n, seed, epoch):
    """Order in which documents are visited during `epoch`."""
    rng = np.random.default_rng(int(seed) + 9973 * int(epoch))
    nb = n // BLOCK
    out = []
    for b in rng.permutation(nb):
        s = int(b) * BLOCK
        out.extend(int(s + j) for j in rng.permutation(BLOCK))
    return out
