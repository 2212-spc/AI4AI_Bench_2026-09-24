"""Streaming shuffle with a bounded reservoir, as used by the iterable dataset."""
import numpy as np


def shuffle_stream(items, buffer_size, seed):
    """Emit every item exactly once, in a shuffled order, holding at most buffer_size items."""
    rng = np.random.default_rng(seed)
    buf = []
    out = []
    for it in items:
        if len(buf) < max(1, int(buffer_size)):
            buf.append(it)
            continue
        j = int(rng.integers(0, len(buf)))
        out.append(buf[j])
        buf[j] = it
    while buf:
        j = int(rng.integers(0, len(buf)))
        out.append(buf.pop(j))
    return out
