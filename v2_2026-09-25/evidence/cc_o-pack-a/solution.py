"""Fast drop-in replacement for pack_ref.pack (MiniStack sequence packer).

Produces bit-identical bin assignments to the reference while being roughly an order
of magnitude faster.  Self-contained: standard library + numpy only.

Algorithm
---------
The reference keeps the open bins in creation order and does a first-fit scan over them
for every sequence, then applies three closing rules.  The expensive part is the pure
Python scan (and the `min(...)` for the memory guard) over what can be thousands of open
bins.  Here the remaining capacities of the open bins are kept in a compact numpy buffer
(creation order preserved), so the scan / argmin become single vectorised calls, and
removals are in-place memmoves.  A pure Python fallback handles exotic inputs (non-int
lengths, values outside int64) with the exact same semantics.
"""
import numpy as np

_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


def _fits_int64(v):
    return type(v) is int and _I64_MIN <= v <= _I64_MAX


def _pack_numpy(lengths, C, t, m, K):
    n = len(lengths)
    out = [0] * n
    # remaining capacity of the currently open bins, in bin-creation order (compact prefix)
    rem = np.empty(n + 1, dtype=np.int64)
    ids = []            # bin id of each open bin, parallel to rem[:cnt]
    cnt = 0             # number of open bins
    nbins = 0           # number of bins created so far
    argmax = np.ndarray.argmax
    argmin = np.ndarray.argmin
    for i, L in enumerate(lengths):
        if L > C:
            # oversized: own bin, never open
            out[i] = nbins
            nbins += 1
            continue
        k = -1
        if cnt:
            view = rem[:cnt]
            mask = view >= L
            k = int(argmax(mask))          # first open bin (creation order) that fits
            if not mask[k]:
                k = -1
        if k < 0:
            k = cnt
            ids.append(nbins)
            r = C - L
            rem[k] = r
            out[i] = nbins
            nbins += 1
            cnt += 1
        else:
            r = int(rem[k]) - L
            rem[k] = r
            out[i] = ids[k]

        if r < t:                              # (1) too little room left
            del ids[k]
            cnt -= 1
            if k < cnt:
                rem[k:cnt] = rem[k + 1:cnt + 1]

        if cnt > m:                            # (2) memory guard: fullest open bin
            v = int(argmin(rem[:cnt]))         # first minimum == smallest id
            del ids[v]
            cnt -= 1
            if v < cnt:
                rem[v:cnt] = rem[v + 1:cnt + 1]

        if (i + 1) % K == 0 and cnt >= 2:      # (3) latency guard: oldest open bin
            del ids[0]
            cnt -= 1
            rem[0:cnt] = rem[1:cnt + 1]
    return out


def _pack_python(lengths, C, t, m, K):
    """Exact semantics of the reference with parallel python lists (fallback path)."""
    n = len(lengths)
    out = [0] * n
    ids = []      # open bin ids, creation order
    remo = []     # remaining capacity, parallel to ids
    nbins = 0
    for i, L in enumerate(lengths):
        if L > C:
            out[i] = nbins
            nbins += 1
            continue
        k = -1
        for j, r in enumerate(remo):
            if r >= L:
                k = j
                break
        if k < 0:
            k = len(ids)
            ids.append(nbins)
            r = C - L
            remo.append(r)
            out[i] = nbins
            nbins += 1
        else:
            r = remo[k] - L
            remo[k] = r
            out[i] = ids[k]
        if r < t:
            del ids[k]
            del remo[k]
        if len(ids) > m:
            v = remo.index(min(remo))          # first minimum == smallest id
            del ids[v]
            del remo[v]
        if (i + 1) % K == 0 and len(ids) >= 2:
            del ids[0]
            del remo[0]
    return out


def pack(lengths, params):
    C = params["capacity"]
    t = params["close_below"]
    m = params["max_open"]
    K = params["flush_every"]
    lengths = list(lengths)
    if not lengths:
        return []
    if not all(type(v) is int for v in lengths):
        # numpy / other integer scalars: normalise to python ints when exact
        try:
            conv = [int(v) for v in lengths]
            if all(c == v for c, v in zip(conv, lengths)):
                lengths = conv
        except (TypeError, ValueError):
            pass
    if (_fits_int64(C) and _fits_int64(t) and type(m) is int and type(K) is int
            and all(map(_fits_int64, lengths))):
        # C - L and rem - L stay within int64 as long as |C| and |L| are modest; guard anyway
        mx = max(lengths)
        mn = min(lengths)
        if (_I64_MIN // 2 < C < _I64_MAX // 2 and _I64_MIN // 2 < mn
                and mx < _I64_MAX // 2):
            return _pack_numpy(lengths, C, t, m, K)
    return _pack_python(lengths, C, t, m, K)
