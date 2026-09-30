"""Reference *fast* solution for family O (blind oracle, kept out of the agent image).

Two segment trees over bin creation index:
  hi[] = remaining capacity of open bins, -1 for closed/unborn  -> leftmost index with value >= L
  lo[] = remaining capacity of open bins, +INF for closed       -> index of the minimum (leftmost wins ties)
plus a monotone pointer for "oldest open bin" and an open-bin counter.
"""
INF = 1 << 60


def pack(lengths, params):
    C = params["capacity"]; t = params["close_below"]; m = params["max_open"]; K = params["flush_every"]
    n = len(lengths)
    size = 1
    while size < n + 2:
        size <<= 1
    hi = [-1] * (2 * size)
    lo = [INF] * (2 * size)
    rem = [0] * (n + 2)
    isopen = bytearray(n + 2)

    def setval(i, v_open, r):
        p = i + size
        hi[p] = r if v_open else -1
        lo[p] = r if v_open else INF
        p >>= 1
        while p:
            a, b = 2 * p, 2 * p + 1
            hi[p] = hi[a] if hi[a] >= hi[b] else hi[b]
            lo[p] = lo[a] if lo[a] <= lo[b] else lo[b]
            p >>= 1

    def leftmost_ge(L):
        if hi[1] < L:
            return -1
        p = 1
        while p < size:
            p <<= 1
            if hi[p] < L:
                p += 1
        return p - size

    def argmin_open():
        p = 1
        while p < size:
            p <<= 1
            if lo[p] > lo[p + 1]:
                p += 1
        return p - size

    out = []
    nbins = 0
    nopen = 0
    oldest = 0
    for i in range(n):
        L = lengths[i]
        if L > C:
            rem[nbins] = 0
            out.append(nbins)
            nbins += 1
            continue
        b = leftmost_ge(L)
        if b < 0:
            b = nbins
            nbins += 1
            rem[b] = C
            isopen[b] = 1
            nopen += 1
            setval(b, True, C)
        rem[b] -= L
        out.append(b)
        if rem[b] < t:
            isopen[b] = 0
            nopen -= 1
            setval(b, False, 0)
        else:
            setval(b, True, rem[b])
        if nopen > m:
            v = argmin_open()
            isopen[v] = 0
            nopen -= 1
            setval(v, False, 0)
        if (i + 1) % K == 0 and nopen >= 2:
            while not isopen[oldest]:
                oldest += 1
            isopen[oldest] = 0
            nopen -= 1
            setval(oldest, False, 0)
    return out
