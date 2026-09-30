"""Baseline: S3-FIFO (Yang et al., SOSP'23): small probationary FIFO (10%), main FIFO with
2-bit frequency and reinsertion, ghost FIFO of keys recently evicted from the small queue.
One-hit wonders (e.g. scans) leave through the small queue quickly.  Never bypasses."""
from collections import OrderedDict


class Policy:
    def __init__(self, capacity):
        self.C = capacity
        self.s_cap = max(1, capacity // 10)
        self.S, self.M, self.G = OrderedDict(), OrderedDict(), OrderedDict()   # key -> freq

    def on_hit(self, key, t):
        if key in self.S: self.S[key] = min(3, self.S[key] + 1)
        else: self.M[key] = min(3, self.M[key] + 1)

    def _evict(self):
        while True:
            if len(self.S) >= self.s_cap or not self.M:
                k, f = self.S.popitem(last=False)
                if f > 0:
                    self.M[k] = 0            # promoted, keep looking for a victim
                    continue
                self.G[k] = 1
                if len(self.G) > self.C: self.G.popitem(last=False)
                return k
            k, f = self.M.popitem(last=False)
            if f > 0:
                self.M[k] = f - 1
                continue
            return k

    def on_miss(self, key, t):
        victim = self._evict() if len(self.S) + len(self.M) >= self.C else None
        if key in self.G:
            del self.G[key]; self.M[key] = 0
        else:
            self.S[key] = 0
        return victim
