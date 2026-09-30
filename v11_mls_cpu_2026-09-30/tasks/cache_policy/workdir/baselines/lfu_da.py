"""Baseline: LFU with dynamic aging (LFU-DA, Arlitt et al. 2000).
priority = in-cache request count + L, where L is the priority of the last evicted object;
evict the lowest priority (ties: least recently used).  Never bypasses."""
import heapq


class Policy:
    def __init__(self, capacity):
        self.C = capacity
        self.pri = {}          # key -> (priority, last t)
        self.heap = []
        self.L = 0.0

    def _push(self, key, p, t):
        self.pri[key] = (p, t)
        heapq.heappush(self.heap, (p, t, key))

    def on_hit(self, key, t):
        p, _ = self.pri[key]
        self._push(key, p + 1, t)

    def on_miss(self, key, t):
        victim = None
        if len(self.pri) >= self.C:
            while True:
                p, t0, k = heapq.heappop(self.heap)
                if self.pri.get(k) == (p, t0):
                    break
            victim = k
            del self.pri[k]
            self.L = p
        self._push(key, self.L + 1, t)
        return victim
