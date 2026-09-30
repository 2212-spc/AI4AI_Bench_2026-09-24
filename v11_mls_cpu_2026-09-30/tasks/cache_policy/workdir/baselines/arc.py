"""Baseline: ARC, Adaptive Replacement Cache (Megiddo & Modha, FAST'03). Never bypasses."""
from collections import OrderedDict


class Policy:
    def __init__(self, capacity):
        self.c = capacity
        self.p = 0.0
        self.T1, self.T2, self.B1, self.B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    def on_hit(self, key, t):
        if key in self.T1:
            del self.T1[key]
        else:
            del self.T2[key]
        self.T2[key] = 1

    def _replace(self, key):
        if self.T1 and (len(self.T1) > self.p or (key in self.B2 and len(self.T1) == self.p)):
            v, _ = self.T1.popitem(last=False); self.B1[v] = 1
        else:
            v, _ = self.T2.popitem(last=False); self.B2[v] = 1
        return v

    def on_miss(self, key, t):
        c, victim = self.c, None
        full = len(self.T1) + len(self.T2) >= c
        if key in self.B1:
            self.p = min(c, self.p + max(len(self.B2) / len(self.B1), 1))
            if full: victim = self._replace(key)
            del self.B1[key]; self.T2[key] = 1
            return victim
        if key in self.B2:
            self.p = max(0.0, self.p - max(len(self.B1) / len(self.B2), 1))
            if full: victim = self._replace(key)
            del self.B2[key]; self.T2[key] = 1
            return victim
        L1 = len(self.T1) + len(self.B1)
        L2 = len(self.T2) + len(self.B2)
        if L1 == c:
            if len(self.T1) < c:
                self.B1.popitem(last=False)
                if full: victim = self._replace(key)
            else:
                victim, _ = self.T1.popitem(last=False)
        elif L1 < c and L1 + L2 >= c:
            if L1 + L2 >= 2 * c:
                self.B2.popitem(last=False)
            if full: victim = self._replace(key)
        elif full:
            victim = self._replace(key)
        self.T1[key] = 1
        return victim
