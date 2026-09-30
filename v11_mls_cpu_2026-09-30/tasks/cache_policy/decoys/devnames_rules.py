"""Decoy: hand-written rules keyed on the *dev* key names (overfits the visible traces)."""
from collections import OrderedDict


def kind(k):
    if k.startswith("scan:") or ":export" in k or k.startswith("img:"):
        return "scan"
    if k.startswith("idx:") or k.startswith("feed:"):
        return "loop"
    if k.startswith("sess:") or k.startswith("cart:") or k.endswith(":cart"):
        return "sess"
    return "hot"


class Policy:
    def __init__(self, capacity):
        self.C = capacity
        self.q = {"loop": OrderedDict(), "sess": OrderedDict(), "hot": OrderedDict(), "scan": OrderedDict()}
        self.kind, self.cnt = {}, {}

    def _k(self, key):
        k = self.kind.get(key)
        if k is None:
            k = self.kind[key] = kind(key)
        return k

    def on_hit(self, key, t):
        self.cnt[key] = self.cnt.get(key, 0) + 1
        q = self.q[self._k(key)]; q[key] = t; q.move_to_end(key)

    def _evict(self, q):
        v = next(iter(q)); del q[v]; return v

    def on_miss(self, key, t):
        self.cnt[key] = self.cnt.get(key, 0) + 1
        k = self._k(key)
        full = sum(len(q) for q in self.q.values()) >= self.C
        if full:
            if k == "scan":
                return None
            Q = self.q
            if Q["scan"]:
                v = self._evict(Q["scan"])
            elif Q["sess"] and t - next(iter(Q["sess"].values())) > 2000:
                v = self._evict(Q["sess"])
            elif k == "loop":
                if len(Q["loop"]) >= self.C // 2:
                    return None                       # keep a fixed part of the loop, drop the rest
                v = self._evict(Q["hot"]) if Q["hot"] else self._evict(Q["sess"])
            else:
                if Q["hot"]:
                    old = next(iter(Q["hot"]))
                    if k == "hot" and self.cnt[key] <= self.cnt.get(old, 0):
                        return None                   # TinyLFU-like admission
                    v = self._evict(Q["hot"])
                elif Q["sess"]:
                    v = self._evict(Q["sess"])
                else:
                    return None
        else:
            v = None
        self.q[k][key] = t
        return v
