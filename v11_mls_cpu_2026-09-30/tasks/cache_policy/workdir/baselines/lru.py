"""Baseline: LRU (least recently used). Never bypasses."""
from collections import OrderedDict


class Policy:
    def __init__(self, capacity):
        self.C = capacity
        self.q = OrderedDict()

    def on_hit(self, key, t):
        self.q.move_to_end(key)

    def on_miss(self, key, t):
        victim = None
        if len(self.q) >= self.C:
            victim, _ = self.q.popitem(last=False)
        self.q[key] = t
        return victim
