"""Decoy: evict a uniformly random resident key."""
import random


class Policy:
    def __init__(self, capacity):
        self.C, self.keys, self.pos, self.rng = capacity, [], {}, random.Random(0)

    def on_hit(self, key, t):
        pass

    def on_miss(self, key, t):
        v = None
        if len(self.keys) >= self.C:
            j = self.rng.randrange(len(self.keys)); v = self.keys[j]
            last = self.keys.pop(); del self.pos[v]
            if last != v:
                self.keys[j] = last; self.pos[last] = j
        self.pos[key] = len(self.keys); self.keys.append(key)
        return v
