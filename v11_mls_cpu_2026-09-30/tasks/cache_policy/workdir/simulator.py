"""Local cache simulator (same rules as the grader, but in-process and without the lock-step pipe).

    from simulator import simulate
    miss = simulate(PolicyClass, keys, capacity)      # bool array, miss[i] for access i

Rules (identical in the grader):
  * access i of key k:  if k is resident -> hit, policy.on_hit(k, i) is called;
  * otherwise miss:     d = policy.on_miss(k, i)
        - if the cache has a free slot, k is inserted (d is ignored);
        - else d must be a resident key (it is evicted and k inserted) or None (bypass: k is not cached).
  * returning a key that is not resident is an error (the setting fails).
"""
import numpy as np


class ProtocolError(Exception):
    pass


def simulate(policy_cls, keys, capacity):
    pol = policy_cls(capacity)
    res, miss = set(), np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        if k in res:
            pol.on_hit(k, i)
            continue
        miss[i] = True
        d = pol.on_miss(k, i)
        if len(res) < capacity:
            res.add(k)
        elif d is not None:
            if d not in res:
                raise ProtocolError(f"access {i}: victim {d!r} is not resident")
            res.remove(d); res.add(k)
    return miss
