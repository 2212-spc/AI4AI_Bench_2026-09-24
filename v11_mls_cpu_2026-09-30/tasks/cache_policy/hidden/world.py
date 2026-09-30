"""Hidden world for cache_policy: synthetic key-value access traces (never shown to agents).

A trace interleaves several *streams*.  Each stream owns one key template (e.g. "user:{id}:profile")
and one reuse behaviour:
  zipf  - hot objects, Zipf popularity whose ranking drifts over time
  scan  - one-pass sequential blocks, a key is never requested twice
  loop  - cyclic sweep over a fixed working set (restarted on a new working set now and then)
  sess  - short-lived sessions: a new session id is touched a few times within a short window, then dies
Stream intensities vary over time (sinusoids / on-off bursts).  Different settings use different
template names, cache sizes and mixtures; several templates share their first token ("user:...")
while behaving differently, so the behaviour is a property of the whole key template.
"""
import heapq
from collections import OrderedDict
import numpy as np


# ----------------------------------------------------------------------------------------- streams
class Zipf:
    def __init__(self, rng, fmt, n, alpha, drift):
        self.rng, self.fmt, self.drift = rng, fmt, drift
        self.ids = rng.choice(np.arange(10_000, 10_000 + 50 * n), size=n, replace=False)
        p = 1.0 / np.arange(1, n + 1) ** alpha
        self.cdf = np.cumsum(p / p.sum())
        self.n = n

    def draw(self, m):
        rng, out = self.rng, []
        ranks = np.searchsorted(self.cdf, rng.random(m))
        flips = rng.random(m) < self.drift
        for i in range(m):
            if flips[i]:   # a popular object cools down, a random one heats up
                a, b = rng.integers(0, max(20, self.n // 100)), rng.integers(0, self.n)
                self.ids[a], self.ids[b] = self.ids[b], self.ids[a]
            out.append(self.fmt.format(id=int(self.ids[ranks[i]])))
        return out


class Scan:
    def __init__(self, rng, fmt, joblen):
        self.rng, self.fmt, self.joblen = rng, fmt, joblen
        self.job, self.blk = int(rng.integers(1, 50)), 0

    def draw(self, m):
        out = []
        for _ in range(m):
            out.append(self.fmt.format(a=self.job, b=self.blk))
            self.blk += 1
            if self.blk >= self.joblen * (0.5 + self.rng.random()):
                self.job += 1; self.blk = 0
        return out


class Loop:
    def __init__(self, rng, fmt, L, passes):
        self.rng, self.fmt, self.L, self.passes = rng, fmt, L, passes
        self.base, self.i, self.p = int(rng.integers(1, 90)), 0, 0

    def draw(self, m):
        out = []
        for _ in range(m):
            out.append(self.fmt.format(a=self.base, b=self.i))
            self.i += 1
            if self.i >= self.L:
                self.i = 0; self.p += 1
                if self.p >= self.passes:      # new working set (e.g. next table / next epoch shard)
                    self.p = 0; self.base += 1
        return out


class Sess:
    def __init__(self, rng, fmts, active, mean_len):
        self.rng, self.fmts, self.active, self.mean_len = rng, fmts, active, mean_len
        self.pool = []   # [id, remaining]

    def _new(self):
        return [f"{int(self.rng.integers(0, 16 ** 8)):08x}", 1 + int(self.rng.geometric(1.0 / self.mean_len))]

    def draw(self, m):
        rng, out = self.rng, []
        for _ in range(m):
            while len(self.pool) < self.active:
                self.pool.append(self._new())
            j = int(rng.integers(0, len(self.pool)))
            s = self.pool[j]
            out.append(self.fmts[int(rng.integers(0, len(self.fmts)))].format(h=s[0]))
            s[1] -= 1
            if s[1] <= 0:
                self.pool[j] = self.pool[-1]; self.pool.pop()
        return out


def _make_stream(rng, spec):
    t = spec["type"]
    if t == "zipf": return Zipf(rng, spec["fmt"], spec["n"], spec["alpha"], spec.get("drift", 0.0))
    if t == "scan": return Scan(rng, spec["fmt"], spec["joblen"])
    if t == "loop": return Loop(rng, spec["fmt"], spec["L"], spec.get("passes", 10 ** 9))
    if t == "sess": return Sess(rng, spec["fmts"], spec["active"], spec["mean_len"])
    raise ValueError(t)


def _weights(spec, t):
    w = np.full(len(t), float(spec["w"]))
    mod = spec.get("mod")
    if mod and mod[0] == "sin":
        _, amp, period, phase = mod
        w *= np.clip(1 + amp * np.sin(2 * np.pi * t / period + phase), 0, None)
    elif mod and mod[0] == "burst":            # on for frac of each period
        _, frac, period, phase = mod
        w *= (((t / period + phase) % 1.0) < frac).astype(float)
    return w


def make_trace(seed, streams, n):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    W = np.stack([_weights(s, t) for s in streams], 1) + 1e-12
    P = W / W.sum(1, keepdims=True)
    lab = (rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1)
    lab = np.minimum(lab, len(streams) - 1)
    keys = np.empty(n, dtype=object)
    for i, s in enumerate(streams):
        idx = np.flatnonzero(lab == i)
        keys[idx] = _make_stream(np.random.default_rng([seed, i]), s).draw(len(idx))
    return list(keys)


# ----------------------------------------------------------------------------------------- settings
N = 150_000
WARM = 30_000   # first WARM accesses are warm-up (policy runs, misses not scored)

SETTINGS = [
    dict(name="dev_a", hidden=False, seed=101, C=1000, streams=[
        dict(type="zipf", fmt="user:{id}:profile", n=30000, alpha=0.85, drift=0.002, w=0.40, mod=("sin", 0.3, 40000, 0.0)),
        dict(type="scan", fmt="scan:job{a}:blk{b:05d}", joblen=4000, w=0.45, mod=("burst", 0.5, 30000, 0.1)),
        dict(type="loop", fmt="idx:shard{a}:page{b}", L=800, passes=25, w=0.20),
        dict(type="sess", fmts=["sess:{h}:cart", "sess:{h}:token"], active=40, mean_len=6, w=0.18,
             mod=("sin", 0.5, 25000, 1.0)),
    ]),
    dict(name="dev_b", hidden=False, seed=202, C=2000, streams=[
        dict(type="zipf", fmt="user:{id}:profile", n=40000, alpha=0.9, drift=0.001, w=0.35),
        dict(type="scan", fmt="user:{a}:export{b}", joblen=6000, w=0.30, mod=("burst", 0.6, 40000, 0.3)),
        dict(type="loop", fmt="feed:{a}:page{b}", L=1400, passes=20, w=0.25, mod=("sin", 0.3, 50000, 2.0)),
        dict(type="sess", fmts=["user:{h}:cart"], active=60, mean_len=5, w=0.15),
    ]),
    dict(name="dev_c", hidden=False, seed=303, C=2500, streams=[
        dict(type="zipf", fmt="shop:{id}:item", n=50000, alpha=0.9, drift=0.002, w=0.45),
        dict(type="zipf", fmt="shop:{id}:stock", n=40000, alpha=0.75, drift=0.001, w=0.20),
        dict(type="sess", fmts=["cart:{h}:line"], active=70, mean_len=4, w=0.25, mod=("sin", 0.4, 30000, 1.5)),
        dict(type="scan", fmt="img:{a}:tile{b}", joblen=3000, w=0.10, mod=("burst", 0.5, 25000, 0.2)),
    ]),
    dict(name="hid_scan", hidden=True, seed=313, C=1500, streams=[
        dict(type="zipf", fmt="acct:{id}:settings", n=30000, alpha=0.8, drift=0.002, w=0.30),
        dict(type="scan", fmt="etl:run{a}:part{b:06d}", joblen=8000, w=0.55, mod=("sin", 0.6, 30000, 0.5)),
        dict(type="scan", fmt="backup:{a}:chunk{b}", joblen=3000, w=0.15, mod=("burst", 0.4, 20000, 0.0)),
        dict(type="sess", fmts=["chk:{h}:cart", "chk:{h}:addr"], active=50, mean_len=7, w=0.20),
    ]),
    dict(name="hid_loop", hidden=True, seed=404, C=800, streams=[
        dict(type="zipf", fmt="item:{id}:meta", n=20000, alpha=0.9, drift=0.001, w=0.35),
        dict(type="loop", fmt="tbl:t{a}:row{b}", L=900, passes=30, w=0.35),
        dict(type="loop", fmt="item:{a}:emb{b}", L=500, passes=40, w=0.15, mod=("sin", 0.5, 30000, 0.0)),
        dict(type="scan", fmt="log:{a}:seg{b}", joblen=5000, w=0.15, mod=("burst", 0.5, 25000, 0.7)),
    ]),
    dict(name="hid_zipf", hidden=True, seed=505, C=2500, streams=[
        dict(type="zipf", fmt="prod:{id}:detail", n=60000, alpha=0.95, drift=0.003, w=0.50),
        dict(type="zipf", fmt="prod:{id}:price", n=60000, alpha=0.7, drift=0.001, w=0.25),
        dict(type="sess", fmts=["prod:{h}:view"], active=80, mean_len=4, w=0.25, mod=("sin", 0.4, 35000, 0.3)),
    ]),
    dict(name="hid_media", hidden=True, seed=606, C=1200, streams=[
        dict(type="zipf", fmt="media:{id}:thumb", n=25000, alpha=0.85, drift=0.002, w=0.30),
        dict(type="scan", fmt="media:{a}:raw{b}", joblen=5000, w=0.30, mod=("burst", 0.5, 30000, 0.4)),
        dict(type="loop", fmt="media:{a}:seg{b}", L=1000, passes=20, w=0.20),
        dict(type="sess", fmts=["media:{h}:play", "media:{h}:pos"], active=40, mean_len=8, w=0.20,
             mod=("sin", 0.5, 20000, 2.0)),
    ]),
]


def build_setting(row):
    keys = make_trace(row["seed"], row["streams"], N)
    return dict(name=row["name"], hidden=row["hidden"], C=row["C"], keys=keys, warm=WARM)


# ----------------------------------------------------------------------------------------- reference sims
def lru_misses(keys, C):
    c, miss = OrderedDict(), np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        if k in c:
            c.move_to_end(k)
        else:
            miss[i] = True
            c[k] = 1
            if len(c) > C:
                c.popitem(last=False)
    return miss


def opt_misses(keys, C):
    """Belady MIN with bypass (optimal miss count for unit-size objects)."""
    n = len(keys)
    nxt, last = np.empty(n, np.int64), {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(keys[i], n + i)   # never again -> beyond the end (distinct values)
        last[keys[i]] = i
    res, heap, miss = {}, [], np.zeros(n, bool)
    for i, k in enumerate(keys):
        if k in res:
            res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
            continue
        miss[i] = True
        if len(res) < C:
            res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
            continue
        while True:
            nu, v = heap[0]
            if res.get(v) == -nu: break
            heapq.heappop(heap)
        if nxt[i] >= -nu:
            continue                           # bypass
        heapq.heappop(heap); del res[v]
        res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
    return miss
