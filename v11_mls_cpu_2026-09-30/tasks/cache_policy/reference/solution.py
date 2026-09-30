"""Reference policy: per-template Gittins-index eviction, with the reuse model learned online from
the request stream.

* Every key is mapped to a *template*: its ':'-separated tokens with digit runs replaced by '#' and hex
  identifiers replaced by '*'  ("media:812:raw00042" -> "media:#:raw#", "sess:9f3a2c1b:cart" -> "sess:*:cart").
* Class of an object = (template, log2-bucket of how many times the key has been requested so far).
* For every class we keep a decayed histogram of reuse intervals observed on the whole request stream
  (every request reveals the interval since the previous request of that key, cached or not), plus the
  mass of requests not followed by a reuse within H accesses ("dead").
* Value of keeping an object of class c and age a (accesses since its last request) = its Gittins
  index: the best achievable hits per unit of cache space-time over all retention plans "keep it until
  age b, then drop it":
        G(c, a) = max_b  P(reuse in (a, b] | no reuse by a) / E[min(time to reuse, b) - a | no reuse by a]
  (LHD's hit density, Beckmann et al. NSDI'18, maximised over the horizon instead of fixing it).
* On a miss with a full cache: candidates = oldest and youngest resident of every class; evict the one
  with the lowest index, or bypass the incoming object if its own index (age 0) is lower still.
  Scans (never reused) get index ~0 and are bypassed; loop objects, whose reuse is due at a fixed
  interval, have an index that rises with age, so the youngest is sacrificed and a stable subset stays;
  session objects lose value once their burst window has passed; frequency buckets separate hot from
  cold objects of the same template.  Nothing depends on the template *names*.
"""
import math, re
from collections import OrderedDict, deque
import numpy as np

_HEX = re.compile(r"[0-9a-f]{6,}")
_DIG = re.compile(r"\d+")
SUB = 4                               # log bins per octave for ages / reuse intervals
DECAY, EVERY = 0.93, 2000             # statistics decay by DECAY per EVERY accesses (drift)


def template(key):
    out = []
    for tok in key.split(":"):
        if _HEX.fullmatch(tok) and any(ch.isdigit() for ch in tok):
            out.append("*")
        else:
            out.append(_DIG.sub("#", tok))
    return ":".join(out)


def bucket(n):
    return min(7, (n - 1).bit_length())


def abin(x):
    return 0 if x < 1 else int(SUB * math.log2(x)) + 1


class Policy:
    def __init__(self, capacity):
        self.C = capacity
        self.H = 16 * capacity                           # reuse beyond H counts as "never"
        self.NB = abin(self.H) + 2
        self.mid = np.array([0.5] + [2 ** ((b - 1 + 0.5) / SUB) for b in range(1, self.NB)])
        self.lo = np.array([0.0] + [2 ** ((b - 1) / SUB) for b in range(1, self.NB)])
        self.lam = -math.log(DECAY) / EVERY
        self.tpl, self.tid, self.cnt, self.last = {}, {}, {}, {}
        self.fifo = deque()
        self.hist, self.dead = {}, {}
        self.tab, self.gtab = {}, [1.0] * self.NB       # class -> index by age bin
        self.res, self.by = {}, {}                       # resident key -> class; class -> OrderedDict(key -> t)

    # ---------------------------------------------------------------- learning from the stream
    def _hist(self, c):
        h = self.hist.get(c)
        if h is None:
            h = self.hist[c] = np.zeros(self.NB); self.dead[c] = 0.0
        return h

    def _observe(self, key, t):
        tid = self.tpl.get(key)
        if tid is None:
            tid = self.tpl[key] = self.tid.setdefault(template(key), len(self.tid))
        n = self.cnt.get(key, 0) + 1
        self.cnt[key] = n
        c = (tid, bucket(n))
        prev = self.last.get(key)
        if prev is not None and t - prev[0] <= self.H:        # weights decayed by interval *start*
            self._hist(prev[1])[abin(t - prev[0])] += math.exp(self.lam * (t - prev[0]))
        self.last[key] = (t, c)
        self.fifo.append((t, key, c))
        while self.fifo[0][0] < t - self.H:
            t0, k0, c0 = self.fifo.popleft()
            if self.last[k0][0] == t0:
                self._hist(c0); self.dead[c0] += math.exp(self.lam * (t - t0))
        if t % EVERY == 0 and t:
            for c2 in self.hist:
                self.hist[c2] *= DECAY; self.dead[c2] *= DECAY
        if t % 1000 == 0:
            self._tables()
        return c

    def _index(self, h, d):
        """Gittins index per age bin: max over stop bins J of hits / space-time (keep until age lo[J])."""
        NB = self.NB
        Sh = np.concatenate([np.cumsum(h[::-1])[::-1], [0.0]])
        Shm = np.concatenate([np.cumsum((h * self.mid)[::-1])[::-1], [0.0]])
        loJ = np.concatenate([self.lo, [self.H]])
        I = np.arange(NB)[:, None]; J = np.arange(NB + 1)[None, :]
        hits = Sh[I] - Sh[J]
        life = (Shm[I] - Shm[J]) - self.lo[:, None] * hits + (loJ[None, :] - self.lo[:, None]) * (Sh[J] + d)
        r = np.where((J > I) & (life > 0), hits / np.maximum(life, 1e-12), 0.0)
        return r.max(1).tolist()

    def _tables(self):
        if not self.hist: return
        gh = sum(self.hist.values()); gd = sum(self.dead.values())
        tot = gh.sum() + gd
        if tot <= 0: return
        prior = 2.0 / tot                                   # weak pooled prior (2 pseudo-observations)
        self.tab = {c: self._index(h + prior * gh, self.dead[c] + prior * gd) for c, h in self.hist.items()}
        self.gtab = self._index(gh, gd)

    def _val(self, c, age):
        tab = self.tab.get(c) or self.gtab
        b = abin(age)
        return tab[b] if b < self.NB else 0.0

    # ---------------------------------------------------------------- cache interface
    def on_hit(self, key, t):
        c = self._observe(key, t)
        del self.by[self.res[key]][key]
        self.res[key] = c
        self.by.setdefault(c, OrderedDict())[key] = t

    def on_miss(self, key, t):
        c = self._observe(key, t)
        if len(self.res) >= self.C:
            best, bv = None, self._val(c, 0)
            for cc, od in self.by.items():
                if not od: continue
                for k in (next(iter(od)), next(reversed(od))):
                    v = self._val(cc, t - od[k])
                    if v < bv: bv, best = v, k
            if best is None:
                return None                                 # bypass
            del self.by[self.res.pop(best)][best]
        else:
            best = None
        self.res[key] = c
        self.by.setdefault(c, OrderedDict())[key] = t
        return best
