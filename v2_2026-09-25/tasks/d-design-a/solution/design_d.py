"""Family D - the value of a measurement you have not taken yet.

Families B and S ask what the surviving records already pin down.  Both frontier models under test now
answer that correctly, including the sealed arm where the censoring mechanism had to be recovered rather
than read.  What they have never been asked is the question an on-call engineer actually faces next:
*which run should I re-launch?*  That is a different capability.  It is not bound computation; it is
reasoning about how information would propagate through a constraint the agent does not control.

The setting is family B's archive.  A day's scheduler digest pins the **sum** of that day's pass rates, so
the launched runs whose rows are gone are not free of one another: pushing one up forces the others down.
Now suppose one of those runs could be recovered - its true value read off a backup.  You do not get to
choose what the value turns out to be; nature does, subject to everything `/app` already says.  Define

    guaranteed width of a contrast after recovering S
        = the largest width the sharp interval could still have, over every assignment consistent with
          `/app` of the values that recovering S would reveal.

Recovering a set can never make the interval wider - knowledge only grows - so this is a well-defined
non-increasing function of S, and "run r is worth recovering" means the guaranteed width strictly drops.

The structure this exposes is the point of the family, and it defeats both obvious heuristics:

  * **a run inside one of the two cells is usually worthless.**  On a day with plenty of slack, fixing one
    run's value lets the adversary move the same slack onto its neighbours; the day's surplus simply
    relocates and the width does not move at all.  Recovering the widest box on the sweep buys nothing if
    that box sits on a loose day.
  * **a run in neither cell is often valuable.**  Such a run is part of the day's total, so its box is
    part of the capacity that lets the other runs absorb surplus.  Shrinking that capacity can force the
    day's remaining surplus down, which squeezes both cells at once.

So the ranking of candidate measurements is not the ranking of their box widths, and it is not "cell
members first".  It is decided by how tight each run's day is - a property of the day, not of the run.

Everything here is exact rational arithmetic and every quantity is computed two ways: a closed form
derived from the single per-day equality, and a continuous-knapsack solved by sorting coefficients.  The
two are asserted equal at every evaluation point, so an arithmetic slip cannot reach the key.  A third,
wholly separate path - build the assignment that attains the maximum, hand it to `truth_b.py` as if the
values had really been recovered, and re-derive the interval - is run by the generator as gate G3.

`RULES` holds the competing procedures the item set has to be able to tell apart.  They are not random
perturbations: each is a specific defensible-sounding reading of "what recovering a run buys you", and
the generator keeps only items on which enough of them give a different answer from the reference.
"""
import itertools
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import truth_b as TB                                                            # noqa: E402

ZERO = Fraction(0)
HALF = Fraction(1, 2)
KNOBS = TB.KNOBS

# How a procedure treats a recovered value, and whether it uses the daily totals at all.
REF = {"reveal": "worst", "coupling": True}
RULES = {
    # "The values I recover are whatever they are, but the *rest* of the day is unchanged" - i.e. the
    # surplus left over for the runs still unknown is the same number as before.  It is the single most
    # natural slip, and it is invisible on a slack day and decisive on a tight one.
    "surplus_stays_put": {"reveal": "fixed", "coupling": True},
    # Optimism: report the narrowest width the recovery could produce rather than the width it guarantees.
    "recovered_value_is_best_case": {"reveal": "best", "coupling": True},
    # Treat the unknown recovered value as if it came in at the middle of its box.
    "recovered_value_is_the_box_midpoint": {"reveal": "mid", "coupling": True},
    # ... or at the top of its box, which is where an optimist expects a rescued run to land.
    "recovered_value_is_the_box_top": {"reveal": "top", "coupling": True},
    # Box arithmetic: each unknown run contributes its own width and the daily digest is never used.
    "day_totals_ignored": {"reveal": "worst", "coupling": False},
    # The rule an agent learns by looking at two or three ordinary days and stopping there.
    "value_is_box_width_over_n": {"reveal": "additive", "coupling": True},
}


class Design(object):
    """Information-value machinery over one archive.  Reads only files the agent can read."""

    def __init__(self, app, rule=None, recoverable=None, grain=None):
        self.T = TB.Truth(app)
        self.rule = dict(rule or REF)
        # What one recovery buys, per run.  `grain[rid]` is the width of the range the recovery leaves:
        # 0 means the exact value comes back, a width at or above the run's own box means the request
        # comes back empty and the run is untouched, and anything between is a window - the recovery
        # narrows the run to an interval of that width, but which interval is not determined by the
        # archive, so the worst case ranges over the placements as well as over the values.
        #
        # The base arm passes neither argument, so every request is exact and this is all inert;
        # `recoverable` is the binary special case, kept because it reads better at the call site.  Both
        # arms therefore run through one code path, and `d-design-a`'s key is unchanged by any of it.
        self.grain = {} if grain is None else {r: Fraction(u) for r, u in grain.items()}
        self.recoverable = None if recoverable is None else set(recoverable)
        self.box = {}
        for d, rids in self.T.free.items():
            for rid in rids:
                self.box[rid] = self.T.support(rid)
        self.unknown = set(self.box)
        self._popcache = {}
        self._daycache = {}
        self._ordcache = {}

    # -- day decomposition ---------------------------------------------------------------------------

    def _pop(self, spec):
        k = tuple(spec[x] for x in KNOBS)
        if k not in self._popcache:
            self._popcache[k] = set(self.T.pop(spec))
        return self._popcache[k]

    def relevant_days(self, a, b):
        """Days on which at least one unrecorded member of either cell finished.  Recovering a run on any
        other day cannot move this contrast at all, whatever its box."""
        pa, pb = self._pop(a), self._pop(b)
        return [d for d in sorted(self.T.free)
                if any(r in pa or r in pb for r in self.T.free[d])]

    def group(self, a, b, rid):
        """P (a member of `to`), M (a member of `from`) or O (neither)."""
        return "P" if rid in self._pop(b) else ("M" if rid in self._pop(a) else "O")

    def u(self, rid):
        """Width of the range a recovery of this run would leave behind, clamped to its box."""
        w = self.box[rid][1] - self.box[rid][0]
        if self.recoverable is not None and rid not in self.recoverable:
            return w
        g = self.grain.get(rid, ZERO)
        return w if g > w else g

    def parts(self, a, b, day, S):
        """(Lp, cP, Lm, cM, Lo, cO, T, C_S) for one day, with the runs in S cut down to what a recovery
        would leave of them.

        Writing v_r = lo_r + x_r, the day's equality is sum(x) = T over x_r in [0, hi_r - lo_r].  A
        recovery of r pins x_r to [a_r, a_r + u_r] for some placement a_r the archive does not determine,
        so it splits r's capacity in two: the part the placement eats, which joins the pool `cs` of
        surplus the recovered runs can absorb, and the residue u_r, which stays where it was - in r's own
        group, still competing for whatever surplus is left.  With u_r = 0 that is the exact reveal the
        base arm uses, and with u_r the full box it is a request that came back empty.
        """
        L = {"P": ZERO, "M": ZERO, "O": ZERO}
        C = {"P": ZERO, "M": ZERO, "O": ZERO}
        T = self.T.resid[day]
        cs = ZERO
        for rid in self.T.free[day]:
            lo, hi = self.box[rid]
            T -= lo
            g = self.group(a, b, rid)
            L[g] += lo
            if rid in S:
                u = self.u(rid)
                cs += (hi - lo) - u
                C[g] += u
                continue
            C[g] += hi - lo
        return L["P"], C["P"], L["M"], C["M"], L["O"], C["O"], T, cs

    @staticmethod
    def _w_closed(t, cP, cM, cO, na, nb):
        """Sharp width of the contrast on one day, given that day's remaining surplus `t`."""
        up = min(t, cP)
        dp = max(ZERO, t - cM - cO)
        um = min(t, cM)
        dm = max(ZERO, t - cP - cO)
        return Fraction(up - dp, 1) / nb + Fraction(um - dm, 1) / na

    def _orders(self, a, b, day, S):
        """The two fill orders for the day's knapsacks.  They depend on the coefficients only, not on
        the surplus, so they are computed once per (contrast, day, S) and reused for every candidate t."""
        key = (tuple(a[x] for x in KNOBS), tuple(b[x] for x in KNOBS), day,
               frozenset(r for r in self.T.free[day] if r in S))
        hit = self._ordcache.get(key)
        if hit is None:
            one = Fraction(1, 1)
            wa, wb = one / self.n(a), one / self.n(b)
            coef = []
            for rid in self.T.free[day]:
                lo, hi = self.box[rid]
                cap = self.u(rid) if rid in S else (hi - lo)
                g = self.group(a, b, rid)
                c = wb if g == "P" else (-wa if g == "M" else ZERO)
                coef.append((rid, cap, c))
            hit = (sorted(coef, key=lambda x: (-x[2], x[0])), sorted(coef, key=lambda x: (x[2], x[0])))
            self._ordcache[key] = hit
        return hit

    def _w_knapsack(self, a, b, day, S, t):
        """The same width, obtained instead by solving the two continuous knapsacks by sorting."""
        hi_order, lo_order = self._orders(a, b, day, S)
        return self._fill(hi_order, t) - self._fill(lo_order, t)

    @staticmethod
    def _fill(order, t):
        """Continuous knapsack: hand the surplus `t` to the runs in the given order until it runs out."""
        left, val = t, ZERO
        for _, cap, c in order:
            if left <= 0:
                break
            take = cap if cap < left else left
            val += take * c
            left -= take
        assert left <= 0 or all(x[1] == 0 for x in order), "surplus exceeds the day's capacity"
        return val

    def n(self, spec):
        return len(self._pop(spec))

    # -- the two public quantities -------------------------------------------------------------------

    def surplus_range(self, T, cs, rest):
        """What the day's remaining surplus can be once the recovered values are in, under this rule.

        The recovered runs absorb some amount sigma of the day's surplus; sigma is whatever their true
        values turn out to be, so it ranges over [max(0, T - rest), min(T, cs)] and the surplus left for
        the runs still unknown is T - sigma.  Every rule below is a different opinion about sigma.
        """
        lo = max(ZERO, T - cs)                 # sigma at most cs
        hi = min(T, rest)                      # what is left must fit in the remaining capacity
        assert lo <= hi, ("empty surplus range", T, cs, rest)
        how = self.rule["reveal"]
        if how in ("worst", "best"):
            return lo, hi
        if how == "fixed":                     # "the rest of the day is unchanged"
            t = min(T, rest)
            return t, t
        if how == "top":                       # recovered runs land at the top of their boxes
            t = max(ZERO, T - cs)
            return t, t
        if how == "mid":                       # ... or in the middle
            t = T - cs * HALF
            t = min(max(t, lo), hi)
            return t, t
        raise AssertionError(how)

    def _day_width(self, a, b, day, S, na, nb):
        Lp, cP, Lm, cM, Lo, cO, T, cs = self.parts(a, b, day, S)
        if cP == 0 and cM == 0:
            return ZERO                                 # no member of either cell is unknown here
        rest = cP + cM + cO
        tlo, thi = self.surplus_range(T, cs, rest)
        best = None
        for t in self._candidate_points(tlo, thi, cP, cM, cO):
            w = self._w_closed(t, cP, cM, cO, na, nb)
            k = self._w_knapsack(a, b, day, S, t)
            assert w == k, ("closed form and knapsack disagree", day, t, w, k)
            if best is None or (w > best if self.rule["reveal"] != "best" else w < best):
                best = w
        return best

    def width(self, a, b, S=()):
        """Guaranteed width of m(b) - m(a) once the values of `S` are revealed, worst case over what
        those values could be.  Days are independent, so the worst cases add."""
        S = set(S)
        na, nb = self.n(a), self.n(b)
        if not self.rule["coupling"]:
            total = ZERO
            for rid in sorted(self.unknown):
                g = self.group(a, b, rid)
                if g == "O":
                    continue
                lo, hi = self.box[rid]
                left = self.u(rid) if rid in S else (hi - lo)
                total += Fraction(left, 1) / (nb if g == "P" else na)
            return total
        if self.rule["reveal"] == "additive":
            base = Design.__new__(Design)
            base.__dict__.update(self.__dict__)
            base.rule = dict(REF)
            base._daycache = {}      # a different rule must not read this one's memo
            base._ordcache = self._ordcache   # orders do not depend on the rule
            total = base.width(a, b, ())
            for rid in sorted(S):
                g = self.group(a, b, rid)
                if g == "O":
                    continue
                lo, hi = self.box[rid]
                total -= Fraction((hi - lo) - self.u(rid), 1) / (nb if g == "P" else na)
            return max(ZERO, total)
        total = ZERO
        ka, kb = tuple(a[x] for x in KNOBS), tuple(b[x] for x in KNOBS)
        for day in sorted(self.T.free):
            # A day's contribution depends on S only through the part of S that finished that day, so the
            # cache key is cut to that part.  Plan search asks for thousands of subsets that differ off
            # this day, and they all land on the same entry.
            key = (ka, kb, day, frozenset(r for r in self.T.free[day] if r in S))
            hit = self._daycache.get(key)
            if hit is None:
                hit = self._day_width(a, b, day, S, na, nb)
                self._daycache[key] = hit
            total += hit
        return total

    @staticmethod
    def _candidate_points(tlo, thi, cP, cM, cO):
        """`_w_closed` is piecewise linear in t with breaks only at these four places, so its maximum on
        an interval is attained at an endpoint or at a break inside it."""
        pts = {tlo, thi}
        for br in (cP, cM, cM + cO, cP + cO):
            if tlo < br < thi:
                pts.add(br)
        return sorted(pts)

    def argmax_surplus(self, a, b, S):
        """For each day, the surplus that attains this rule's width - the input the witness needs."""
        S = set(S)
        na, nb = self.n(a), self.n(b)
        out = {}
        for day in sorted(self.T.free):
            Lp, cP, Lm, cM, Lo, cO, T, cs = self.parts(a, b, day, S)
            rest = cP + cM + cO
            tlo, thi = self.surplus_range(T, cs, rest)
            if cP == 0 and cM == 0:
                out[day] = (T if cs == 0 else thi, ZERO)
                continue
            best = None
            for t in self._candidate_points(tlo, thi, cP, cM, cO):
                w = self._w_closed(t, cP, cM, cO, na, nb)
                if best is None or w > best[1]:
                    best = (t, w)
            out[day] = (best[0], best[1])
        return out

    def voi(self, a, b, rid):
        """Does recovering this one run strictly narrow the guaranteed width?"""
        assert rid in self.unknown, ("%s is not an unrecorded launched run" % rid)
        return self.width(a, b, {rid}) < self.width(a, b)

    def plan(self, a, b, cands, target):
        """Smallest subset of `cands` whose recovery guarantees a width of at most `target`.

        Brute force over subsets in increasing size - the candidate lists are short and the point is that
        the optimum is *not* reachable by ranking candidates, so a greedy search would be wrong even
        though it is cheaper.  Every optimal subset is returned, because the verifier accepts any of them.
        """
        for k in range(0, len(cands) + 1):
            hits = []
            for sub in itertools.combinations(sorted(cands), k):
                if self.width(a, b, set(sub)) <= target:
                    hits.append(list(sub))
            if hits:
                return k, hits
        return None, []

    def greedy_plan(self, a, b, cands, target):
        """What a sensible-looking greedy agent would return: repeatedly add the candidate that narrows
        the guaranteed width most, ties broken by box width then id.  Recorded so the generator can check
        that it is *not* optimal - if greedy matched the optimum the item would not test search."""
        chosen, cur = [], self.width(a, b)
        pool = sorted(cands)
        while pool and cur > target:
            scored = []
            for rid in pool:
                w = self.width(a, b, set(chosen + [rid]))
                scored.append((w, -(self.box[rid][1] - self.box[rid][0]), rid))
            scored.sort()
            w, _, rid = scored[0]
            if w >= cur and chosen:
                break                                        # no further progress available
            chosen.append(rid)
            pool.remove(rid)
            cur = w
        return (len(chosen) if cur <= target else None), chosen, cur
