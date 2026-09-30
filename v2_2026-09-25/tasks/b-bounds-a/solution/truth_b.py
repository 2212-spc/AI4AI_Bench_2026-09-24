"""Second, independent implementation of the family-B estimand, used only to certify the key.

Written from the archive's own documents rather than from `bounds_b.py`: different parser (manual line
splitting instead of csv.DictReader), different clock (datetime instead of calendar.timegm), different
arithmetic (exact Fractions instead of floats), a per-run classification pass instead of a per-cell
accumulation, and - the part that matters - a different derivation of the day-constrained extremum.
`bounds_b.py` sorts the day's runs by coefficient and hands out the surplus greedily; this file uses the
closed forms

    max sum_A y = min( sum_A hi , R - sum_rest lo )        (a cell)
    max ( sum_P y / n_b - sum_M y / n_a )                  (a contrast)
        = (Lp + min(T, cP)) / n_b - (Lm + max(0, T - cP - cO)) / n_a,   T = R - Lp - Lm - Lo

which are derived by hand from the single equality constraint rather than computed by a search.  Two
implementations agreeing item by item rules out arithmetic slips; it does not prove the *rules* are the
ones the archive implies, which is why the generator separately checks the hidden values against the
published intervals and re-verifies a witness assignment against every constraint.
"""
import bisect
import datetime
import glob
import json
import os
from fractions import Fraction

KNOBS = ["suite", "curriculum", "retrieval", "decoder"]
ONE = Fraction(1)
ZERO = Fraction(0)


def rows(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","))) for ln in lines[1:]]


def epoch(s):
    return datetime.datetime.strptime(s.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc).timestamp()


class Truth(object):
    def __init__(self, app):
        self.plan = {r["run_id"]: r for r in rows(os.path.join(app, "plan.csv"))}
        self.started = set(r["run_id"] for r in rows(os.path.join(app, "launch_log.csv")))
        self.value = {}
        for r in rows(os.path.join(app, "eval_results.csv")):
            self.value[r["run_id"]] = Fraction(r["pass_rate"])
        for path in sorted(glob.glob(os.path.join(app, "recovered", "*.csv"))):
            for r in rows(path):
                v = Fraction(r["pass_rate_pct"]) / 100
                if r["run_id"] in self.value and self.value[r["run_id"]] != v:
                    raise AssertionError("recovered value contradicts eval_results for %s" % r["run_id"])
                self.value[r["run_id"]] = v
        self.dropped = {}
        for r in rows(os.path.join(app, "retention_log.csv")):
            self.dropped[r["run_id"]] = (r["reason"], epoch(r["dropped_at"]))
        pol = json.load(open(os.path.join(app, "retention_policy.json")))["versions"]
        self.pol = [(epoch(p["from"]), epoch(p["to"]), Fraction(str(p["floor"]))) for p in pol]
        self.pol_from = [a for a, _, _ in self.pol]
        su = json.load(open(os.path.join(app, "suites.json")))
        self.fl = {k: (ZERO if v["scoring"] == "normalized" else Fraction(str(v["random_guess"])))
                   for k, v in su.items()}
        self.fin = {r["run_id"]: epoch(r["finished_at"])
                    for r in rows(os.path.join(app, "runtime.csv"))}
        self.day = {r["run_id"]: r["finished_at"][:10]
                    for r in rows(os.path.join(app, "runtime.csv"))}

        # day -> ([unrecorded run ids], residual total those runs must add up to)
        self.free, resid = {}, {}
        for r in rows(os.path.join(app, "daily_digest.csv")):
            resid[r["date"]] = Fraction(r["sum_pass_rate"])
        for rid in sorted(self.started):
            d = self.day[rid]
            if rid in self.value:
                resid[d] -= self.value[rid]
            else:
                self.free.setdefault(d, []).append(rid)
        self.resid = resid

    def floor_at(self, t):
        """The floor in force at instant `t`, located by bisection rather than by scanning."""
        i = bisect.bisect_right(self.pol_from, t) - 1
        assert 0 <= i < len(self.pol) and t < self.pol[i][1], ("no policy version covers %s" % t)
        return self.pol[i][2]

    def support(self, rid):
        """(lo, hi) for one launched run, as exact rationals, dispatched on why the value is missing.

        Three reasons, three different shapes of evidence.  A row deleted for being under the floor is
        bounded *above* by that floor.  A row lost with its shard file had already been written, so it had
        already passed the screen on its finishing day: it is bounded *below*.  A run whose row the sink
        never received was never screened, and is bounded by nothing but the metric's own range.
        """
        if rid in self.value:
            return self.value[rid], self.value[rid]
        base = self.fl[self.plan[rid]["suite"]]
        why = self.dropped[rid][0] if rid in self.dropped else "never_written"
        if why == "below_retention_floor":
            return base, self.floor_at(self.dropped[rid][1]) - Fraction(1, 10000)
        if why == "shard_file_lost":
            f = self.floor_at(self.fin[rid])
            return (f if f > base else base), ONE
        return base, ONE

    def pop(self, spec):
        want = tuple(spec[k] for k in KNOBS)
        return [rid for rid, r in sorted(self.plan.items())
                if tuple(r[k] for k in KNOBS) == want and rid in self.started]

    def split(self, pop):
        """(n, sum of recorded values, the unrecorded members grouped by day)."""
        ksum, by_day = ZERO, {}
        for rid in pop:
            if rid in self.value:
                ksum += self.value[rid]
            else:
                by_day.setdefault(self.day[rid], []).append(rid)
        return len(pop), ksum, by_day

    def cell(self, spec):
        pop = self.pop(spec)
        if not pop:
            return None
        n, ksum, mine = self.split(pop)
        lo = hi = ksum
        for d, rids in mine.items():
            others = [r for r in self.free[d] if r not in set(rids)]
            sup = [self.support(r) for r in rids]
            osup = [self.support(r) for r in others]
            R = self.resid[d]
            hi += min(sum((s[1] for s in sup), ZERO), R - sum((s[0] for s in osup), ZERO))
            lo += max(sum((s[0] for s in sup), ZERO), R - sum((s[1] for s in osup), ZERO))
        return lo / n, hi / n

    def diff(self, a, b):
        """Sharp interval for m(b) - m(a), with both cells competing for the same daily totals."""
        pa, pb = self.pop(a), self.pop(b)
        na, ka, ma = self.split(pa)
        nb, kb, mb = self.split(pb)
        lo = hi = kb / nb - ka / na
        for d in sorted(set(ma) | set(mb)):
            P, M = mb.get(d, []), ma.get(d, [])
            taken = set(P) | set(M)
            O = [r for r in self.free[d] if r not in taken]
            Lp, Hp = [sum(x, ZERO) for x in zip(*[self.support(r) for r in P])] if P else (ZERO, ZERO)
            Lm, Hm = [sum(x, ZERO) for x in zip(*[self.support(r) for r in M])] if M else (ZERO, ZERO)
            Lo, Ho = [sum(x, ZERO) for x in zip(*[self.support(r) for r in O])] if O else (ZERO, ZERO)
            cP, cM, cO = Hp - Lp, Hm - Lm, Ho - Lo
            T = self.resid[d] - Lp - Lm - Lo
            if T < 0 or T > cP + cM + cO:
                raise AssertionError("day %s is infeasible: surplus %s of capacity %s" % (d, T, cP + cM + cO))
            up = min(T, cP)
            dm = max(ZERO, T - cP - cO)
            hi += (Lp + up) / nb - (Lm + dm) / na
            um = min(T, cM)
            dp = max(ZERO, T - cM - cO)
            lo += (Lp + dp) / nb - (Lm + um) / na
        return lo, hi


def solve(app, queries):
    T = Truth(app)
    out = {"cells": {}, "contrasts": {}, "decisions": {}}
    for q in queries["cells"]:
        lo, hi = T.cell(q["cell"])
        out["cells"][q["id"]] = {"lo": float(lo), "hi": float(hi)}
    for q in queries["contrasts"]:
        lo, hi = T.diff(q["from"], q["to"])
        out["contrasts"][q["id"]] = {"lo": float(lo), "hi": float(hi)}
    for q in queries["decisions"]:
        lo, hi = T.diff(q["from"], q["to"])
        m = Fraction(str(q["margin"]))
        out["decisions"][q["id"]] = "yes" if lo > m else ("no" if hi <= m else "cannot_tell")
    return out
