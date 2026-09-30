"""Reference bounds for family B, plus the library of plausible-but-wrong procedures.

Everything here reads only the files the agent is given.  There is no model and no sampling distribution,
so nothing can be certified by resampling: a configuration's mean over its launched runs is a linear
function of per-run values, and the surviving records constrain those values as a polytope.  The image of
a linear function over a polytope is a closed interval, so the truth is exact arithmetic - and *looseness*
becomes an error, because a wider interval is no longer the safe answer.

The polytope is what makes this more than bookkeeping.  Each unrecorded run is confined to its own
interval, but the scheduler's daily accounting also pins the *sum* of every value that finished on a given
day, so the runs of that day compete: pushing one configuration to its maximum forces the others down.
Treating the runs one at a time gives bounds that are valid and too wide, and the item selection is
required to punish that.

One thing this file does *not* claim.  "The cells compete, so a difference must be optimised jointly"
sounds right and is false here: with a single equality per day, the assignment that maximises one
configuration and the one that minimises a disjoint configuration can be chosen to coincide, so a
difference still combines crosswise from two separately optimised intervals.  That was written as a decoy
first, measured against the reference, found to disagree on nothing, and moved to `EQUIVALENT` - where
`gen_b.py` now asserts it stays equal.  The rule of the house is that a decoy earns its place by failing
items, and a procedure that fails none of them is not a mistake, whatever the author expected.

Each entry of `CANDIDATES` flips exactly one rule to a mistake a competent analyst actually makes: the
daily totals ignored or misread, the day taken from the start time instead of the finish time, the wrong
denominator, the recovery shards ignored or trusted too far, percent read as a fraction, one global
retention floor instead of the one in force, the half-open policy interval closed the wrong way, an
operator deletion mistaken for a quality deletion, a difference formed from separate intervals, a decision
read off the midpoint.  `gen_b.py` then *picks the queries* so that every one of these misses at least two
items - the items are chosen to kill the decoys rather than the decoys chosen to fit the items.
"""
import calendar
import csv
import glob
import json
import os
import time

KNOBS = ["suite", "curriculum", "retrieval", "decoder"]
REF = {}
EPS = 1e-12
GRID = 1e-4                                       # every published `pass_rate` is a multiple of this


def iso(s):
    return calendar.timegm(time.strptime(s.strip(), "%Y-%m-%dT%H:%M:%SZ"))


class Arch(object):
    """The published archive, parsed once."""

    def __init__(self, app):
        self.app = app
        self.plan = list(csv.DictReader(open(os.path.join(app, "plan.csv"))))
        self.launch = list(csv.DictReader(open(os.path.join(app, "launch_log.csv"))))
        self.results = list(csv.DictReader(open(os.path.join(app, "eval_results.csv"))))
        self.retention = list(csv.DictReader(open(os.path.join(app, "retention_log.csv"))))
        self.runtime = list(csv.DictReader(open(os.path.join(app, "runtime.csv"))))
        self.digest = list(csv.DictReader(open(os.path.join(app, "daily_digest.csv"))))
        self.policy = json.load(open(os.path.join(app, "retention_policy.json")))["versions"]
        self.suites = json.load(open(os.path.join(app, "suites.json")))
        self.recovered = []
        for p in sorted(glob.glob(os.path.join(app, "recovered", "*.csv"))):
            self.recovered += list(csv.DictReader(open(p)))
        self.cell_of = {r["run_id"]: tuple(r[k] for k in KNOBS) for r in self.plan}
        self.suite_of = {r["run_id"]: r["suite"] for r in self.plan}
        self.launched = {r["run_id"] for r in self.launch}
        self.started_at = {r["run_id"]: r["started_at"] for r in self.launch}
        self.finished_at = {r["run_id"]: r["finished_at"] for r in self.runtime}
        self.res = {r["run_id"]: float(r["pass_rate"]) for r in self.results}
        self.drop = {r["run_id"]: r for r in self.retention}
        self.rec = {}
        for r in self.recovered:
            self.rec[r["run_id"]] = float(r["pass_rate_pct"])

    # ---- the rules, each with a switch exactly one decoy flips -----------------------------------
    def population(self, cell, spec):
        if spec.get("denominator") == "plan":
            return [r["run_id"] for r in self.plan if tuple(r[k] for k in KNOBS) == cell]
        pop = [r["run_id"] for r in self.plan
               if tuple(r[k] for k in KNOBS) == cell and r["run_id"] in self.launched]
        if spec.get("denominator") == "plus_recovered_ghosts":
            extra = [r["run_id"] for r in self.recovered
                     if self.cell_of.get(r["run_id"]) == cell and r["run_id"] not in self.launched]
            pop = pop + sorted(set(extra))
        if spec.get("denominator") == "rows_not_runs":
            dup = [r["run_id"] for r in self.recovered if self.cell_of.get(r["run_id"]) == cell
                   and r["run_id"] in self.launched]
            pop = pop + dup                                   # duplicate recovery rows counted twice
        return pop

    def known(self, rid, spec):
        if rid in self.res:
            return self.res[rid]
        if spec.get("ignore_recovered") or rid not in self.rec:
            return None
        v = self.rec[rid]
        return v if spec.get("pct_is_fraction") else v / 100.0

    def floor(self, suite, spec):
        sp = self.suites[suite]
        if spec.get("floor") == "always_guess":
            return float(sp["random_guess"])
        if spec.get("floor") == "always_zero":
            return 0.0
        return 0.0 if sp["scoring"] == "normalized" else float(sp["random_guess"])

    def _pol(self, t, spec):
        """The quality floor in force at instant `t`, under the published half-open convention."""
        closed_left = spec.get("retention") != "closed_right"
        for p in self.policy:
            a, b = iso(p["from"]), iso(p["to"])
            if (a <= t < b) if closed_left else (a < t <= b):
                return float(p["floor"])
        return 0.0

    def lower(self, rid, spec):
        """Lower end of a run's support.

        The archiver screens each row against the floor in force when the row is written, so a row that
        *existed* is evidence its value cleared that floor.  For a run whose row was later lost with the
        shard file, that evidence survives the row: the value is at least the floor of its finishing day.
        A run whose row never reached the archiver was never screened and gets no such bound, which is
        why the rule keys on the reason and not on the mere fact of being unrecorded.
        """
        base = self.floor(self.suite_of[rid], spec)
        mode = spec.get("survivorship")
        if mode == "ignore":
            return base
        e = self.drop.get(rid)
        if e is not None and e["reason"] == "shard_file_lost":
            t = iso(e["dropped_at"] if mode == "at_deletion" else self.finished_at[rid])
        elif e is None and mode == "all_launched":
            t = iso(self.finished_at[rid])
        else:
            return base
        return max(base, self._pol(t, spec))

    def ceiling(self, rid, spec):
        """Upper end of a run's support: the retention floor in force iff the row was deleted *for being
        below it*; losing the file a row was stored in says nothing about the value it held.

        The screen removes a row that is *strictly* under the floor, and every `pass_rate` is a multiple
        of 0.0001, so the largest value a removed row could have held is one grid step below the floor.
        The step is far smaller than the published tolerance - `WITHIN_TOL` records that both readings
        grade the same - but the key is the sharp one.
        """
        e = self.drop.get(rid)
        if e is None or spec.get("retention") == "no_bound":
            return 1.0
        if e["reason"] != "below_retention_floor" and not spec.get("any_reason_bounds"):
            return 1.0
        if spec.get("retention") == "latest_floor":
            return float(self.policy[-1]["floor"])
        f = self._pol(iso(e["dropped_at"]), spec)
        return f if spec.get("grid") == "ignore" else round(f - GRID, 6)

    def day_of(self, rid, spec):
        if spec.get("day_key") == "started":
            return self.started_at.get(rid, "")[:10]
        return self.finished_at.get(rid, "")[:10]


def allocate(items, R, maximize):
    """Extremum of `sum_i c_i y_i` over one day, subject to `sum_i y_i = R` and `lo_i <= y_i <= hi_i`.

    Every variable starts at its lower end; the surplus `R - sum lo` is then handed out to the most
    valuable coefficients first (least valuable first, when minimising).  Because the constraint is a
    single equality, that greedy pass is exactly optimal, and it also hands back a witness - the full
    assignment that attains the extremum, which `gen_b.py` re-checks against every published constraint.

    Wrong rules can make the day's total unreachable; rather than crash on a decoy the surplus is clamped
    and counted, so an infeasible decoy scores as the nonsense it is while the reference path is asserted
    to be clamp-free.
    """
    y = [lo for _, lo, _ in items]
    T = R - sum(y)
    clamped = 0.0
    if T < -EPS:
        clamped, T = -T, 0.0
    order = sorted(range(len(items)), key=lambda i: items[i][0], reverse=maximize)
    for i in order:
        if T <= EPS:
            break
        room = items[i][2] - items[i][1]
        a = room if room < T else T
        y[i] += a
        T -= a
    if T > EPS:
        clamped += T
    return sum(items[i][0] * y[i] for i in range(len(items))), y, clamped


class Ctx(object):
    """One archive read under one rule set: per-run supports, and the day constraints they must satisfy."""

    def __init__(self, arch, spec):
        self.arch, self.spec = arch, spec
        self.sup, self.val, self.day = {}, {}, {}
        self.unk = {}
        self.clamped = 0.0
        self._c, self._d = {}, {}
        for rid in sorted(arch.launched):
            v = arch.known(rid, spec)
            self.val[rid] = v
            self.sup[rid] = ((v, v) if v is not None else
                             (arch.lower(rid, spec), arch.ceiling(rid, spec)))
            self.day[rid] = arch.day_of(rid, spec)
        self.R = {}
        if spec.get("digest") != "ignore":
            for rid in sorted(arch.launched):
                if self.val[rid] is None:
                    self.unk.setdefault(self.day[rid], []).append(rid)
            for row in arch.digest:
                d = row["date"]
                tot = float(row["sum_pass_rate"])
                if spec.get("digest") == "mean_as_total":
                    tot *= float(row["n_runs"])
                self.R[d] = tot
            for rid in sorted(arch.launched):
                if self.val[rid] is not None and self.day[rid] in self.R:
                    self.R[self.day[rid]] -= self.val[rid]

    def extremum(self, weight, maximize):
        """Extremum of `sum_rid weight[rid] * y_rid` over the unrecorded launched runs.

        Without the daily totals the runs are independent and each goes to whichever end its own
        coefficient prefers.  With them, every run that finished on a touched day enters the day's
        problem - including the ones with coefficient zero, which are the slack that lets the others
        move.  Days are separate constraints, so their extrema simply add.
        """
        if not self.R:
            return sum((w * self.sup[r][1] if (w > 0) == maximize else w * self.sup[r][0])
                       for r, w in weight.items())
        total = 0.0
        for d in sorted({self.day[r] for r in weight}):
            rids = self.unk.get(d, [])
            items = [(weight.get(r, 0.0), self.sup[r][0], self.sup[r][1]) for r in rids]
            v, _, c = allocate(items, self.R.get(d, 0.0), maximize)
            total += v
            self.clamped += c
        return total

    def witness(self, weight, maximize):
        """The full assignment attaining `extremum`, for the sharpness gate."""
        y = dict((r, v) for r, v in self.val.items() if v is not None)
        for d in sorted(self.unk):
            rids = self.unk[d]
            items = [(weight.get(r, 0.0), self.sup[r][0], self.sup[r][1]) for r in rids]
            _, vals, _ = allocate(items, self.R.get(d, 0.0), maximize)
            for r, v in zip(rids, vals):
                y[r] = v
        return y

    # ---- the estimand -------------------------------------------------------------------------
    def parts(self, cell):
        """(n, known sum, weights on unrecorded launched runs, supports of unlaunched members)."""
        pop = self.arch.population(cell, self.spec)
        ksum, weight, free = 0.0, {}, []
        for rid in pop:
            if rid in self.sup:
                v = self.val[rid]
                if v is None:
                    weight[rid] = weight.get(rid, 0.0) + 1.0
                else:
                    ksum += v
            else:                                   # planned but never launched: no day, no constraint
                v = self.arch.known(rid, self.spec)
                if v is None:
                    free.append((self.arch.floor(self.arch.suite_of[rid], self.spec), 1.0))
                else:
                    ksum += v
        return len(pop), ksum, weight, free

    def point(self, n, ksum, weight, free):
        """The two rules that refuse to report an interval at all."""
        if self.spec.get("complete_case"):
            k = n - len(weight) - len(free)
            return (ksum / k, ksum / k) if k else None
        fl = sum(self.sup[r][0] * w for r, w in weight.items()) + sum(f[0] for f in free)
        return (ksum + fl) / n, (ksum + fl) / n

    def cell(self, cell):
        if cell in self._c:
            return self._c[cell]
        self._c[cell] = v = self._cell(cell)
        return v

    def _cell(self, cell):
        n, ksum, weight, free = self.parts(cell)
        if n == 0:
            return None
        if self.spec.get("complete_case") or self.spec.get("impute_at_floor"):
            return self.point(n, ksum, weight, free)
        lo = (ksum + self.extremum(weight, False) + sum(f[0] for f in free)) / n
        hi = (ksum + self.extremum(weight, True) + sum(f[1] for f in free)) / n
        return lo, hi

    def diff(self, ca, cb):
        """Sharp interval for m(cb) - m(ca).

        The two configurations share no runs, but they do share days, so the difference is *not* the
        crosswise difference of their separate intervals: raising one cell's runs consumes the same
        daily budget the other cell's runs would need.  The coefficients go into one optimisation.
        """
        if (ca, cb) in self._d:
            return self._d[(ca, cb)]
        self._d[(ca, cb)] = v = self._diff(ca, cb)
        return v

    def _diff(self, ca, cb):
        mode = self.spec.get("contrast")
        if self.spec.get("complete_case") or self.spec.get("impute_at_floor"):
            mode = "independent_cells"
        if mode in ("pairwise", "independent_cells"):
            a, b = self.cell(ca), self.cell(cb)
            if mode == "pairwise":
                return b[0] - a[0], b[1] - a[1]
            return b[0] - a[1], b[1] - a[0]
        na, ka, wa, fa = self.parts(ca)
        nb, kb, wb, fb = self.parts(cb)
        base = kb / nb - ka / na
        w = {}
        for r, x in wb.items():
            w[r] = w.get(r, 0.0) + x / nb
        for r, x in wa.items():
            w[r] = w.get(r, 0.0) - x / na
        hi = base + self.extremum(w, True) + sum(f[1] for f in fb) / nb - sum(f[0] for f in fa) / na
        lo = base + self.extremum(w, False) + sum(f[0] for f in fb) / nb - sum(f[1] for f in fa) / na
        return lo, hi


def decide(d, margin, spec):
    lo, hi = d
    mode = spec.get("decision")
    if mode == "midpoint":
        return "yes" if 0.5 * (lo + hi) > margin else "no"
    if mode == "any_feasible":
        return "yes" if hi > margin else "no"
    if lo > margin:
        return "yes"
    if hi <= margin:
        return "no"
    return "cannot_tell"


def answer(arch, queries, spec=None):
    spec = spec or REF
    ctx = Ctx(arch, spec)
    out = {"cells": {}, "contrasts": {}, "decisions": {}}
    cache = {}

    def D(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = ctx.diff(a, b)
        return cache[(a, b)]

    def key(q, f):
        return tuple(q[f][k] for k in KNOBS)

    for q in queries["cells"]:
        lo, hi = ctx.cell(key(q, "cell"))
        out["cells"][q["id"]] = {"lo": round(lo, 6), "hi": round(hi, 6)}
    for q in queries["contrasts"]:
        lo, hi = D(key(q, "from"), key(q, "to"))
        out["contrasts"][q["id"]] = {"lo": round(lo, 6), "hi": round(hi, 6)}
    for q in queries["decisions"]:
        out["decisions"][q["id"]] = decide(D(key(q, "from"), key(q, "to")), q["margin"], spec)
    out["_clamped"] = ctx.clamped
    return out


CANDIDATES = {
    # the daily accounting: ignored, misread, or keyed on the wrong timestamp
    "daily_totals_ignored_runs_bounded_one_at_a_time": {"digest": "ignore"},
    "daily_total_read_as_a_daily_mean": {"digest": "mean_as_total"},
    "day_taken_from_the_start_time": {"day_key": "started"},
    # no interval at all: the two ways an analyst turns a partially identified quantity into a point
    "complete_case_point_estimate": {"complete_case": True},
    "unknown_runs_imputed_at_the_floor": {"impute_at_floor": True},
    # wrong population
    "denominator_counts_planned_runs": {"denominator": "plan"},
    "denominator_counts_recovery_ghosts": {"denominator": "plus_recovered_ghosts"},
    "duplicate_recovery_rows_counted_twice": {"denominator": "rows_not_runs"},
    # wrong handling of the secondary sink
    "recovery_shards_ignored": {"ignore_recovered": True},
    "percent_read_as_fraction": {"pct_is_fraction": True},
    # wrong support
    "floor_is_random_guess_everywhere": {"floor": "always_guess"},
    "floor_is_zero_everywhere": {"floor": "always_zero"},
    # wrong retention bound
    "one_global_retention_floor": {"retention": "latest_floor"},
    "retention_bound_discarded": {"retention": "no_bound"},
    "policy_interval_closed_on_the_right": {"retention": "closed_right"},
    "lost_shard_treated_as_a_quality_deletion": {"any_reason_bounds": True},
    # the survivorship deduction: skipped, over-applied, or read at the wrong instant.  The first of these
    # is the whole family in one line - it is the answer an analyst gets by treating "the row is gone" as
    # "the value is unconstrained", which is right for the outage and wrong for the lost shard.
    "surviving_row_evidence_ignored": {"survivorship": "ignore"},
    "survivorship_applied_to_every_unrecorded_run": {"survivorship": "all_launched"},
    # wrong algebra on top of right bounds
    "contrast_endpoints_subtracted_pairwise": {"contrast": "pairwise"},
    "decision_read_off_the_midpoint": {"decision": "midpoint"},
    "decision_yes_if_any_feasible_value_clears": {"decision": "any_feasible"},
}

# Procedures that look like mistakes and are not.  `gen_b.py` requires these to agree with the reference
# on every published item, which turns each one into a standing check on the reference itself: if a later
# change to the world made the crosswise shortcut wrong, this gate would fail rather than pass quietly.
EQUIVALENT = {
    "contrast_from_two_independent_cell_intervals": {"contrast": "independent_cells"},
    # Reading the survivorship floor at the instant the shard was lost instead of at the instant the row
    # was written is a real misreading, and in *this* archive it happens to be harmless: both instants
    # fall inside one policy version.  Punishing it would be punishing a distinction the published files
    # cannot make, so it ships as an equivalence - and if a future world ever moves the lost shard across
    # a policy boundary, the gate below turns it back into a decoy by failing.
    "survivorship_floor_read_at_the_deletion_timestamp": {"survivorship": "at_deletion"},
}

# Family B has no "different but equally valid estimator": the identified set is unique, so the role
# ALTERNATIVES played in family S is taken here by `truth_b.py`, a second, independent implementation
# written from the published records alone, which `gen_b.py` requires to agree item by item.
ALTERNATIVES = {}

# Readings that differ from the reference by less than the grader can see.  They are not equivalences -
# the numbers really do differ - and they are not decoys, because no item could ever separate them at the
# published tolerance.  Shipping them in their own category is the honest option: the gate pins how much
# room is left between the two readings, so a future change that widens the gap turns into a failure
# instead of an unfair item.
WITHIN_TOL = {
    "screened_row_bounded_by_the_floor_itself": {"grid": "ignore"},
}
