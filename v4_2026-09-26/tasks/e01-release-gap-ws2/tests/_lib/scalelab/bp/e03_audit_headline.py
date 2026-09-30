"""Audit a teammate's analysis script: one silent defect, three provably harmless red herrings.

The team has already spent its measurement budget.  What is shipped is the evidence they collected (an
exact corpus scan, a cheap detector table, repeated per-item scoring passes for two models) plus the
script that turned it into the number in the review memo.  Exactly one step of that script makes a wrong
inference.  It runs, it prints a plausible number, no test fails and no log line is anomalous.

Why the keys are exact.  Every question here is a question about *this evidence*: what the shipped rows
imply once the defective step is fixed.  So the key is a deterministic function of the shipped bytes, not
an estimate of the world, and the oracle reproduces it with zero error.  The difficulty is entirely
inferential - find the step, redo the aggregation, and carry a known unknown through to a release
decision - which is the point of the form: a defect that changes no test outcome is invisible to a
test-oracle benchmark, and nothing here crashes.

Which defect is injected is drawn per instance from three of the taxonomy's entries, each realised as a
one-line change in the same script.  The other two ways of being wrong are present in the *world* (a
cheap detector that misses most contamination, an extractor that drops a fifth of one model's answers),
so a naive analyst has three separate chances to produce the wrong number and the injected defect is not
guessable from which mechanisms are active.

Exactly-one-defect check (the audit answer is a closed-set choice, so ambiguity is unfairness).  Against
the nine taxonomy entries, as shipped: D1/D2/D7 are the injected variants, one per instance, and the
other two of the three are correct in that instance.  D3 cannot be injected here (this bank's slices are
drawn uniformly, so micro and macro differ by about 0.002, well under the accepted width) and the script
uses the item-weighted form.  D4 needs a judge; there is none.  D5 needs a reported standard error or
interval; the script reports none.  D6 needs unequal denominators across the averaged units; every
repetition presents the same 1500 items, so mean-of-ratios and ratio-of-means agree to 1e-5.  D8 needs a
threshold-crossing claim; the script makes none (the D7 variant thresholds a noisy score to *filter*,
and the exact scan it ignores is loaded and printed two lines above, which is D7's own text).  D9 needs
failed runs; this lab has none.
"""
import io
import json
import math

import numpy as np

from ..common import draw_card, run_rows
from ..labs import evallab as EV
from ..lab import Session
from .. import queries as Q
from ..defects import DEFECTS

ID = "e03-audit-headline"
TITLE = "Audit the release headline"
CARDS = ["E1", "E2", "E3", "E6", "E7", "E8"]
OBSTACLES = ["O14", "O15", "O4", "O16", "O10", "O5"]
CLAIM = ("The number in the review memo is what the team's own evidence implies, and the release bar is "
         "cleared.")
DIFFICULTY = {"depth": 4, "nuisance": ["E2", "E3", "E6", "E7", "E8"], "anti_prior": ["a2"]}
# Three of the six cards are not separable by a drop, for a reason worth stating: the keys are defined on
# a *subset* and over *repetitions* that those very cards create, so switching the card off leaves the
# queried quantity unchanged while still being indispensable to getting it right.  G12 is the right
# instrument for that shape - `naive_ignore:<card>` leaves the mechanism on and fails to model it - and
# each exemption below names the G12 rival that carries the certification.
EXEMPT_LOAD_BEARING = {
    "E1": "E1 is the item bank itself - the substrate every other card acts on, not a separable effect: "
          "neutralising it removes the population the questions are about rather than one mechanism.",
    "E3": "E3 (corpus duplication) defines the clean subset the headline is about.  Setting kappa to zero "
          "removes the memorisation *premium* but not the duplicated items, so the clean-subset key is "
          "unchanged to the last digit - a drop cannot certify this card, and pretending otherwise would "
          "be a noise kill.  Certified by G12 instead: `naive_ignore:E3`, the analyst who never excludes "
          "contaminated items, is killed with margin (the premium is screened at draw time to exceed "
          "0.05, more than ten tolerances).",
    "E6": "E6 (the between-call shift) is why handling repetitions is an inferential choice rather than "
          "bookkeeping, and it is what the D1 variant exploits.  Dropping it perturbs every shipped row, "
          "so `drop:E6` would be killed by resampling noise rather than by structure.  Certified by G12: "
          "`naive_ignore:E6`, the analyst who quotes one pass instead of averaging the six, is killed "
          "with margin.",
    "E8": "E8 is the decoy mechanism.  The cheap detector's scores are shipped and the memo argues from "
          "them, but the keys are computed from the exact scan, so a world without the cheap detector has "
          "the same answers.  Certified by G12: `naive_ignore:E8`, the analyst who trusts the detector, "
          "is killed with margin on every instance, not only on the ones where it is the injected defect.",
}
CERT_N = 250

N_ITEMS, N_SLICES = 1500, 3
NEW, REF = "M_new", "M_ref"
FMT, FMT2 = "mc_letter", "mc_cloze"
MODELS = (NEW, REF)
W0 = (0.15, 0.85)
W_GRID = 41
MODEL_COST = 0.05
SCAN_COST, FAST_COST = 1.2, 0.03
DET_THRESH = 0.5
R_NB = 6                          # scoring passes per model in the shipped notebook
MAX_WIDTH = 0.003                 # the accepted width on a recomputed number (it is arithmetic, not a measurement)
HALF = 0.0015

SCRIPT = "analysis/headline.py"
INJECTABLE = ("D1_selection_max_over_seeds", "D2_denominator_scored_only",
              "D7_contaminated_items_included")
HERRINGS = ("hardcoded_n", "round6", "sort_seeds")


# ------------------------------------------------------------------------------------------ world
def _set(p, path, v):
    cur = p
    parts = path.split(".")
    for k in parts[:-1]:
        cur = cur[k]
    cur[parts[-1]] = v
    return p


def _centred(rng, sd, n):
    x = rng.normal(0.0, sd, n)
    return [float(v) for v in (x - x.mean())]


_C = np.array([2.0, -1.0, -1.0]) / math.sqrt(2.0)
_D = np.array([0.0, 1.0, -1.0]) / math.sqrt(2.0 / 3)


def _profile(sd, sign, phi):
    return [float(x) for x in sd * (math.cos(phi) * sign * _C + math.sin(phi) * _D)]


def draw(rng):
    """The same mechanism set as e01 with one extra requirement: the per-slice spread has to be wide
    enough that the release weighting moves the headline by more than the accepted width, or a2 is a point
    and the release bar cannot be straddled."""
    why = "no draw attempted"
    for _ in range(300):
        p = {"n_items": N_ITEMS, "n_slices": N_SLICES, "bank_seed": int(rng.integers(1, 10 ** 6)),
             "det_fast_cost": FAST_COST, "det_scan_cost": SCAN_COST}
        p.update(draw_card(rng, "E1", {"b_mu": (-0.35, 0.35)}))
        e2 = draw_card(rng, "E2", {"ext_base": (0.07, 0.13), "ext_slice_sd": (0.02, 0.06),
                                   "ext_model_sd": (0.10, 0.15), "fmt_off_sd": (0.10, 0.35)})
        e3 = draw_card(rng, "E3", {"dup_frac": (0.22, 0.34), "dup_lam": (4.0, 9.0), "kappa": (0.65, 1.15)})
        e6 = draw_card(rng, "E6", {"sig_call": (0.14, 0.20)})
        e7 = draw_card(rng, "E7", {"slice_b_sd": (0.45, 0.70), "slice_off_sd": (0.25, 0.45)})
        p.update(draw_card(rng, "E8", {"det_a": (0.6, 1.1), "det_b": (-3.2, -2.5), "det_s": (0.8, 1.3)}))
        p["dup_frac"], p["dup_lam"] = e3["dup_frac"], e3["dup_lam"]
        p["sig_call"] = e6["sig_call"]
        p["slice_b"] = _profile(e7["slice_b_sd"], 1.0 if rng.random() < 0.5 else -1.0, rng.uniform(0.0, 0.55))
        p["fmt_ext"] = {}
        for f, scale in ((FMT, 1.0), (FMT2, float(rng.uniform(0.3, 0.8)))):
            base = e2["ext_base"] * scale
            off = _centred(rng, e2["ext_slice_sd"] * scale, N_SLICES)
            p["fmt_ext"][f] = {"base": base, "by_slice": [max(d, 0.02 - base) for d in off]}
        th = p["b_mu"] + float(rng.uniform(-0.15, 0.35))
        p["models"] = {
            NEW: dict(theta=th, kappa=e3["kappa"], ext=e2["ext_model_sd"], cost=MODEL_COST, length=300.0,
                      fmt={FMT: 0.0, FMT2: float(rng.normal(0.0, e2["fmt_off_sd"]))},
                      slice_off=_profile(e7["slice_off_sd"], -1.0, rng.uniform(0.0, 0.55))),
            REF: dict(theta=th - float(rng.uniform(0.15, 0.45)), kappa=0.0, ext=0.0, cost=MODEL_COST,
                      length=300.0, fmt={FMT: 0.0, FMT2: float(rng.normal(0.0, e2["fmt_off_sd"]))},
                      slice_off=_profile(e7["slice_off_sd"], +1.0, rng.uniform(0.0, 0.55)))}
        p["w0_rel"] = float(rng.uniform(*W0))
        p["defect"] = INJECTABLE[int(rng.integers(len(INJECTABLE)))]
        ok, why = _analytic_ok(EV.full(p))
        if ok:
            return p
    raise RuntimeError("no admissible draw (last: %s)" % why)


def _analytic_ok(pf):
    """Draw-time screening on the world's exact quantities.  The binding conditions are on the *shipped
    bytes* and are checked in `wellposed` once the notebook exists; this only rejects worlds that cannot
    possibly satisfy them, so that the expensive path runs on plausible candidates."""
    idx = np.arange(int(pf["n_items"]))
    _a, _b, g, dup = EV.bank(pf)
    clean = dup == 0
    ok = EV.extract_ok(pf, NEW, FMT, idx)
    pc = np.zeros(len(idx))
    for x, wt in zip(EV._GH_X, EV._GH_W):
        pc += wt * EV.p_correct(pf, NEW, FMT, idx, dtheta=pf["sig_call"] * x)
    rel = [float(pc[clean & (g == s) & ok].sum()) / max(int((clean & (g == s)).sum()), 1)
           for s in range(N_SLICES)]
    mix = [_mix(rel, w) for w in _grid()]
    ext = 1.0 - float(ok.mean())
    prem = (float(pc[ok].sum()) / len(idx)) - (float(pc[clean & ok].sum()) / max(int(clean.sum()), 1))
    checks = [(0.28 <= _mix(rel, 0.5) <= 0.80, "clean accuracy %.3f outside [0.28, 0.80]" % _mix(rel, 0.5)),
              (max(mix) - min(mix) >= 0.045, "slice spread %.4f too small for a2" % (max(mix) - min(mix))),
              (ext >= 0.16, "extraction failures %.3f too rare for the D2 variant to bite" % ext),
              (abs(prem) >= 0.05, "contamination premium %+.4f too small for the D7 variant" % prem)]
    for good, why in checks:
        if not good:
            return False, why
    return True, "clean %.4f; slice spread %.4f; extraction %.3f; premium %+.4f" % (
        _mix(rel, 0.5), max(mix) - min(mix), ext, prem)


def spec(p):
    return {"lab": "evallab",
            "knobs": {"svc": {"type": "choice", "values": ["score", "corpus"], "default": "score"},
                      "model": {"type": "choice", "values": list(MODELS), "default": NEW},
                      "split": {"type": "choice", "values": ["all", "s0", "s1", "s2"], "default": "all"},
                      "fmt": {"type": "choice", "values": [FMT, FMT2], "default": FMT},
                      "n": {"type": "float", "min": 10, "max": N_ITEMS, "int": True},
                      "reps": {"type": "float", "min": 1, "max": 400, "default": 1, "int": True},
                      "detail": {"type": "choice", "values": [0, 1], "default": 0},
                      "mode": {"type": "choice", "values": ["fast", "scan"], "default": "fast"}},
            "n_items": N_ITEMS, "n_slices": N_SLICES, "models": {m: dict(p["models"][m]) for m in MODELS},
            # The team has already spent most of its budget producing what is in the notebook.  What is
            # left is enough to re-measure a piece of it, not to redo the campaign: the questions are about
            # the evidence in hand, and the remaining budget is there for checking, not for a new design.
            "caps": {"run_cost": 4000.0, "total_cost": 6000.0, "max_runs": 12}}


def LAB_EXTRA(w):
    return ["", "The team's own passes are in `notebook/runs.jsonl` (the per-item tables the memo cites are "
                "there in full), the exact corpus scan in `notebook/scan_exact.csv`, the cheap detector's "
                "scores in `notebook/detector_fast.csv`, and the script that produced the memo's headline "
                "in `%s`." % SCRIPT,
            "", "A `score` request with `detail=1` returns a per-item table (`item`, `slice`, `scored`, "
                "`correct`) for its first repetition, which is what the notebook's passes recorded."]


def known_unknowns(p):
    return [{"name": "release weight on slice 0", "param": "w0_rel", "range": list(W0),
             "text": "The release report aggregates the three slices with weight w0 on slice 0 and the "
                     "remaining 1 - w0 split 2:1 between slices 1 and 2.  Product has not fixed w0; it is "
                     "known only to lie in [%g, %g], and no request to this service depends on that choice. "
                     "Questions about the release report must cover the whole range." % W0}]


# ------------------------------------------------------------------------------------------ weights
def _weights(w0):
    return [w0, (1.0 - w0) * 2.0 / 3.0, (1.0 - w0) / 3.0]


def _mix(per_slice, w0):
    return float(sum(w * a for w, a in zip(_weights(w0), per_slice)))


def _grid():
    return list(np.linspace(W0[0], W0[1], W_GRID))


# ------------------------------------------------------------------------------- the audited script
def _script_lines(defect, fixed_herrings=()):
    """The shipped script, as a list of lines, plus the 1-based line number of the defective step.

    One generator for every variant: the shipped text, the text with the defect repaired, and the text
    with a red herring "repaired" instead.  G16 runs all of them and compares the numbers they print, so
    the variants cannot drift from the shipped bytes - they are the shipped bytes with one line swapped.
    """
    fh = set(fixed_herrings)
    L = []
    site = None

    def add(*lines):
        L.extend(lines)

    def mark(line):
        nonlocal site
        site = len(L) + 1
        L.append(line)

    add('#!/usr/bin/env python3',
        '"""Headline number for the release review.',
        '',
        'Reads what we collected in notebook/ and prints the clean-subset accuracy of %s that the' % NEW,
        'review memo quotes.  Nothing here calls the lab; the passes are already on disk.',
        '',
        '    python3 %s' % SCRIPT,
        '"""',
        'import csv',
        'import json',
        '',
        'RUNS = "notebook/runs.jsonl"',
        'DETECTOR = "notebook/detector_fast.csv"',
        'SCAN = "notebook/scan_exact.csv"',
        'MODEL = "%s"' % NEW,
        'DET_THRESH = %g       # the cheap detector score above which the memo calls an item suspect' % DET_THRESH)
    if "hardcoded_n" in fh:
        add('')
    else:
        add('N_BANK = %d          # size of the item bank' % N_ITEMS)
    add('',
        '',
        'def load_rows(path):',
        '    """Every lab row we have, one JSON object per line."""',
        '    rows = []',
        '    for line in open(path):',
        '        line = line.strip()',
        '        if line:',
        '            rows.append(json.loads(line))',
        '    return rows',
        '',
        '',
        'def detail_passes(rows, model):',
        '    """The scoring passes that recorded a per-item table."""',
        '    out = [r for r in rows if r.get("svc") == "score" and r.get("model") == model',
        '           and r.get("fmt") == "%s" and r.get("detail")]' % FMT)
    if "sort_seeds" not in fh:
        add('    out.sort(key=lambda r: r["config"]["seed"])')
    add('    return out',
        '',
        '',
        'def contaminated(n_bank):',
        '    """Which bank items are in the pretraining corpus.  We have both the exact scan and the cheap',
        '    detector\'s scores, so print both counts before deciding which set to exclude."""',
        '    exact = set()',
        '    for row in csv.DictReader(open(SCAN)):',
        '        if int(row["occurrences"]) > 0:',
        '            exact.add(int(row["item"]))',
        '    cheap = set()',
        '    for row in csv.DictReader(open(DETECTOR)):',
        '        if float(row["overlap"]) >= DET_THRESH:',
        '            cheap.add(int(row["item"]))',
        '    print("contamination: exact scan %d of %d items, cheap detector flags %d"',
        '          % (len(exact), n_bank, len(cheap)))')
    if defect == "D7_contaminated_items_included":
        mark('    return cheap        # the scan is a sample of the corpus; the detector covers the whole bank')
    else:
        add('    return exact')
    add('',
        '',
        'def per_slice_counts(passes, dirty):',
        '    """For each pass, per slice: items presented, answers extracted, answers correct, over the',
        '    items that are not contaminated."""',
        '    out = []',
        '    for r in passes:',
        '        acc = {}',
        '        for x in r["detail"]:',
        '            if x["item"] in dirty:',
        '                continue',
        '            c = acc.setdefault(x["slice"], [0, 0, 0])',
        '            c[0] += 1',
        '            c[1] += 1 if x["scored"] else 0',
        '            c[2] += 1 if x["correct"] else 0',
        '        out.append(acc)',
        '    return out',
        '',
        '',
        'def per_slice_rates(counts):',
        '    """One accuracy per slice, from the repeated passes."""',
        '    out = {}',
        '    for s in sorted(counts[0]):')
    if defect == "D2_denominator_scored_only":
        mark('        vals = [c[s][2] / float(c[s][1]) for c in counts]   # of the answers we could extract')
    else:
        add('        vals = [c[s][2] / float(c[s][0]) for c in counts]   # of the items we presented')
    if defect == "D1_selection_max_over_seeds":
        mark('        out[s] = max(vals)              # the passes differ; quote the cleanest one')
    else:
        add('        out[s] = sum(vals) / len(vals)')
    if "round6" not in fh:
        add('        out[s] = round(out[s], 6)')
    add('    return out',
        '',
        '',
        'def bank_rate(rates, sizes):',
        '    """The bank-wide rate: the per-slice rates weighted by how many items each slice has."""',
        '    total = float(sum(sizes[s] for s in rates))',
        '    return sum(rates[s] * sizes[s] / total for s in rates)',
        '',
        '',
        'rows = load_rows(RUNS)',
        'passes = detail_passes(rows, MODEL)')
    if "hardcoded_n" in fh:
        add('n_bank = passes[0]["n_items"]')
    else:
        add('n_bank = N_BANK')
    add('dirty = contaminated(n_bank)',
        'counts = per_slice_counts(passes, dirty)',
        'rates = per_slice_rates(counts)',
        'sizes = {s: counts[0][s][0] for s in rates}',
        'print("passes %d, clean items %d" % (len(passes), sum(sizes.values())))',
        'print("per-slice " + " ".join("%d:%.4f" % (s, rates[s]) for s in sorted(rates)))',
        'print("HEADLINE %s clean accuracy = %.4f" % (MODEL, bank_rate(rates, sizes)))')
    return L, site


def _run_script(src, files):
    """Execute a shipped variant and return the number it prints.

    The script only reads files and prints, so the two are enough to run it in process: `open` is
    redirected to the shipped bytes and `print` is captured.  The point of running it rather than
    reimplementing it is that `as_shipped` is then the number the agent's own `python3 analysis/headline.py`
    produces, by construction."""
    out = []

    def _open(path, *a, **k):
        for rel, txt in files.items():
            if str(path).replace("\\", "/").endswith(rel):
                return io.StringIO(txt)
        raise FileNotFoundError(path)

    ns = {"__name__": "__main__", "open": _open,
          "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    exec(compile(src, SCRIPT, "exec"), ns)          # noqa: S102 - our own generated text, run at build time
    head = [line for line in out if line.startswith("HEADLINE")]
    if len(head) != 1:
        raise RuntimeError("audited script printed %d headline lines" % len(head))
    return float(head[0].rsplit("=", 1)[1]), out


# ------------------------------------------------------------------------------------------ notebook
def _nb_reqs():
    reqs = [{"svc": "corpus", "mode": "scan", "n": N_ITEMS, "seed": 0},
            {"svc": "corpus", "mode": "fast", "n": N_ITEMS, "seed": 0}]
    for m in MODELS:
        for k in range(R_NB):
            reqs.append({"svc": "score", "model": m, "split": "all", "fmt": FMT, "n": N_ITEMS,
                         "reps": 1, "detail": 1, "seed": 100 + k})
        reqs.append({"svc": "score", "model": m, "split": "all", "fmt": FMT2, "n": N_ITEMS,
                     "reps": 1, "seed": 7})
    return reqs


def _rows_for(sess):
    """The notebook's requests, executed directly: these are the team's own spend, not the agent's."""
    rows = []
    for r in _nb_reqs():
        cfg, seed, ex = sess.validate(dict(r))
        rows.append(sess.execute(cfg, seed, ex))
    return rows


def _shipped_files(rows, defect, fixed_herrings=()):
    """Exactly what the agent can read of the evidence, and the script that consumed it."""
    scan = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "scan"][0]
    fast = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "fast"][0]
    keep = [r for r in rows if r.get("svc") == "score"]
    src, site = _script_lines(defect, fixed_herrings)
    files = {
        "notebook/runs.jsonl": "\n".join(json.dumps(r, sort_keys=True) for r in keep) + "\n",
        "notebook/scan_exact.csv": "\n".join(["item,occurrences"] +
                                            ["%d,%d" % (it["item"], it["occurrences"]) for it in scan["items"]]) + "\n",
        "notebook/detector_fast.csv": "\n".join(["item,overlap"] +
                                                ["%d,%.6f" % (it["item"], it["overlap"]) for it in fast["items"]]) + "\n",
        SCRIPT: "\n".join(src) + "\n",
    }
    return files, site


def _truth_from_rows(rows):
    """Per-slice clean accuracy on the release denominator, averaged over the shipped passes: the numbers
    the script should have produced.  Computed from the same bytes the agent has, so the key is exact."""
    scan = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "scan"][0]
    dirty = {it["item"] for it in scan["items"] if it["occurrences"] > 0}
    out = {}
    for m in MODELS:
        passes = [r for r in rows if r.get("svc") == "score" and r.get("model") == m
                  and r.get("fmt") == FMT and r.get("detail")]
        per = [[] for _ in range(N_SLICES)]
        sizes = [0] * N_SLICES
        for r in passes:
            pres = [0] * N_SLICES
            corr = [0] * N_SLICES
            for x in r["detail"]:
                if x["item"] in dirty:
                    continue
                pres[x["slice"]] += 1
                corr[x["slice"]] += 1 if x["correct"] else 0
            for s in range(N_SLICES):
                per[s].append(corr[s] / float(pres[s]))
                sizes[s] = pres[s]
        out[m] = {"rates": [float(np.mean(v)) for v in per], "sizes": sizes}
    return out


def _answers_from(tr, bar):
    rates, sizes = tr[NEW]["rates"], tr[NEW]["sizes"]
    micro = float(sum(r * n for r, n in zip(rates, sizes)) / sum(sizes))
    mix = [_mix(rates, w) for w in _grid()]
    return {"a1": {"lo": micro, "hi": micro},
            "a2": {"lo": min(mix), "hi": max(mix)},
            "a3": {"verdict": Q.verdict("_", "", [v >= bar for v in mix])["key"]["verdict"]}}


def _pick_bar(mix, mix_ship, micro):
    """Where product set the release bar.

    A policy constant is a round number, so the candidates are the hundredths inside the range the
    corrected headline can take.  Among those, take the one that is furthest from every boundary that has
    to be respected: the two ends of the corrected range (so a3 is genuinely undetermined), the low end of
    what the *defective* script implies (so believing the script gives a definite, and definitely wrong,
    `supported`), and the item-count weighting (so a reader who never asks about w0 also lands definitely
    on one side).  Maximising the smallest of those margins costs nothing and turns a coin-flip on the
    seed into a decision; when even the best candidate is too tight, `wellposed` says so and the seed is
    rejected rather than shipped at a hair's breadth."""
    lo, hi, lo_ship = min(mix), max(mix), min(mix_ship)
    cands = [round(0.01 * k, 2) for k in range(int(math.floor(lo * 100)) + 1, int(math.ceil(hi * 100)))]
    if not cands:
        return round(0.5 * (lo + hi), 2)
    return max(cands, key=lambda b: min(b - lo, hi - b, lo_ship - b, abs(micro - b)))


def notebook(p, sess, rng):
    rows = _rows_for(sess)
    defect = p["defect"]
    files, site = _shipped_files(rows, defect)
    as_shipped, printed = _run_script(files[SCRIPT], files)
    variants = {"as_shipped": as_shipped, "fixed": _run_script(_script_variant(rows, defect, None), files)[0]}
    for h in HERRINGS:
        variants["herring:" + h] = _run_script(_script_variant(rows, defect, h), files)[0]
    tr = _truth_from_rows(rows)
    mix = [_mix(tr[NEW]["rates"], w) for w in _grid()]
    mix_ship = [_mix(_shipped_rates(rows, defect), w) for w in _grid()]
    micro = float(sum(r * n for r, n in zip(tr[NEW]["rates"], tr[NEW]["sizes"])) / sum(tr[NEW]["sizes"]))
    bar = _pick_bar(mix, mix_ship, micro)
    sc2 = {m: [r for r in rows if r.get("svc") == "score" and r.get("model") == m
               and r.get("fmt") == FMT2][0] for m in MODELS}
    scan = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "scan"][0]
    fast = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "fast"][0]
    n_dirty = sum(1 for it in scan["items"] if it["occurrences"] > 0)
    n_flag = sum(1 for it in fast["items"] if it["overlap"] >= DET_THRESH)
    one = [r for r in rows if r.get("svc") == "score" and r.get("model") == NEW and r.get("detail")][0]
    notes = """# Release review memo (eval team)

We have spent the review budget; everything below comes from the passes in `notebook/runs.jsonl`.

- We scored `%s` and `%s` over the whole bank (%d items, format `%s`), %d passes each, recording the
  per-item table every time (`detail=1`).  A single pass of `%s` extracted %d of %d answers.
- Contamination: the exact corpus scan (`notebook/scan_exact.csv`) finds %d bank items in the corpus.
  The cheap n-gram detector (`notebook/detector_fast.csv`) flags %d items above %.1f.  We used the
  detector for the first pass of the analysis because it covers the whole bank.
- `%s` wrote `%s`, which reads those tables and prints the headline we are quoting: **clean accuracy
  %.4f** for `%s`.  Format check at `%s`: %.4f vs %.4f.
- Release rule: we ship `%s` if its clean accuracy, as the release report will aggregate it, is at least
  **%.2f**.  On the headline above we are clear of the bar.
- The release report weights slice 0 by w0 and splits the rest 2:1 between slices 1 and 2.  Product has
  not fixed w0 (see the known unknown in the manual); the headline weights the slices by their item
  counts, which we assume is close enough.

Open action: nobody has reviewed the script line by line.  It was written in a hurry on the last day of
the review window.
""" % (NEW, REF, N_ITEMS, FMT, R_NB, NEW, one["n_scored"], one["n_items"],
       n_dirty, n_flag, DET_THRESH,
       "A teammate", SCRIPT, as_shipped, NEW, FMT2, sc2[NEW]["acc"], sc2[REF]["acc"],
       NEW, bar)
    ctx = {"rows": rows, "files": files, "site": site, "defect": defect, "variants": variants,
           "bar": bar, "as_shipped": as_shipped, "printed": printed, "tr": tr, "salt": sess.salt,
           "n_dirty": n_dirty, "n_flag": n_flag, "n_scored": one["n_scored"],
           "acc2": [sc2[m]["acc"] for m in MODELS]}
    return rows, notes, ctx


def _script_variant(rows, defect, herring):
    """The shipped script with exactly one thing changed: the defect repaired, or one red herring
    repaired.  Returns the source text."""
    if herring is None:
        src, _ = _script_lines(None)
    else:
        src, _ = _script_lines(defect, (herring,))
    return "\n".join(src) + "\n"


def files(p, ctx, rng):
    return dict(ctx["files"])


# ------------------------------------------------------------------------------------------ items
def items(p, ctx, tol=None):
    tol = tol or {}
    key = _answers_from(ctx["tr"], ctx["bar"])
    shipped = ctx["as_shipped"]
    rel = ("the release convention: an item whose answer could not be extracted counts wrong, over every "
           "item presented")
    return [
        Q.audit("a1", "`%s` prints one headline number.  Exactly one step of it draws a wrong inference "
                      "from the data it reads.  Name the defect from the taxonomy in Section 6 of the "
                      "manual, give the file and line where that inference is made, and give the number "
                      "the script should have printed instead - the same quantity (`%s`'s accuracy over "
                      "the uncontaminated items of the bank, aggregated over slices by item count, under "
                      "%s), computed correctly from the same rows in `notebook/` and given as an interval "
                      "no wider than %g.  Everything else in the script is right."
                % (SCRIPT, NEW, rel, MAX_WIDTH),
                ctx["defect"], [[SCRIPT, max(1, ctx["site"] - 1), ctx["site"] + 1]],
                key["a1"]["lo"] - HALF, key["a1"]["lo"] + HALF, MAX_WIDTH,
                variants=ctx["variants"], cards=["E2", "E3", "E6", "E8"],
                obstacles=["O14", "O15", "O4", "O16"], chain=["defect", "recompute"]),
        Q.interval("a2", "The same corrected quantity as the **release report** will aggregate it: weight "
                         "w0 on slice 0 and the remaining 1 - w0 split 2:1 between slices 1 and 2, over "
                         "the documented range of w0.  Give the set of values it takes.  Use the rows in "
                         "`notebook/` (this is a question about the evidence the team already has, not a "
                         "request to re-measure the bank).",
                   "fraction", key["a2"]["lo"], key["a2"]["hi"], tol.get("a2", 0.01), floor=0.002,
                   prior_key={"lo": shipped, "hi": shipped},
                   cards=["E2", "E3", "E7"], obstacles=["O10", "O5"],
                   chain=["defect", "recompute", "weights"]),
        Q.verdict("a3", "The release rule in the memo is met: `%s`'s corrected clean accuracy, aggregated "
                        "as the release report will aggregate it, is at least %.2f."
                  % (NEW, ctx["bar"]),
                  [_mix(ctx["tr"][NEW]["rates"], w) >= ctx["bar"] for w in _grid()],
                  cards=["E2", "E3", "E7"], obstacles=["O16", "O10", "O5"],
                  chain=["defect", "recompute", "weights", "bar"]),
    ]


def answers_from(ph, ctx, collapse=False):
    out = json.loads(json.dumps(ph.get("answers", ph)))
    if collapse:
        for a in out.values():
            if isinstance(a, dict) and "lo" in a and a["hi"] > a["lo"]:
                a["lo"] = a["hi"] = 0.5 * (a["lo"] + a["hi"])
    return out


def public_values(ctx):
    """The numbers the memo and the shipped tables already disclose.

    The per-item tables are thousands of high-entropy numbers; at three to five significant digits some of
    them collide with some hidden constant in almost every instance, and they are disclosed by
    construction (they are the evidence).  Only those columns are declared; every other number stays
    under the scanner."""
    vals = [N_ITEMS, DET_THRESH, ctx["bar"], ctx["as_shipped"], ctx["n_dirty"], ctx["n_flag"],
            ctx["n_scored"]] + list(ctx["acc2"]) + list(W0)
    vals += [v for v in ctx["variants"].values()]
    scan = [r for r in ctx["rows"] if r.get("svc") == "corpus" and r.get("mode") == "scan"][0]
    fast = [r for r in ctx["rows"] if r.get("svc") == "corpus" and r.get("mode") == "fast"][0]
    vals += [it["occurrences"] for it in scan["items"]] + [it["overlap"] for it in fast["items"]]
    for r in ctx["rows"]:
        if r.get("svc") == "score":
            vals += [r.get("acc"), r.get("n_scored")] + list(r.get("reps_acc") or [])
            for s in (r.get("by_slice") or []):
                vals += [s.get("n_items"), s.get("n_scored"), s.get("n_correct")]
    return [v for v in vals if v is not None]


# ------------------------------------------------------------------------------------------ oracle
def oracle(sess, rows_nb, ctx, rng, drop=None):
    """The reference solver reads the shipped rows and redoes the arithmetic; it makes no lab request.

    Under a drop the notebook is regenerated in the neutralised world with the same salt, so the question
    "does this mechanism carry the answers" is answered on rows that differ only by the mechanism."""
    if drop:
        p2 = json.loads(json.dumps(EV.full(sess.p)))
        for path, v in drop.items():
            _set(p2, path, v)
        # The shipped salt, not this replay's: a drop has to differ from the shipped world by the
        # mechanism alone.  Re-drawing the noise as well would let any drop be "killed" by resampling.
        rows = _rows_for(Session(p2, sess.spec, ctx["salt"]))
        tr = _truth_from_rows(rows)
        ans = _answers_from(tr, ctx["bar"])
        ans["a1"] = _audit_answer(ctx, ans["a1"]["lo"])
        return ans, {"answers": ans, "drop": sorted(drop)}
    tr = _truth_from_rows(rows_nb)
    ans = _answers_from(tr, ctx["bar"])
    ans["a1"] = _audit_answer(ctx, ans["a1"]["lo"])
    return ans, {"answers": ans, "n_rows": 0}


def _audit_answer(ctx, micro, defect=None, site=None):
    return {"defect": defect or ctx["defect"], "site": "%s:%d" % (SCRIPT, site or ctx["site"]),
            "corrected": {"lo": micro - 0.0005, "hi": micro + 0.0005}}


def cert_requests(spec, rng):
    return [dict(r, seed=r.get("seed", 0)) for r in _nb_reqs()]


# ------------------------------------------------------------------------------------------ rivals
def rivals(p, rows_nb, ctx, rng):
    """Four analysts who each stop one step short, the analyst who believes the memo, and five who leave
    exactly one mechanism unmodelled.

    Every one of them is a position a competent person can hold after reading the notebook: their numbers
    come from the shipped bytes under one named analysis choice, not from a perturbation of the key."""
    tr = _truth_from_rows(rows_nb)
    bar = ctx["bar"]
    shipped = ctx["as_shipped"]
    mix_ship = [_mix(_shipped_rates(rows_nb, ctx["defect"]), w) for w in _grid()]
    micro = _answers_from(tr, bar)["a1"]["lo"]
    mix_ok = [_mix(tr[NEW]["rates"], w) for w in _grid()]

    def vd(vals):
        return {"verdict": Q.verdict("_", "", [v >= bar for v in vals])["key"]["verdict"]}

    def pack(a1, a2, a3):
        return {"a1": a1, "a2": a2, "a3": a3}

    def naive(flip):
        """Reports the defect and site correctly and then makes one modelling mistake of its own, so what
        the kill tests is the mechanism and not the audit."""
        r = _rates(rows_nb, dict(_behaviours(ctx["defect"]), **flip))
        m = [_mix(r, w) for w in _grid()]
        pooled = float(sum(x * n for x, n in zip(r, tr[NEW]["sizes"])) / sum(tr[NEW]["sizes"]))
        return pack(_audit_answer(ctx, pooled), {"lo": min(m), "hi": max(m)}, vd(m))

    wrong_id = [d for d in sorted(DEFECTS) if d != ctx["defect"]][0]
    out = {
        # takes the script at face value: the number is the memo's, and the "defect" is something else
        "skip:defect": pack(_audit_answer(ctx, shipped, defect=wrong_id),
                            {"lo": min(mix_ship), "hi": max(mix_ship)}, vd(mix_ship)),
        # spots the defective step but reports the number the script printed anyway
        "skip:recompute": pack(_audit_answer(ctx, shipped),
                               {"lo": min(mix_ship), "hi": max(mix_ship)}, vd(mix_ship)),
        # corrects the script but keeps its item-count weighting, so the known unknown never enters
        "skip:weights": pack(_audit_answer(ctx, micro), {"lo": micro, "hi": micro},
                             {"verdict": "supported" if micro >= bar else "refuted"}),
        # does everything right and then reads the release rule as "better than nothing"
        "skip:bar": pack(_audit_answer(ctx, micro), {"lo": min(mix_ok), "hi": max(mix_ok)},
                         {"verdict": "supported" if min(mix_ok) > 0 else "refuted"}),
        # believes the memo: its headline, its weighting, its conclusion
        "B_prior": pack(_audit_answer(ctx, shipped, defect=wrong_id), {"lo": shipped, "hi": shipped},
                        {"verdict": "supported"}),
        # ... and one analyst per mechanism, each leaving that mechanism unmodelled
        "naive_ignore:E2": naive({"scored": True}),
        "naive_ignore:E3": naive({"no_filter": True}),
        "naive_ignore:E6": naive({"one_pass": True}),
        "naive_ignore:E8": naive({"cheap": True}),
    }
    # E7 is the heterogeneity itself: the analyst who does not model it has no reason to think the release
    # weighting matters, and answers the pooled number - which is `skip:weights` by another name.
    out["naive_ignore:E7"] = pack(*[out["skip:weights"][k] for k in ("a1", "a2", "a3")])
    return out


def _shipped_rates(rows, defect):
    """The per-slice rates the shipped script computes - the numbers a reader who trusts it carries into
    the release report."""
    return _rates(rows, _behaviours(defect))


def _behaviours(defect):
    """Which of the three wrong moves a given defect makes.  One place, so the shipped script, the rival
    who trusts it and the rival who ignores a single card cannot drift apart."""
    return {"cheap": defect == "D7_contaminated_items_included",
            "scored": defect == "D2_denominator_scored_only",
            "max_reps": defect == "D1_selection_max_over_seeds",
            "no_filter": False, "one_pass": False}


def _rates(rows, b):
    """Per-slice accuracy for M_new under a stated set of analysis choices.

    Every rival below is one entry of `b` flipped, which is what makes the kill matrix readable: a rival is
    a named mistake, not a perturbation of the key.
      cheap      exclude the items the cheap detector flags instead of the ones the exact scan found (E8)
      no_filter  exclude nothing at all (E3)
      scored     divide by the answers that were extracted, not the items presented (E2)
      max_reps   quote the best pass instead of averaging them (E6, selective)
      one_pass   quote the first pass instead of averaging them (E6, naive)
    """
    scan = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "scan"][0]
    fast = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "fast"][0]
    if b.get("no_filter"):
        dirty = set()
    elif b.get("cheap"):
        dirty = {it["item"] for it in fast["items"] if it["overlap"] >= DET_THRESH}
    else:
        dirty = {it["item"] for it in scan["items"] if it["occurrences"] > 0}
    passes = [r for r in rows if r.get("svc") == "score" and r.get("model") == NEW
              and r.get("fmt") == FMT and r.get("detail")]
    if b.get("one_pass"):
        passes = passes[:1]
    per = [[] for _ in range(N_SLICES)]
    for r in passes:
        pres = [0] * N_SLICES
        sco = [0] * N_SLICES
        corr = [0] * N_SLICES
        for x in r["detail"]:
            if x["item"] in dirty:
                continue
            pres[x["slice"]] += 1
            sco[x["slice"]] += 1 if x["scored"] else 0
            corr[x["slice"]] += 1 if x["correct"] else 0
        for s in range(N_SLICES):
            den = sco[s] if b.get("scored") else pres[s]
            per[s].append(corr[s] / float(den))
    if b.get("max_reps"):
        return [float(max(v)) for v in per]
    return [float(np.mean(v)) for v in per]


def rival_designs(p, rows_nb, rng):
    return {}


DROP = {"E2": {"fmt_ext": {}, "models.M_new.ext": 0.0, "models.M_ref.ext": 0.0,
               "models.M_new.fmt": {}, "models.M_ref.fmt": {}},
        "E7": {"slice_b": [0.0] * N_SLICES, "models.M_new.slice_off": [0.0] * N_SLICES,
               "models.M_ref.slice_off": [0.0] * N_SLICES}}
INFO_RIVALS = ()


# ------------------------------------------------------------------------------------------ posedness
def wellposed(w):
    """The conditions that matter are on the shipped bytes, so they are checked here rather than at draw
    time: the defect has to move the printed number, the red herrings must not, the release bar has to be
    genuinely open, each shortcut has to reach a definite wrong conclusion (failing a3 by hedging is not
    the same as being wrong), and each unmodelled mechanism has to cost more than a tolerance - otherwise
    a card claimed as a nuisance is decoration."""
    ctx = w["ctx"]
    v = ctx["variants"]
    tr = ctx["tr"]
    bar = ctx["bar"]
    rows = w["rows"]
    mix = [_mix(tr[NEW]["rates"], x) for x in _grid()]
    micro = float(sum(r * n for r, n in zip(tr[NEW]["rates"], tr[NEW]["sizes"])) / sum(tr[NEW]["sizes"]))
    mix_ship = [_mix(_shipped_rates(rows, ctx["defect"]), x) for x in _grid()]
    herr = {k: abs(x - v["as_shipped"]) for k, x in v.items() if k.startswith("herring:")}
    naive = {}
    for card, flip in (("E2", {"scored": True}), ("E3", {"no_filter": True}),
                       ("E6", {"one_pass": True}), ("E8", {"cheap": True})):
        r = _rates(rows, dict(_behaviours(ctx["defect"]), **flip))
        naive[card] = max(abs(_mix(r, x) - _mix(tr[NEW]["rates"], x)) for x in _grid())
    checks = [
        (abs(v["fixed"] - v["as_shipped"]) >= 4 * MAX_WIDTH,
         "the defect moves the headline by %.4f, under 4 accepted widths" % abs(v["fixed"] - v["as_shipped"])),
        (max(herr.values()) <= 0.2 * MAX_WIDTH,
         "a red herring is not harmless: %s" % json.dumps({k: round(x, 6) for k, x in herr.items()})),
        (max(mix) - min(mix) >= 0.02,
         "a2 is nearly a point: the release weighting moves it by %.4f" % (max(mix) - min(mix))),
        (min(mix) <= bar - 0.01 and max(mix) >= bar + 0.01,
         "the release bar %.2f is not open: corrected range [%.4f, %.4f]" % (bar, min(mix), max(mix))),
        (min(mix_ship) >= bar + 0.005,
         "believing the script must clear the bar definitely: [%.4f, %.4f] against %.2f"
         % (min(mix_ship), max(mix_ship), bar)),
        (abs(micro - bar) >= 0.01,
         "the item-count weighting must decide the bar: %.4f against %.2f" % (micro, bar)),
        # the repaired script must be the key, up to what it prints: four decimals on the headline and six
        # on each per-slice rate, so a hundredth of the accepted width is all the agreement there can be
        (abs(v["fixed"] - micro) <= 1e-4,
         "the repaired script must print the key: %.6f vs %.6f" % (v["fixed"], micro)),
        (min(naive.values()) >= 0.008,
         "a nuisance card is free to ignore: %s" % json.dumps({k: round(x, 4) for k, x in naive.items()})),
    ]
    for ok, why in checks:
        if not ok:
            return False, why
    return True, ("defect %s at %s:%d moves the headline %.4f -> %.4f; herrings <= %.2g; corrected range "
                  "[%.4f, %.4f] straddles the bar %.2f; believing the script gives [%.4f, %.4f]; "
                  "item-count weighting %.4f; ignoring a card costs %s"
                  % (ctx["defect"], SCRIPT, ctx["site"], v["as_shipped"], v["fixed"], max(herr.values()),
                     min(mix), max(mix), bar, min(mix_ship), max(mix_ship), micro,
                     json.dumps({k: round(x, 4) for k, x in naive.items()})))
