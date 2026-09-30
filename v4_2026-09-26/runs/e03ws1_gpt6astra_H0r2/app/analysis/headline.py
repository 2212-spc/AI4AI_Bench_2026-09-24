#!/usr/bin/env python3
"""Headline number for the release review.

Reads what we collected in notebook/ and prints the clean-subset accuracy of M_new that the
review memo quotes.  Nothing here calls the lab; the passes are already on disk.

    python3 analysis/headline.py
"""
import csv
import json

RUNS = "notebook/runs.jsonl"
DETECTOR = "notebook/detector_fast.csv"
SCAN = "notebook/scan_exact.csv"
MODEL = "M_new"
DET_THRESH = 0.5       # the cheap detector score above which the memo calls an item suspect
N_BANK = 1500          # size of the item bank


def load_rows(path):
    """Every lab row we have, one JSON object per line."""
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def detail_passes(rows, model):
    """The scoring passes that recorded a per-item table."""
    out = [r for r in rows if r.get("svc") == "score" and r.get("model") == model
           and r.get("fmt") == "mc_letter" and r.get("detail")]
    out.sort(key=lambda r: r["config"]["seed"])
    return out


def contaminated(n_bank):
    """Which bank items are in the pretraining corpus.  We have both the exact scan and the cheap
    detector's scores, so print both counts before deciding which set to exclude."""
    exact = set()
    for row in csv.DictReader(open(SCAN)):
        if int(row["occurrences"]) > 0:
            exact.add(int(row["item"]))
    cheap = set()
    for row in csv.DictReader(open(DETECTOR)):
        if float(row["overlap"]) >= DET_THRESH:
            cheap.add(int(row["item"]))
    print("contamination: exact scan %d of %d items, cheap detector flags %d"
          % (len(exact), n_bank, len(cheap)))
    return cheap        # the scan is a sample of the corpus; the detector covers the whole bank


def per_slice_counts(passes, dirty):
    """For each pass, per slice: items presented, answers extracted, answers correct, over the
    items that are not contaminated."""
    out = []
    for r in passes:
        acc = {}
        for x in r["detail"]:
            if x["item"] in dirty:
                continue
            c = acc.setdefault(x["slice"], [0, 0, 0])
            c[0] += 1
            c[1] += 1 if x["scored"] else 0
            c[2] += 1 if x["correct"] else 0
        out.append(acc)
    return out


def per_slice_rates(counts):
    """One accuracy per slice, from the repeated passes."""
    out = {}
    for s in sorted(counts[0]):
        vals = [c[s][2] / float(c[s][0]) for c in counts]   # of the items we presented
        out[s] = sum(vals) / len(vals)
        out[s] = round(out[s], 6)
    return out


def bank_rate(rates, sizes):
    """The bank-wide rate: the per-slice rates weighted by how many items each slice has."""
    total = float(sum(sizes[s] for s in rates))
    return sum(rates[s] * sizes[s] / total for s in rates)


rows = load_rows(RUNS)
passes = detail_passes(rows, MODEL)
n_bank = N_BANK
dirty = contaminated(n_bank)
counts = per_slice_counts(passes, dirty)
rates = per_slice_rates(counts)
sizes = {s: counts[0][s][0] for s in rates}
print("passes %d, clean items %d" % (len(passes), sum(sizes.values())))
print("per-slice " + " ".join("%d:%.4f" % (s, rates[s]) for s in sorted(rates)))
print("HEADLINE %s clean accuracy = %.4f" % (MODEL, bank_rate(rates, sizes)))
