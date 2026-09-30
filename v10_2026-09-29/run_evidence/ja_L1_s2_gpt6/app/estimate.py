"""Reproduce the two-stage, verification-based finite-pool estimates."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
pilot = json.loads((ROOT / 'data/audit_design.json').read_text())
second = json.loads((ROOT / 'data/second_stage_design.json').read_text())
labels = {model: {} for model in 'AB'}
for path in sorted((ROOT / 'data').glob('verify_*.csv')):
    for row in csv.DictReader(path.open()):
        model, item = row['model'], int(row['item'])
        assert item not in labels[model], 'Duplicate verification'
        labels[model][item] = int(row['verified_correct'])

rates = {}
summary = {}
for model in 'AB':
    initial = [i for group in pilot[model].values() for i in group['pilot']]
    remainder = [i for group in second[model].values() for i in group['population']]
    assert len(initial) + len(remainder) == 600
    assert len(set(initial + remainder)) == 600
    estimated_total = sum(labels[model][i] for i in initial)
    variance = 0.0
    rows = []
    for name, group in second[model].items():
        N, n = len(group['population']), len(group['sample'])
        y = np.array([labels[model][i] for i in group['sample']], dtype=float)
        estimated_total += N * y.mean()
        variance += N**2 * (1 - n / N) * y.var(ddof=1) / n
        rows.append(dict(group=name, remaining_population=N, sample=n, passes=int(y.sum())))
    rates[model] = estimated_total / 600
    summary[model] = dict(rate=rates[model], verified=len(labels[model]),
                          nominal_standard_error=float(np.sqrt(variance) / 600),
                          second_stage=rows)
assert sum(len(x) for x in labels.values()) == 190
summary['cost'] = {'review': 120.0, 'verify': 380.0, 'total': 500.0}
(ROOT / 'data/estimates.json').write_text(json.dumps(summary, indent=2) + '\n')
difference = rates['B'] - rates['A']
(ROOT / 'answers.md').write_text(
    f"Q1: {rates['B']:.3f}\nQ2: {difference:+.3f}\nQ3: {'SHIP' if difference > 0 else 'HOLD'}\n"
)
print(json.dumps(summary, indent=2))
