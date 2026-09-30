# /app/report.json

```json
{"culprit": "H-2",
 "effect": {"lo": 1.31, "hi": 1.58},
 "harness_only": ["H-4"]}
```

* `culprit` (string) - one change id.
* `effect.lo`, `effect.hi` (numbers) - score points of **quality** gained by reverting the change you
  named in `culprit`.  Graded against that change, not against the correct one, so a wrong culprit costs
  you one item and not two.  `hi - lo` must be <= 0.45.
* `harness_only` (list of strings, possibly empty) - exact set match; order is ignored.

Nothing else in the file is read.  There is no partial credit inside an item.
