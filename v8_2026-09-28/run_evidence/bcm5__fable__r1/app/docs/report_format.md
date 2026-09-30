# /app/report.json

```json
{"culprit": "H-2",
 "effect": {"lo": 1.31, "hi": 1.58}}
```

* `culprit` (string) - one change id.
* `effect.lo`, `effect.hi` (numbers) - external-suite points gained by rolling back the change you named in
  `culprit`.  Graded against that change, not against the correct one, so a wrong culprit costs you one item
  and not two.  `hi - lo` must be <= 0.45.

Nothing else in the file is read.  There is no partial credit inside an item.
