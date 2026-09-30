# /app/plan.json

```json
{"opts": ["C-114", "C-203", "C-241"],
 "speedup": {"lo": 1.61, "hi": 1.68},
 "weakest": "C-241"}
```

* `opts` (list of strings, non-empty, distinct) - the changes you ship.  Total review cost must be <= the
  cap stated in the task (**11 review-days** for this instance).
* `speedup.lo`, `speedup.hi` (numbers) - production speedup of the set in `opts`, i.e. stock production
  latency divided by your set's production latency.  Graded against YOUR set, not against the optimal one,
  so a suboptimal set costs you one item and not two.  `(hi-lo)/truth` must be <= 0.06.
* `weakest` (string) - must be one of the ids in your own `opts`.

Nothing else in the file is read, and there is no partial credit inside an item.
