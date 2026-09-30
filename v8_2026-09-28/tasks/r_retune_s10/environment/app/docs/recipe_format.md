# /app/recipe.json

```json
{"lr": 0.00042, "wd": 0.071,
 "loss": {"lo": 2.4861, "hi": 2.4951}}
```

* `lr`, `wd` (positive numbers) - the recipe for the next run.
* `loss.lo`, `loss.hi` (numbers) - interval for the expected final validation loss of the next run with the
  recipe in this file.  Graded against your recipe, not against the best one, so a suboptimal recipe costs you
  one item and not two.  `hi - lo` must be <= 0.011.

Nothing else in the file is read.  There is no partial credit inside an item.
