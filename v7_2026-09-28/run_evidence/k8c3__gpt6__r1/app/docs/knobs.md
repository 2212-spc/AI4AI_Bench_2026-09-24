# The four knobs

| knob | recipe | meaning |
|---|---|---|
| `capacity` | 1.0 | student capacity headroom on the watched slice. Below 1.0 the student cannot absorb what the teacher knows about that slice. |
| `leak` | 0.0 | fraction of the eval slice that overlaps the fine-tuning set. Raises the **offline** number. Has no effect on what users get. |
| `mixture` | 1.0 | the watched slice's share of the fine-tuning mixture, relative to the recipe. |
| `lr_scale` | 1.0 | fine-tuning learning rate relative to the recipe default. |

Notes from the training team, collected during the incident review:

* "offline eval and online serving use the same prompts but different sampling; they normally agree to
  within a few thousandths"
* "we have never re-tuned the learning rate after changing model size - the recipe value has always worked"
* "the data team added a new source this quarter; the overlap audit is still open"
* "the watched slice is small, so its share of the mixture moves easily when anything else is added"
