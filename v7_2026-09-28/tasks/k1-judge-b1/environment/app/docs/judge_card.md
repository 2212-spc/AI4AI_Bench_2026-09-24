# Judge model card (excerpt) - `judge-3-pro`

Internal eval, %s arena. Reported by the eval team, not re-verified.

| slice | agreement with human |
|---|---|
| overall | 0.86 |
| short answers | 0.89 |
| long answers | 0.83 |

**Known issues.** The judge has a documented preference for longer, more elaborate answers; on a
side-by-side where the candidate simply wrote more, it calls a win more often than humans do. The size of
this effect has not been measured per topic.

**What this card does not tell you.** "Agreement" is a single number that mixes both directions of error.
Two judges with the same agreement can have completely different false-positive and false-negative rates,
and only the split matters for correcting a win rate. The overall figure is also computed on *this* arena's
mix; it does not transfer to a different mix.
