# Production traffic (ops dashboard, last 30 days)

The arena was assembled for coverage, not to look like traffic - it deliberately over-samples the hard
strata so that regressions show up. Production is weighted differently, and the exact weights are handed to
your estimator as `env.production_mix`.

This matters more than it sounds: a win rate is an average over a mix, so two correct estimators that
average under different mixes return different numbers, and only one of them answers the question that was
asked.
