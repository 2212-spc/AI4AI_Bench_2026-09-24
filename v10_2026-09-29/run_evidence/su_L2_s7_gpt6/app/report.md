# Nightly classifier update

I kept the command line and output format unchanged, but changed the training path in `repo/train.py` in three ways:

- Standardized features and applied full covariance whitening estimated from the input corpus.
- Replaced ordinary cross entropy with a likelihood that models random crowd-label replacement. This prevents the network from fitting the observed crowd class frequencies as if they were gold frequencies.
- Used the 32-pass production budget with a 256-unit MLP and averaged late checkpoints to reduce stochastic variation. Weight decay scales with corpus size through `decay_power`, so the larger production corpus is regularized less than the small development sample.

The noise objective has a finite-difference gradient check, thread limits are set before importing NumPy, and the production input/output path was exercised locally. On the supplied sample/dev pair, the whitening plus noise-aware objective improved the two-seed mean dev accuracy from about 0.666 for the shipped settings to about 0.694 in the final small-data check. The production run uses the same settings while learning from all supplied corpus rows.
