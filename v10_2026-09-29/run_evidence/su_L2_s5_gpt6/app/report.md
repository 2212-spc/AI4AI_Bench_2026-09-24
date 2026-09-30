# Nightly classifier update

The shipped trainer keeps the numpy MLP and the production interface, but adds a generalized cross entropy noise filter. During each update, the gradient from a row is multiplied by the model's predicted probability for its crowd label raised to `robust_exponent`. Rows whose crowd label is inconsistent with the learned structure therefore stop dominating training, while high-confidence rows retain almost their full gradient. This is enabled with exponent `0.5`; the regularization was retuned to weight decay `2.0` for the 20,000-step full-corpus run.

On the available 4,000-row crowd sample and 1,000-row gold development set, the original configuration reached 0.696 accuracy with seed 0. The noise-filtered configuration reached 0.713 under the same 1,000-step comparison. The production command remains unchanged, and the trainer still writes an integer `int64` class for every prediction row.
