# Fix the MiniLab v2 regression

The repository in `/app` is MiniLab v2, a NumPy re-implementation of a small
training pipeline. Read `/app/README.md` first: on the frozen canonical
configuration, v2 reaches a worse validation MSE than the v1 reference
implementation.

Your job is to make the v2 **training code** (`/app/minilab/`) faithfully
implement the reference semantics described in the README, so that it
reproduces v1's results.

Rules:

* Do **not** edit `configs/`, `data/`, or the hyperparameter values, and do not
  change the model architecture, initialisation, number of steps, or the data
  sampling. Keep the public API `minilab.trainer.train(cfg, x_train, y_train, seed)`
  returning `{"W1","b1","W2","b2"}`.
* Hyperparameter compensation (e.g. silently rescaling the learning rate) is not
  a fix.

How you will be graded: the grader imports your `minilab.trainer.train` and
trains it on the canonical configuration **and on several held-out
configurations** (different `grad_accum_steps`, `momentum`, `lr`,
`weight_decay`, seeds and freshly sampled data from the same distribution). For
every configuration, the mean held-out MSE of your code must match the v1
reference implementation within statistical tolerance, and no run may diverge.

When you are done, also write `/app/FIX_NOTES.md` listing each change you made
and why.

You have 3600 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
