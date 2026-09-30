# How a release is built and measured

The training pipeline and the evaluation harness are configured from the same manifest.  A change entry in
the manifest can carry a training-side patch, a harness-side patch, or both; the manifest does not say
which, and the vendor does not publish per-change numbers.

Two properties of the setup are worth writing down before spending credits:

* **Weights are a function of the training-side configuration only.**  `train revert=<X>` fixes the
  weights; nothing you do at scoring time changes them.  Conversely, re-scoring an archived checkpoint
  under a different harness configuration cannot change that checkpoint's quality.
* **`offset` is a function of the harness configuration only.**  Two different checkpoints scored under
  the same harness configuration carry the same offset.

Archived from the previous cycle and free to re-score as often as you like (they are already trained):
`previous`, the pre-release checkpoint, and `release`, the current one.
