# Nightly classifier changes

The repository now trains four numpy MLPs from the command-line corpus and
averages their class probabilities. The production command is unchanged. No
training rows, dev rows, normalization statistics, or fitted weights are shipped.

The original sample sweep favored weight decay 3.0 because weaker decay allowed
substantial memorization of noisy crowd labels. Experiments on 1,000, 2,000, and
4,000 training rows showed that the useful decay strength decreases with more
data. Matching epochs alone does not establish that the sample's best
regularization remains appropriate for a corpus twenty times larger.

I replaced fixed decay with corpus-size scaling and used multiple widths and
scaling powers to hedge the uncertain extrapolation. At 80,000 rows the members
use (width, decay) = (256, 0.3354), (256, 0.1500), (128, 0.1057), and
(128, 0.3354). Each trains for all 20,000 requested steps. Learning rate, batch
size, and warmup/cosine scheduling remain as in the original job.

The loss now scores crowd labels using `0.75 * softmax(logits) + 0.25 / K`.
This assumes approximately uniform label corruption and reduces pressure to
memorize inconsistent labels. Noise rates of 0.2–0.3 helped several sample
configurations; feature-noise augmentation did not, so it remains disabled.
The ensemble also reduces sensitivity to initialization and decay choice.

Validation:

- At 1,000 steps per member on the 4,000-row sample, the final ensemble obtained
  gold-dev accuracy **0.720 for seed 0 and 0.714 for seed 1** (mean **0.717**).
  The original tuning log's winning single-model seed-0 result was **0.685**.
  These are development results used for model selection, not held-out estimates.
- Numerical finite differences verified the loss gradient, including zero-noise
  and label-smoothing cases.
- The unchanged production CLI completed 20,000 steps per member on an
  80,000-row timing fixture with 20,000 prediction rows in **71.8 CPU seconds**,
  using **86.4 MiB peak resident memory**. BLAS uses one thread. The fixture
  repeats available rows and therefore measures resources only.
- Predictions have shape `(20000,)`, dtype int64, and class IDs in 0–9.

The full corpus and held-out gold test set were unavailable. The production
accuracy target of 0.785 is therefore **not verified**; the remaining uncertainty
is chiefly regularization extrapolation and the uniform-noise assumption.
