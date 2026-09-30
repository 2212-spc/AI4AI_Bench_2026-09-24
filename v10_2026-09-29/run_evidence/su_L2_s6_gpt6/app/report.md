# Nightly classifier deployment

## Changes
- Retained the numpy two-hidden-layer MLP, training-only standardization, AdamW, batch 128, learning rate 0.003, and warmup/cosine schedule. The command-line interface and integer prediction format are unchanged.
- Replaced blind transfer of sample-tuned decay 3.0 with dataset-size-dependent decay. Three independent members use exponents 0.75, 1.0 and 0.5 relative to 4,000 training rows: production decay values are 0.3172, 0.1500 and 0.6708. Average their class probabilities. At sample size, all use the original decay 3.0.
- Restrict numerical-library threads to one; bound aggregate training CPU to 100 seconds, leaving headroom under the 120-second limit. The guard logs any member that cannot finish its requested steps. Chunk prediction to bound memory.

## Rationale and limits
The original sweep establishes a good setting for 4,000 noisy examples, not an optimum for 80,000. Equal passes do not mean equal optimization or regularization: production has 20 times as many updates and additional independent evidence. Decoupled decay also acts on every update. Reducing decay is a transfer hypothesis, not a mathematical guarantee that preserving integrated shrinkage preserves the fitted objective. The ensemble brackets the uncertainty in the scaling rate and reduces initialization variance. No class is excluded or relabeled; class imbalance alone does not establish asymmetric label noise.

Earlier local trials with pseudo-label filtering and alternative simple models did not justify replacing the MLP. Repeating sample rows does not add independent examples and is unsuitable for selecting full-corpus regularization. It is used only for runtime and output-contract testing. No fitted sample/dev models, rows, feature statistics, or gold-derived calibration parameters are shipped.

## Validation
- Sample training, 1,000 steps, seed 0: gold dev accuracy 0.744; valid integer output.
- Sample training, 1,000 steps, seed 1: gold dev accuracy 0.740; valid integer output.
- The exact 20,000-step CLI completed on an 80,000-row repeated-sample resource fixture with 20,000 prediction rows. trained 3 members; 52.10s CPU, 52.10s wall. Output shape, int64 dtype and class range passed. This measures resources only, NOT full-corpus accuracy.
- Repository artifact audit passed; size 10505 bytes.

**Production score is unmeasured.** Neither the 80,000 independent training rows nor the held-out gold test labels are accessible. These checks cannot certify the requested 0.805 mean test accuracy; the configuration is a conservative transfer candidate, not a verified pass.
