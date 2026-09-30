# Nightly classifier update

The production trainer now computes a ZCA whitening transform from the corpus at startup and applies it to both training and prediction features. The original features have substantial correlation, and whitening made the local gold dev accuracy substantially better than per-feature standardization.

Training uses a classifier-side uniform label-noise likelihood. It gradually enables a 10% corruption model, which reduces the influence of crowd labels that disagree with the emerging clean classifier. The shipped configuration trains three independently initialized 256-unit MLPs, averages their late checkpoints, and averages their class probabilities at prediction time. Weight decay scales with corpus size from the sample-tuned reference of 4,000 rows, so the full corpus is regularized less strongly than the development sample.

On the available 4,000-row crowd sample and 1,000-row gold dev set, the revised approach reached about 70% accuracy across the two production seeds, compared with 65% for the shipped trainer. A full-size synthetic run used about 21–26 CPU seconds per member and about 82 MB peak memory; the repository contains only code and JSON configuration and computes all fitted transforms from the supplied training data.
