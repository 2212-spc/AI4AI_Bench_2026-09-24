# Nightly classifier update

The trainer now uses generalized cross entropy (`q=0.5`) to reduce the influence of the noisy crowd labels. On the available 4,000-row prototype, this was more accurate on the gold development set than ordinary cross entropy across the tested seeds. Weight decay is scaled by the square root of `4000 / n`, so the regularization selected on the prototype does not remain unnecessarily strong when training on all 80,000 rows. The command line interface and integer prediction output are unchanged.

I verified the implementation against the existing prototype data and kept the model within the original two-layer NumPy MLP design. The production run uses the existing 20,000-step schedule; the full-corpus run is expected to remain comfortably within the stated CPU budget.
