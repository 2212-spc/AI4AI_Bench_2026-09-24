# Scaling and schedule analysis

Read the manual and notebook before running experiments. Used 10 runs and 1.99623e19 FLOPs, below the 2e19 budget. Raw measurements are in lab_runs.jsonl.

The notebook treats unfinished cosine checkpoints as completed runs. Paired cosine/WSD runs and WSD cooldown branches show an additive schedule penalty, so that interpretation biases the fitted scaling exponents.

Finished-run and branch data support L(N,D) = E + A*(N/1e8)^(-alpha) + B*(D/1e9)^(-beta). The completed-data-only fit has alpha=0.4032, beta=0.3273 and predicts production loss 2.362. A joint fit that explicitly includes the schedule penalty uses all checkpoints, has alpha=0.4059, beta=0.3316 and predicts 2.372. Its residual RMS is 0.0058 nats. A fit with free penalties at each sampled schedule fraction predicts 2.371, which checks the smooth schedule fit. All these fits select option B.

The joint model has E=1.70274, A=0.970835, B=1.652168. It models the cosine checkpoint penalty as 0.185510*(1+cos(pi*f))/2 and the WSD checkpoint penalty as 0.187764*min(1,(1-f)/0.2)^1.215467. These penalties vanish at completion. fit_scaling.py and fit_joint.py reproduce the main fits.

For q3, the N and data terms cancel because the two evaluations have the same model size and tokens seen. The fitted cosine penalty at f=0.4 is 0.1214 nats; direct matched checkpoint/branch differences are 0.1168 and 0.1176 at the two tested model sizes.

For q4, the loss difference is B*60^(-beta)*(f^(-beta)-1) plus the cosine penalty. Both terms decrease with f, so the full interval over the documented unknown f in [0.5,0.9] has endpoints 0.01965 and 0.20259 nats. This range represents the unknown stopping time. The q2 and q3 intervals are degenerate, as required; they do not include statistical uncertainty.

For q5, the fitted cosine loss at 80% is 0.0281 nats lower than the completed WSD loss at 60%. This comparison does not depend on f_stop, and so is supported rather than undetermined.
