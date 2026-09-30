# Recipe change triage under a compute budget

You own the MiniLab training recipe. `/app/RECIPE.md` describes the current production recipe, `/app/CHANGES.md`
lists six proposed changes (c1-c6) and `/app/notes/teammate_ablation.md` has a teammate's quick ablation and plan.

Decide which subset of the six changes to ship so that the **expected validation MSE (over training seeds) of the
shipped recipe is as low as possible**.

You can train the recipe on the shared cluster with `/app/lab` (run `python3 /app/lab --help`). Every `lab run` costs one
run; the budget for this task is **80 runs** in total and further runs are refused. `lab status` and
`lab history` are free. There is no other way to train the real model; the training data are not available locally.

Deliverables:
1. `/app/decision.json`: `{"ship": ["c.."], "predicted_val_mse": <float>}` where `predicted_val_mse` is your
   estimate of the expected validation MSE (over seeds) of the recipe with exactly the shipped changes.
2. `/app/REPORT.md`: a short justification with the evidence you relied on.

Grading (hidden truth = many-seed evaluation of every subset): you pass if the true expected val MSE of your
shipped subset is within 4.5% of the best of all 64 subsets **and** your `predicted_val_mse` is within
6% of the true expected val MSE of the subset you shipped.

You have 3600 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
