# Pick the width and learning rate for one fixed-compute pre-training run

Your team has been allocated compute for **exactly one** pre-training run. The allocation fixes the product

    width^2 * steps = 1.007e+12

so choosing a wider model buys capacity and pays for it in steps. You choose the model **width** (from the
allowed list) and the **peak learning rate**; everything else in the recipe is fixed.

The catch the team keeps hitting: above some learning rate a run at a given width becomes unstable and
**diverges** - the loss spikes and the run is lost. A diverged target run is scored as **8.0**, i.e. a
total write-off. Runs a little below that edge are not free either: they train with visible loss-spike damage.
Where the edge sits depends on the width, and the team has never measured how.

## The lab

Run `/app/bin/lab spec` first. You can train small proxy models: `lab train n=512 steps=4000 eta=1e-3`
returns either `diverged: true` (with the step it blew up at) or a noisy `final_loss`. A run costs
`n^2 * steps` from your lab budget of **6.0e+10**, which is about 6.0% of the target run -
enough for many small runs, nowhere near enough for one at the target width.

Repeating the same `(n, steps, eta)` with a different `seed` gives an independent run.

## Deliverable: `/app/config.json`

    {"n": 2048, "eta": 0.001, "predicted_loss": {"lo": 0.0, "hi": 0.0}}

* `n` - the width of the target run, one of [362, 512, 724, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585]. `steps` follows from the budget.
* `eta` - the peak learning rate, in [1e-05, 8e-02].
* `predicted_loss` - an interval for what **your own configuration** scores as the target run: its expected
  final loss, counting a divergence as 8.0. Width at most 0.40.

## How it is graded (mechanically, against the simulator; only the file counts)

1. **R0** - `config.json` is valid: `n` in the allowed list, `eta` in range, interval width <= 0.40.
2. **R1** - your configuration's score is within **0.0600** of the best any allowed (width, learning rate)
   can achieve on this run.
3. **R2** - your interval contains your own configuration's score.

All three must hold. Your lab history is not graded - spend the budget as you see fit. When you are done,
write `/app/config.json` and stop.
