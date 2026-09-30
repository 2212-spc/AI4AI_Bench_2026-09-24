# Schedule the batch size for one fixed-token training run

`solver-7b` is about to be pre-trained for a fixed number of tokens. The token budget is split into
**40 equal segments**. Before each segment you choose the **batch size**; everything else in the recipe is
fixed. Your goal is to finish the whole run in as few **GPU-seconds** as possible.

Two things set the cost of a segment:

* the optimizer needs some number of steps to get through the segment's tokens, and that number depends on
  the batch size you picked - a bigger batch needs fewer steps, but with diminishing returns;
* each step costs `t_overhead + batch / throughput` seconds, both of which you are told exactly.

The team has never characterised the first part. What they do know: the corpus is not homogeneous - the data
schedule switches source partway through the run, and nobody has checked what that does to the step counts.
**The switch is scheduled per run.** Every run of this job gets its own switch point, drawn anywhere between
40% and 80% of the way through the run (segments 16-32). The phase-2 corpus is assembled per run as well,
so how much the switch costs, and how the cost grows after it, also vary run to run (by up to +-85%). Where
and how the dev replica switched tells you nothing about the real run. The ramp *before* the switch is a
property of the model and is the same every time.

## The lab

Run `/app/bin/lab spec` first. `lab profile` runs segments of a **dev replica** of the same run at batch
sizes you choose, and reports the realised step count and seconds for each. Profiling costs exactly the
seconds it simulates, out of a budget of **8.3 GPU-seconds** - about 2.2 times what the run itself takes -
so a profile at a big batch is expensive. `lab evaluate` runs your current `/app/sched.py` end to end on a
dev replica (cost 0.7, i.e. 9% of the budget).

## Deliverable: `/app/sched.py`

A module defining `run(env)`, run in a sandbox on **5 hidden runs** of the same service (see
`/app/docs/sched_api.md`). It must call `env.run_segment(batch)` exactly 40 times. Your scheduler sees
each segment's realised step count as it goes - it is a closed-loop controller, not a fixed list.

## How it is graded (mechanically, against the simulator; only the file counts)

Let `T` be your mean total seconds over the hidden runs. With `T_const` the time of the best possible
**constant** batch size and `T_ref` the time of a well-tuned adaptive reference scheduler, both measured on
those same hidden runs,

    s = (T_const - T) / (T_const - T_ref)

1. **R0** - `sched.py` completes all 40 segments on every hidden run, without crashing and without an
   out-of-range batch.
2. **R1** - `s >= 0.90`.

On the dev replica the best constant batch takes about **3.78 s** and the reference scheduler about
**3.47 s**. Both bars are re-measured on the hidden runs, so aim to beat the reference, not a number.

Only `/app/sched.py` is graded. Write it, check it with `lab evaluate`, then stop.
