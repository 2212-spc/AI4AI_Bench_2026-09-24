# solver-7b pre-training, batch-size notes

* the run is cut into 40 equal-token segments; batch size may be changed at a segment boundary only
* per-step time is `t_overhead + batch/throughput`, both measured on this cluster and stable to ~1%
* realised step counts per segment fluctuate by a few percent run to run
* the data schedule switches source partway through (the "phase 2" corpus); it is scheduled per run, so the
  dev replica's switch is **not** the hidden runs' switch; the switch point is in the
  training config, which the data team owns
* previous runs used a constant batch, tuned once by sweeping a few values on a short replica
* open ticket: "phase 2 seems to need more steps than phase 1 at the same batch" (unresolved, no numbers)
