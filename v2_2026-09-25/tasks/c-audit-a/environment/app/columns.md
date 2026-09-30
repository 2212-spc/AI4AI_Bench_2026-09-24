# Columns of `log.csv`

One row per completed training run of the MiniStack 1.4B recipe on the shared cluster, 6000 runs.

## Set before the run (launch settings)
| column | values |
|---|---|
| `lr_scale` | 0.5, 1.0, 2.0 |
| `warmup` | 200, 1000 |
| `micro_bs` | 4, 8, 16 |
| `grad_accum` | 1, 2, 4 |
| `zero_stage` | 1, 2 |
| `offload` | 0, 1 |
| `seq_len` | 1024, 2048 |
| `act_ckpt` | 0, 1 |
| `dropout` | 0.0, 0.1 |
| `opt_eps` | 1e-08, 1e-06 |

## Probed on the assigned node before step 0
Measured by the launcher's hardware check, *before* any of the settings above take effect, and not
influenced by any of them.

| column | meaning |
|---|---|
| `nccl_bw_gbps` | all-reduce bandwidth measured by the pre-flight NCCL probe |
| `sm_clock_mhz` | SM clock reported by the pre-flight probe |

## Measured during the run
| column | meaning |
|---|---|
| `grad_norm_p95` | 95th percentile of the global grad norm over the run |
| `throughput_toks_s` | mean training throughput |
| `step_time_ms` | mean optimiser step wall time |

## Outcome
| column | meaning |
|---|---|
| `val_loss` | final validation loss, lower is better |

The cluster has been through more than one hardware refresh and jobs were placed by hand and by the
queue, not at random.  Nothing else about the placement was recorded.
