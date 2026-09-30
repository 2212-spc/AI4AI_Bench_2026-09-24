# Telemetry columns

`/app/data/history.csv` (production, all traffic on A) - one row per (day, hour):

| column | meaning |
|---|---|
| day, dow, hour | calendar day number, weekday, hour of day (00-23) |
| requests | requests that arrived in that hour |
| replicas | replica count the autoscaler ran in that hour |
| utilization | mean fraction of replicas busy |
| abandon_rate | fraction of arriving requests that abandoned in the queue |
| mean_queue_wait_s | mean time in queue over all arriving requests (abandoned ones count the time they waited; served ones that never queued count 0) |
| mean_rating | mean rating of served answers |
| mean_gen_time_s | mean generation time of served answers |
| mean_score | mean per-request score (the metric) over all arriving requests |

Experiment CSVs (`/app/data/abtest_NNN.csv`) - same (day, hour) grain; calendar days continue after the
history (experiments run one after another):

| column | meaning |
|---|---|
| fraction_B | fraction routed to B in that hour |
| replicas, utilization, abandon_rate, mean_queue_wait_s | pool-level, as above |
| requests_A / requests_B | arrivals per arm |
| rating_A / rating_B | mean rating of served answers per arm |
| gen_time_A_s / gen_time_B_s | mean generation time of served answers per arm |
| score_A / score_B | mean per-request score per arm (all arriving requests of that arm) |

Hour-level queue measurements (abandon_rate, mean_queue_wait_s) carry hour-to-hour measurement noise of
roughly 10%; it is unbiased.  Ratings are noisy per request.
