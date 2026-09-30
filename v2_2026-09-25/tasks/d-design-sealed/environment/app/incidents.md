# Incident notes for the 2026-06 evaluation sweep

**INC-4471 - evaluation sink outage (workers w03, w07).**  Between 2026-06-27T00:00:00Z and
2026-07-03T12:00:00Z the primary results writer on workers w03 and w07 dropped its connection.  Runs that
*started* in that window on those two workers never got a row into the archive at all - the number never
left the worker.  The failure was in the writer, not in the evaluation: it does not depend in any way on
what the run scored.

Some of those values were later scraped out of the per-worker stdout tails and are kept under
`recovered/`, one shard per collection pass.  The stdout banner prints the score as a percentage, so the
shards carry `pass_rate_pct`; it is the same quantity as `pass_rate`, in different units.  The scrape was
opportunistic:

  * it also re-captured runs whose primary row was fine, so a run can appear in both places;
  * it also captured runs whose row had already been removed by the quality screen;
  * the shards were assembled from planning tickets, so they contain a few run ids for runs that were
    never launched at all, and a few rows appear twice.

Where a run appears more than once the values agree; treat the shards as a value source, never as a run
list.

**Write-time quality screen.**  A result row is written the moment its run finishes, and the archiver
checks it against the quality floor in force at that moment; a row under the floor is not kept.  The
removals are listed in `retention_log.csv` with reason `below_retention_floor` and the instant of the
check, and the floors are versioned in `retention_policy.json`.  The screen has run unchanged for the
whole sweep.  It is part of the archiver, so the only thing it ever inspects is a row that reached the
archive.

**INC-4506 - lost shard file.**  The archiver keeps result rows in one file per worker per UTC day.
During a storage migration the file holding worker w01's rows for 2026-07-04 was lost along with its
replica; the rows it held are gone and are listed in `retention_log.csv` with reason `shard_file_lost`.
The migration walked the store by file name.  Nothing in it looked inside a file, so which rows were lost
has nothing to do with what any of them said.

**INC-4530 - sink outage on the late campaign (worker w09).**  After the main sweep closed,
three short follow-up campaigns were run on w09 between 2026-07-11T00:00:00Z and 2026-07-15T00:00:00Z to try
two curriculum settings that the sweep had not covered.  w09 had been rebuilt during the
storage migration and came back without its results-writer credentials, which nobody noticed until the
campaigns were over: none of its rows reached the archive.  As with INC-4471 the failure is in the
writer, so which runs lost their rows has nothing to do with what they scored, and no scrape was
attempted - the worker was reimaged.  Runs from other workers that finished on the same days were written
and screened normally.

**Scheduler accounting.**  The scheduler keeps its own ledger, on a different host from the results
writer and untouched by any of these incidents: `runtime.csv` records when each launched run finished, and
`daily_digest.csv` records, for each UTC day, how many runs finished that day and what their `pass_rate`
values added up to.  The digest is written at 23:59 from the scores the scheduler received directly from
each worker, so its totals still cover runs whose rows the archive screened out, lost, or never received.
`daily_digest.csv` is exact to the four decimals it prints.

**Not affected.**  `plan.csv`, `launch_log.csv`, `retention_log.csv`, `runtime.csv` and
`daily_digest.csv` are complete for the whole sweep, including the late campaigns.
