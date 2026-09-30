# Serving incident: cohort-mix drift

The release dashboard stayed nearly flat while quality complaints rose in the watched customer cohort.
Three model settings changed together. The dashboard is an aggregate of a target cohort and a stress cohort;
its mixture changed during the release, so a dashboard intervention is not a causal measurement.

Recipe / shipped / permitted repair cap:

    router       1.000   0.683   0.746
    calibration  1.000   1.236   1.189
    features     1.000   0.679   0.744

Dashboard before/after: 0.5595 -> 0.5448. Frozen target before/after: 0.9047 -> 0.6709.

Use `/app/bin/lab spec`, then `lab probe` with a config and cohort `dashboard`, `frozen`, or `stress`.
The frozen cohort is the target of the repair. You have 32 credits. Omitted config keys keep shipped values.

Write `/app/report.json` as `{"cause":"router|calibration|features","fix":{"knob":value,...}}`.
The grader checks hidden frozen-cohort recovery, a counterfactual held-knob attribution, and that the fix moves
the named cause. A one-knob reset can be insufficient because calibration optimum moves with routing and features.
