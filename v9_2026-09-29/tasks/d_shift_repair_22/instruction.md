# Serving incident: cohort-mix drift

The release dashboard stayed nearly flat while quality complaints rose in the watched customer cohort.
Three model settings changed together. The dashboard is an aggregate of a target cohort and a stress cohort;
its mixture changed during the release, so a dashboard intervention is not a causal measurement.

Recipe / shipped / permitted repair cap:

    router       1.000   0.713   0.816
    calibration  1.000   1.142   1.091
    features     1.000   0.740   0.833

Dashboard before/after: 0.4704 -> 0.4422. Frozen target before/after: 0.8977 -> 0.4549.

Use `/app/bin/lab spec`, then `lab probe` with a config and cohort `dashboard`, `frozen`, or `stress`.
The frozen cohort is the target of the repair. You have 32 credits. Omitted config keys keep shipped values.

Write `/app/report.json` as `{"cause":"router|calibration|features","fix":{"knob":value,...}}`.
The grader checks hidden frozen-cohort recovery, a counterfactual held-knob attribution, and that the fix moves
the named cause. A one-knob reset can be insufficient because calibration optimum moves with routing and features.
