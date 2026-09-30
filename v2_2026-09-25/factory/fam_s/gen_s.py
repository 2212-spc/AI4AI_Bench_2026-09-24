"""Family S generator: recover contrasts from a sweep whose result table is silently incomplete.

Gates
  G1  the reference clears every item on K independent worlds                (the task is solvable)
  G2  every published tolerance is >= 5 sigma of the reference's own error   (tolerances are measured)
  G3  every plausible-but-wrong procedure misses at least 2 items            (no shortcut passes)
  G4  the alternative-but-correct procedures also clear every item           (no single blessed recipe)
  G5  the dead region really is dead, and an alternative world with a
      different effect there produces an observationally identical table     (abstention is provable)
  G6  the guard fraction is not the raw missing fraction anywhere it matters (preemption must be modelled)
  G7  blanket answers - all identified, all underdetermined - both fail      (no base-rate shortcut)
"""
import json, math, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import world_s as W                                                          # noqa: E402
import estimator_s as E                                                      # noqa: E402
import calibrate_s as K                                                      # noqa: E402
from core import harbor                                                      # noqa: E402

REASON = {"no_surviving_run": "every launched run in one of the two cells was removed from the table",
          "combination_never_launched": "the cell exists in principle but the sweep never launched it",
          "level_never_launched": "the level itself never appears in the sweep"}

LAUNCHER = """# `sweeprun` v3.1 - operator notes

## What the sweep was

A replicated factorial over {nknob} knobs. Every cell that policy allowed was launched {reps} times with
independent seeds. Assignment is by design, not by operator choice: there is no selection on anything.

Two policy restrictions applied to the grid itself:

* `precision=fp16` was withdrawn before this sweep started and was never launched.
* `precision=fp8` requires the large-node partition, so it was only ever launched at `batch=1024`.

## manifest.csv

One row per *launched* run: `run_id` plus the knob settings. {nman} rows. This file is written at submit
time and is never edited afterwards.

## results.csv

One row per run that *finished and was recorded*: {nres} rows. A launched run is missing from this file for
exactly one of two reasons, and the log does not record which.

**1. Divergence guard.** `--abort-above-val-loss {tau}` is set fleet-wide. A run whose final validation
loss would exceed {tau} is aborted and its row is discarded; nothing is written. The guard reads the
outcome, so it fires more often exactly where the loss is high.

**2. Preemption.** The cluster preempts jobs by priority class. Priority is assigned from the requested
node count, which on this cluster is a function of `batch` alone:

| batch | nodes | priority class |
|------:|------:|:---------------|
{ptable}

The scheduler looks at the priority class and at nothing else - it cannot see the loss, and it decides
before the run produces one. Preemption rates per class are not published anywhere and drift week to week.

## val_loss

Final validation loss, one number per recorded run. Run-to-run spread within a cell is dominated by seed
noise and is well described by a normal law with a common scale across cells.
"""

INSTR = """# What the sweep would have said if you could see all of it

`/app` holds a finished hyper-parameter sweep: `manifest.csv` (every run that was launched),
`results.csv` (every run that was recorded), `sweep.json` (the grid and the launcher constants) and
`launcher.md` (how the launcher behaved). `results.csv` is missing {nmiss} of the {nman} launched runs.
`launcher.md` explains the two mechanisms that remove rows; only one of them looks at the outcome.

`queries.json` asks for {nq} contrasts. Each is "hold this baseline cell fixed, move one knob from A to B,
what happens to the mean final validation loss" - a difference of two cell means, in loss units, at the
stated baseline. `report_cells.json` asks, for {nr} named cells, what fraction of the runs launched there
were removed *by the divergence guard specifically*.

Some of the contrasts cannot be answered from this sweep at all. Saying so is a correct answer and is
worth exactly as much as a number; inventing a number for one of them is wrong, and so is refusing to
answer one that the data do determine.

## Deliverable

`/app/answers.json`:

```json
{{
  "queries": {{
    "q01": {{"verdict": "identified", "delta": -0.0421}},
    "q07": {{"verdict": "underdetermined", "reason": "no_surviving_run"}}
  }},
  "censoring": {{"r1": 0.0, "r2": 0.31}}
}}
```

Every query id in `queries.json` and every cell id in `report_cells.json` must appear. `delta` is
`mean(B) - mean(A)` in loss units. `reason` must be exactly one of:

{reasons}

## How it is graded

* All-or-nothing over {nitems} items: {nq} verdicts (with the reason code where you abstain), {ndelta}
  numbers, {nr} fractions.
* Each query in `queries.json` carries its own `tol`, and each report cell its own; an answer counts if it
  is within that tolerance of the truth. The tolerances are not arbitrary - each is five standard
  deviations of the sampling error of a correct estimator on this much data, so a correct procedure
  clears them with room to spare. If your own procedure's error is anywhere near its tolerance, that is
  evidence the procedure is wrong, not that the tolerance is tight. You can check yours: you have enough
  information to simulate the whole pipeline and re-estimate.
* A fraction must lie in [0, 1].

Python 3 with numpy is available and there is no network.
"""

README = """# {slug}

Generated by the heterogeneous task factory, family **S** (recovery under outcome-dependent missingness),
instance seed `{seed}`.

## Difficulty

The sweep is a clean replicated factorial - no confounding, no operator selection, nothing to adjust for.
The whole difficulty is in the {nmiss} rows that are not there. Dropping them and taking cell means, which
is what every sweep-analysis script does, is biased downward by up to {cc_worst:.3f} loss units in exactly
the cells the interesting contrasts land in, and blows {cc_fail} of {nitems} items. The over-correction -
abstaining wherever rows are missing - blows {ab_fail}. Imputing at the guard blows {im_fail}. Inverting
the observed survival rate, which is the textbook move for censored data, blows {sv_fail}, because the
second removal mechanism has nothing to do with the outcome and has to be modelled separately before the
first one can be read off.

Four of the {nq} contrasts are not identified, for three structurally different reasons, and the reason
code is graded: one region has no surviving run at all, one cell was never launched though its levels
were, and one level was never launched. An agent that treats "underdetermined" as a single bucket loses.

## Reference solution

`solution/ref_solve.py` reads exactly the agent's files and fits a saturated truncated-normal MLE - one
mean per launched cell plus one common scale - so nothing depends on an assumed response surface. Its
worst error over {kseeds} independent worlds is {ref_worst:.4f} against a floor tolerance of 0.030.

## Verification

Tolerances are measured, not chosen: five sigma of the reference's own sampling error over {kseeds}
worlds. `authoring/certificate.json` records the full candidate scan, the dead-region certificate and the
indistinguishable-world construction.
"""


def blanket(kind, key):
    if kind == "all_identified":
        qs = {q: {"verdict": "identified", "delta": 0.0} for q in key["queries"]}
    else:
        qs = {q: {"verdict": "underdetermined", "reason": "no_surviving_run"} for q in key["queries"]}
    return {"queries": qs, "censoring": {r: 0.0 for r in key["censoring"]}}


def run_verifier(tests, app):
    logs = tempfile.mkdtemp()
    env = dict(os.environ, APP=app, TESTS=tests, REWARD_DIR=logs)
    p = subprocess.run([sys.executable, os.path.join(tests, "verify_s.py")], env=env,
                       capture_output=True, text=True, timeout=900)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"reward": None, "stderr": p.stderr[-500:]}


def dead_region_certificate(free_terms=(0.20, 0.35, 0.60)):
    """The three-way term lives where nothing survives, so the data cannot see it.

    For each candidate value we bound the total-variation distance between the observed-data laws by the
    probability that at least one run in the dead region survives the guard.
    """
    dead = [c for c in W.cells() if c["lr"] == 1e-3 and c["init_scale"] == 1.0 and c["sched"] == "cosine"]
    out = []
    for f in free_terms:
        p = max(W.p_survive(c, f) for c in dead)
        n = W.REPS * len(dead)
        out.append({"free_term": f, "max_per_run_survival": p, "n_runs_in_region": n,
                    "prob_any_row_appears": 1.0 - (1.0 - p) ** n,
                    "true_delta_q07": round(W.true_delta("lr", 3e-4, 1e-3,
                                                         dict(W.BASELINE, init_scale=1.0), f), 4)})
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260925
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/bench2/tasks/s-censored-a"
    slug = os.path.basename(out)
    cal = json.load(open(os.path.join(HERE, "calibration.json")))

    Q, RC = W.queries(), W.report_cells()
    truth = {q["id"]: W.true_delta(q["knob"], q["from"], q["to"], q["baseline"]) for q in Q}
    man, res = W.sample(seed)

    # ---- key ---------------------------------------------------------------------------------------
    d = E.Data(man, res, W.KNOBS, W.TAU)
    reasons = {}
    for q in Q:
        ka = E._key(dict(q["baseline"], **{q["knob"]: q["from"]}), W.KNOBS)
        kb = E._key(dict(q["baseline"], **{q["knob"]: q["to"]}), W.KNOBS)
        for cfg_v, k in ((q["from"], ka), (q["to"], kb)):
            if cfg_v not in d.levels[q["knob"]] and k not in d.launched:
                reasons[q["id"]] = "level_never_launched"
            elif k not in d.launched:
                reasons.setdefault(q["id"], "combination_never_launched")
            elif len(d.obs.get(k, [])) == 0:
                reasons.setdefault(q["id"], "no_surviving_run")
    key = {"queries": {}, "censoring": {}}
    for q in Q:
        qid = q["id"]
        if cal["verdict_key"][qid] == "underdetermined":
            key["queries"][qid] = {"verdict": "underdetermined", "reason": reasons[qid]}
        else:
            key["queries"][qid] = {"verdict": "identified", "delta": round(truth[qid], 6),
                                   "tol": cal["tol_delta"][qid]}
    for rid, c in RC:
        key["censoring"][rid] = {"value": round(W.p_guard(c), 6), "tol": cal["tol_censor"][rid]}

    # ---- gates -------------------------------------------------------------------------------------
    cert = {"seed": seed, "n_launched": len(man), "n_recorded": len(res),
            "n_missing": len(man) - len(res),
            "G1_reference_clears_every_item": cal.get("G_reference_passes"),
            "G2_tolerances_are_5_sigma_of_reference": {
                "ref_sd": cal["ref_sd"], "ref_worst_abs": cal["ref_worst_abs"],
                "tol_delta": cal["tol_delta"], "tol_censor": cal["tol_censor"],
                "ok": all(cal["tol_delta"][q] >= min(0.030, 5 * cal["ref_sd"][q] - 1e-9)
                          for q in cal["ref_sd"])},
            "G3_every_wrong_candidate_fails": cal.get("G_every_candidate_fails"),
            "G4_alternative_correct_routes_pass": cal.get("G_alternatives_pass"),
            "candidate_scan": {k: v["n_fail_per_seed"] for k, v in cal["candidate_scan"].items()},
            "borderline": cal.get("borderline"),
            "calibration_seeds": cal["seeds"], "candidate_seeds": cal["candidate_seeds"]}

    dead_rows = [r for r in res if r["lr"] == 1e-3 and r["init_scale"] == 1.0 and r["sched"] == "cosine"]
    cert["G5_dead_region_has_no_row"] = (len(dead_rows) == 0)
    cert["G5_indistinguishable_worlds"] = dead_region_certificate()
    cert["G5_ok"] = bool(cert["G5_dead_region_has_no_row"]
                         and max(x["prob_any_row_appears"] for x in cert["G5_indistinguishable_worlds"]) < 1e-3
                         and len({x["true_delta_q07"] for x in cert["G5_indistinguishable_worlds"]}) > 1)

    raw = {}
    for rid, c in RC:
        k = E._key(c, W.KNOBS)
        raw[rid] = round(1.0 - len(d.obs.get(k, [])) / float(d.launched[k]), 5)
    cert["G6_raw_missing_vs_truth"] = {rid: {"raw_missing": raw[rid],
                                             "true_guard_fraction": key["censoring"][rid]["value"],
                                             "tol": key["censoring"][rid]["tol"]} for rid, _ in RC}
    cert["G6_ok"] = any(abs(raw[rid] - key["censoring"][rid]["value"]) > key["censoring"][rid]["tol"]
                        for rid, _ in RC)

    # ---- export ------------------------------------------------------------------------------------
    app = out + "/environment/app"
    os.makedirs(app, exist_ok=True)
    harbor.csv_rows(app + "/manifest.csv", W.MAN_COLS, man)
    harbor.csv_rows(app + "/results.csv", W.RES_COLS, res)
    spec = {"knobs": W.KNOBS, "levels": {k: v for k, v in W.LEVELS.items()},
            "replicates_per_launched_cell": W.REPS, "guard_val_loss": W.TAU,
            "columns": {"manifest.csv": W.MAN_COLS, "results.csv": W.RES_COLS}}
    harbor.w(app + "/sweep.json", json.dumps(spec, indent=1))
    harbor.w(app + "/queries.json", json.dumps(
        [{"id": q["id"], "knob": q["knob"], "from": q["from"], "to": q["to"], "baseline": q["baseline"],
          "tol": key["queries"][q["id"]].get("tol", cal["tol_delta"].get(q["id"], 0.030))} for q in Q],
        indent=1))
    harbor.w(app + "/report_cells.json", json.dumps(
        [{"id": rid, "cell": c, "tol": key["censoring"][rid]["tol"]} for rid, c in RC], indent=1))
    ptable = "\n".join("| %d | %d | %s |" % (b, n, p) for b, n, p in
                       [(256, 2, "guaranteed"), (512, 4, "standard"), (1024, 8, "spot / preemptible")])
    harbor.w(app + "/launcher.md", LAUNCHER.format(nknob=len(W.KNOBS), reps=W.REPS, tau=W.TAU,
                                                   nman=len(man), nres=len(res), ptable=ptable))

    agent_timeout = 5400
    nq, nr = len(Q), len(RC)
    ndelta = sum(1 for v in key["queries"].values() if v["verdict"] == "identified")
    harbor.w(out + "/instruction.md", INSTR.format(
        nmiss=len(man) - len(res), nman=len(man), nq=nq, nr=nr, nitems=nq + nr, ndelta=ndelta,
        reasons="\n".join("* `%s` - %s" % (k, v) for k, v in sorted(REASON.items())))
        + harbor.SUFFIX_T.format(t=agent_timeout))
    harbor.w(out + "/task.toml", harbor.task_toml(
        ["/app/answers.json"], "S",
        ["missing-data", "censoring", "identification", "abstention", "hyperparameter-sweep"],
        agent_timeout, 2.5, verifier_timeout=600))
    harbor.w(out + "/environment/Dockerfile",
             "FROM python:3.12-slim\nRUN pip install --no-cache-dir %s\nWORKDIR /app\nCOPY app/ /app/\n"
             % harbor.NUMPY)
    harbor.w(out + "/tests/Dockerfile",
             "FROM python:3.12-slim\nRUN pip install --no-cache-dir %s\nCOPY . /tests/\n" % harbor.NUMPY)
    harbor.w(out + "/tests/test.sh", harbor.TEST_SH.format(verify="verify_s.py"), 0o755)
    harbor.w(out + "/tests/verify_s.py", open(os.path.join(HERE, "verify_s.py")).read())
    harbor.w(out + "/tests/key.json", json.dumps(key, indent=1))
    harbor.w(out + "/solution/ref_solve.py", open(os.path.join(HERE, "ref_solve.py")).read())
    harbor.w(out + "/solution/estimator_s.py", open(os.path.join(HERE, "estimator_s.py")).read())
    harbor.w(out + "/solution/solve.sh",
             "#!/bin/bash\nset -e\ncp /app/../solution/*.py /tmp/ 2>/dev/null || true\n"
             "python3 \"$(dirname \"$0\")/ref_solve.py\" /app\n", 0o755)

    # ---- post-export self-checks that need the exported tree ----------------------------------------
    tests = out + "/tests"
    work = tempfile.mkdtemp()
    shutil.copytree(app, work + "/app")
    t0 = time.time()
    subprocess.run([sys.executable, out + "/solution/ref_solve.py", work + "/app"],
                   capture_output=True, text=True, timeout=1800, check=True)
    cert["reference_solve_seconds"] = round(time.time() - t0, 1)
    cert["G1_exported_reference_reward"] = run_verifier(tests, work + "/app")["reward"]
    for kind in ("all_identified", "all_underdetermined"):
        b = tempfile.mkdtemp()
        json.dump(blanket(kind, key), open(b + "/answers.json", "w"))
        cert["G7_" + kind + "_reward"] = run_verifier(tests, b)["reward"]
    cert["G7_ok"] = (cert["G7_all_identified_reward"] == 0 and cert["G7_all_underdetermined_reward"] == 0)

    cert["accepted"] = bool(cert["G1_reference_clears_every_item"] and cert["G1_exported_reference_reward"] == 1
                            and cert["G2_tolerances_are_5_sigma_of_reference"]["ok"]
                            and cert["G3_every_wrong_candidate_fails"] and cert["G4_alternative_correct_routes_pass"]
                            and cert["G5_ok"] and cert["G6_ok"] and cert["G7_ok"])
    harbor.w(out + "/authoring/certificate.json", json.dumps(cert, indent=1))

    cc = cal["candidate_scan"]
    harbor.w(out + "/README.md", README.format(
        slug=slug, seed=seed, nmiss=len(man) - len(res), nq=nq, nitems=nq + nr, kseeds=len(cal["seeds"]),
        ref_worst=max(cal["ref_worst_abs"].values()),
        cc_worst=max(abs(v) for v in [0.0] + [x[1] for x in cc["complete_case"]["example"]
                                              if isinstance(x[1], float)]),
        cc_fail=min(cc["complete_case"]["n_fail_per_seed"]),
        ab_fail=min(cc["correct_fit_but_abstains_on_any_hole"]["n_fail_per_seed"]),
        im_fail=min(cc["impute_at_guard"]["n_fail_per_seed"]),
        sv_fail=min(cc["survival_rate_with_preemption_model"]["n_fail_per_seed"])))
    print(json.dumps({k: v for k, v in cert.items()
                      if k not in ("G6_raw_missing_vs_truth", "G2_tolerances_are_5_sigma_of_reference")},
                     indent=1)[:4000])
    print("exported" if cert["accepted"] else "REJECTED", out)
    return 0 if cert["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
