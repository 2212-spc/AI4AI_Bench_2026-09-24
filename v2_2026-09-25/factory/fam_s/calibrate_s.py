"""Family S calibration: derive per-item tolerances from the reference's own sampling error.

A fixed global tolerance is a guess, and a guess in either direction breaks the task: too tight and a
*correct but less efficient* estimator is marked wrong, too loose and the complete-case decoy slips
through.  So the tolerance is measured instead.

  step 1 (`ref`)   run the reference over K independent worlds, record per-query |error| and its sd,
                   and set tol_q = max(FLOOR, Z * sd_q) rounded up.  The published tolerance is therefore
                   a statement about how much information the data actually contain for that contrast.
  step 2 (`cand`)  run every plausible-but-wrong procedure against those tolerances and require each one
                   to blow at least MIN_FAIL items.  Anything that passes is either a genuine alternative
                   route (then it belongs in ALTERNATIVES) or the task is not discriminating.

Both steps write JSON so they can be run in separate processes.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world_s as W                                                          # noqa: E402
import estimator_s as E                                                      # noqa: E402

FLOOR_D = 0.030          # never publish a tolerance tighter than this on a delta
FLOOR_C = 0.040          # ... or on a censoring fraction
Z = 5.0                  # tolerance in units of the reference's own sd
MIN_FAIL = 2             # a wrong procedure must miss at least this many items


def run_ref(seeds):
    Q, RC = W.queries(), W.report_cells()
    truth = {q["id"]: W.true_delta(q["knob"], q["from"], q["to"], q["baseline"]) for q in Q}
    de, ce, verd = {}, {}, {}
    for s in seeds:
        man, res = W.sample(s)
        d = E.Data(man, res, W.KNOBS, W.TAU)
        ans, cen = E.answer(d, Q, RC)
        for q in Q:
            a = ans[q["id"]]
            verd.setdefault(q["id"], set()).add(a["verdict"])
            if a["verdict"] == "identified":
                de.setdefault(q["id"], []).append(a["delta"] - truth[q["id"]])
        for rid, c in RC:
            ce.setdefault(rid, []).append((cen[rid] if cen[rid] is not None else -1) - W.p_guard(c))
    tol_d, tol_c = {}, {}
    for k, v in de.items():
        tol_d[k] = max(FLOOR_D, Z * float(np.std(v)) + abs(float(np.mean(v))))
        tol_d[k] = math.ceil(tol_d[k] * 1000) / 1000.0
    for k, v in ce.items():
        tol_c[k] = max(FLOOR_C, Z * float(np.std(v)) + abs(float(np.mean(v))))
        tol_c[k] = math.ceil(tol_c[k] * 1000) / 1000.0
    return {"seeds": list(seeds), "tol_delta": tol_d, "tol_censor": tol_c,
            "ref_bias": {k: round(float(np.mean(v)), 5) for k, v in de.items()},
            "ref_sd": {k: round(float(np.std(v)), 5) for k, v in de.items()},
            "ref_worst_abs": {k: round(float(np.max(np.abs(v))), 5) for k, v in de.items()},
            "ref_censor_worst_abs": {k: round(float(np.max(np.abs(v))), 5) for k, v in ce.items()},
            "ref_verdicts": {k: sorted(v) for k, v in verd.items()},
            "verdict_key": {q["id"]: ("underdetermined" if len(sorted(verd[q["id"]])) == 1
                                      and sorted(verd[q["id"]])[0] == "underdetermined" else "identified")
                            for q in Q}}


def score(ans, cen, tol, truth, vkey, RC):
    """Item-level scoring exactly as the verifier will do it."""
    bad = []
    for qid, want in vkey.items():
        got = ans.get(qid, {}).get("verdict")
        if got != want:
            bad.append((qid, "verdict:%s" % got))
        elif want == "identified":
            e = abs(ans[qid]["delta"] - truth[qid])
            if e > tol["tol_delta"][qid]:
                bad.append((qid, round(e, 4)))
    for rid, c in RC:
        g = cen.get(rid)
        t = W.p_guard(c)
        if g is None:
            bad.append((rid, "missing"))
        elif abs(g - t) > tol["tol_censor"][rid]:
            bad.append((rid, round(abs(g - t), 4)))
    return bad


def run_cand(tol, seeds):
    Q, RC = W.queries(), W.report_cells()
    truth = {q["id"]: W.true_delta(q["knob"], q["from"], q["to"], q["baseline"]) for q in Q}
    vkey = tol["verdict_key"]
    out = {}
    pool = ([("REFERENCE", E.REF)] + sorted(E.ALTERNATIVES.items()) + sorted(E.BORDERLINE.items())
            + sorted(E.CANDIDATES.items()))
    for name, spec in pool:
        fails = []
        for s in seeds:
            man, res = W.sample(s)
            d = E.Data(man, res, W.KNOBS, W.TAU)
            ans, cen = E.answer(d, Q, RC, spec)
            fails.append(score(ans, cen, tol, truth, vkey, RC))
        out[name] = {"n_fail_per_seed": [len(f) for f in fails], "example": fails[0][:6]}
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    path = os.path.join(HERE, "calibration.json")
    if what == "ref":
        seeds = [int(x) for x in sys.argv[2].split(",")]
        r = run_ref(seeds)
        json.dump(r, open(path, "w"), indent=1)
        print(json.dumps({k: r[k] for k in ("tol_delta", "tol_censor", "ref_worst_abs",
                                            "ref_censor_worst_abs", "verdict_key")}, indent=1))
    else:
        tol = json.load(open(path))
        seeds = [int(x) for x in sys.argv[2].split(",")]
        r = run_cand(tol, seeds)
        tol["candidate_scan"] = r
        tol["candidate_seeds"] = seeds
        ok_ref = all(n == 0 for n in r["REFERENCE"]["n_fail_per_seed"])
        ok_alt = all(all(n == 0 for n in r[a]["n_fail_per_seed"]) for a in E.ALTERNATIVES)
        ok_cand = all(min(r[c]["n_fail_per_seed"]) >= MIN_FAIL for c in E.CANDIDATES)
        tol["G_reference_passes"] = ok_ref
        tol["G_alternatives_pass"] = ok_alt
        tol["G_every_candidate_fails"] = ok_cand
        tol["borderline"] = {b: r[b]["n_fail_per_seed"] for b in E.BORDERLINE}
        json.dump(tol, open(path, "w"), indent=1)
        for k, v in r.items():
            print("%-42s fails=%s %s" % (k, v["n_fail_per_seed"], v["example"][:3]))
        print("REF_OK", ok_ref, "ALT_OK", ok_alt, "CAND_OK", ok_cand)
