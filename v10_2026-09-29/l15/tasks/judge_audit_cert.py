"""Pool certificate for judge_audit (every number from judge_audit.certify, i.e. the same policy code the gate uses).

  python3 -m l15.tasks.judge_audit_cert <n_seeds> [reps]

For seeds 0..n-1: the sampled instance (sample_params), its exact truth, and the Monte-Carlo pass rate of every policy.
Writes l15/tasks/judge_audit_cert.json and prints a one-line summary per seed plus pool aggregates."""
import json, os, sys
import numpy as np
from . import judge_audit as J

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_audit_cert.json")


def main():
    n = int(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else J.MC_REPS
    rows = {}
    for s in range(n):
        p = J.sample_params(s)
        ok, info = J.certify(p, reps)
        rows[str(s)] = {"params": p, "ok": ok, **info}
        P = info["policies"]
        print("seed %2d k=%d L0=%4.0f ok=%s D=%+.3f %s | %s" % (
            s, p["k"], p["L0"], ok, info["truth"]["D"], info["truth"]["decision"],
            " ".join("%s=%.2f" % (k, v["pass_rate"]) for k, v in P.items())), flush=True)
    agg = {}
    for k in next(iter(rows.values()))["policies"]:
        v = np.array([r["policies"][k]["pass_rate"] for r in rows.values()])
        b = np.array([r["policies"][k]["bias_q2"] for r in rows.values()])
        agg[k] = {"pass_min": float(v.min()), "pass_mean": round(float(v.mean()), 4), "pass_max": float(v.max()),
                  "bias_q2_mean": round(float(b.mean()), 4), "bias_q2_min": float(b.min()), "bias_q2_max": float(b.max())}
    summ = {"n": n, "reps": reps, "n_ok": sum(r["ok"] for r in rows.values()),
            "decisions": {d: sum(r["truth"]["decision"] == d for r in rows.values()) for d in ("SHIP", "HOLD")},
            "aggregate": agg, "seeds": rows}
    json.dump(summ, open(OUT + ".tmp", "w"), indent=1, default=float); os.replace(OUT + ".tmp", OUT)
    print("n_ok %d/%d  decisions %s" % (summ["n_ok"], n, summ["decisions"]))
    for k, a in agg.items():
        print("  %-26s pass %.2f..%.2f (mean %.2f)  bias_q2 %+.3f..%+.3f" % (k, a["pass_min"], a["pass_max"], a["pass_mean"],
                                                                         a["bias_q2_min"], a["bias_q2_max"]))


if __name__ == "__main__":
    main()
