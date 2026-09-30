"""Reference solution for family S.  Reads exactly the files the agent gets - no privileged access.

    python3 ref_solve.py /app            ->  /app/answers.json
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimator_s as E                                                      # noqa: E402


def g(v):
    """Canonical string form of a knob value, matching how the CSVs are printed."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return "%g" % v
    return str(v)


def load(app):
    spec = json.load(open(os.path.join(app, "sweep.json")))
    knobs, tau = spec["knobs"], spec["guard_val_loss"]
    man = list(csv.DictReader(open(os.path.join(app, "manifest.csv"))))
    res = list(csv.DictReader(open(os.path.join(app, "results.csv"))))
    return spec, E.Data(man, res, knobs, tau)


def main():
    app = sys.argv[1] if len(sys.argv) > 1 else "/app"
    spec, d = load(app)
    queries = json.load(open(os.path.join(app, "queries.json")))
    cells = json.load(open(os.path.join(app, "report_cells.json")))
    Q = [{"id": q["id"], "knob": q["knob"], "from": g(q["from"]), "to": g(q["to"]),
          "baseline": {k: g(v) for k, v in q["baseline"].items()}} for q in queries]
    RC = [(c["id"], {k: g(v) for k, v in c["cell"].items()}) for c in cells]
    ans, cen = E.answer(d, Q, RC)
    out = {"queries": {}, "censoring": {r: (0.0 if cen[r] is None else cen[r]) for r, _ in RC}}
    for q in Q:
        a = ans[q["id"]]
        out["queries"][q["id"]] = ({"verdict": "identified", "delta": round(a["delta"], 5)}
                                   if a["verdict"] == "identified"
                                   else {"verdict": "underdetermined", "reason": a["reason"]})
    json.dump(out, open(os.path.join(app, "answers.json"), "w"), indent=1)
    print(json.dumps(out)[:1500])


if __name__ == "__main__":
    main()
