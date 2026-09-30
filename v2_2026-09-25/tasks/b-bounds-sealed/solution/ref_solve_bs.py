"""Reference solution for family B, sealed arm.  Reads only what the agent is given.

The mechanism block is *derived*, not pasted: `discover_bs.discover` recovers the screen code and the grid
from the published files, and that recovered mechanism is what the bounds are then computed under.  If the
files did not determine it, this script could not produce it either.

    python3 ref_solve_bs.py /app
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bounds_b as B                                                          # noqa: E402
import discover_bs as D                                                       # noqa: E402


def main():
    app = sys.argv[1] if len(sys.argv) > 1 else "/app"
    found = D.discover(app)
    assert found["screen_code"] and found["batch_code"] and found["grid"], found
    assert abs(found["grid"] - B.GRID) < 1e-12, ("the archive implies a different grid", found["grid"])
    arch = B.Arch(app)
    queries = json.load(open(os.path.join(app, "queries.json")))
    out = B.answer(arch, queries, B.REF)
    for sec in ("cells", "contrasts", "supports"):
        for qid in out[sec]:
            out[sec][qid] = {k: round(v, 6) for k, v in out[sec][qid].items()}
    out.pop("_clamped", None)
    out["mechanism"] = {"screen_code": found["screen_code"], "grid": found["grid"]}
    json.dump(out, open(os.path.join(app, "answers.json"), "w"), indent=1)
    print(json.dumps({"mechanism": out["mechanism"], "supports": out["supports"]}))


if __name__ == "__main__":
    main()
