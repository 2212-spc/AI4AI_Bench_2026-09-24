"""Reference solution for family B.  Reads only what the agent is given; writes /app/answers.json.

    python3 ref_solve_b.py /app
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bounds_b as B                                                          # noqa: E402


def main():
    app = sys.argv[1] if len(sys.argv) > 1 else "/app"
    arch = B.Arch(app)
    queries = json.load(open(os.path.join(app, "queries.json")))
    out = B.answer(arch, queries, B.REF)
    for sec in ("cells", "contrasts"):
        for qid in out[sec]:
            out[sec][qid] = {k: round(v, 6) for k, v in out[sec][qid].items()}
    json.dump(out, open(os.path.join(app, "answers.json"), "w"), indent=1)
    print(json.dumps(out)[:1200])


if __name__ == "__main__":
    main()
