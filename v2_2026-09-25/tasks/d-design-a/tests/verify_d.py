"""Grade `d-design-a`.  Compares against the key only - no archive reading, no re-derivation."""
import json, os, sys

KEY = json.load(open(os.path.join(os.environ.get("TESTS", "/tests"), "key.json")))
APP = os.environ.get("APP", "/app")
sys.path.insert(0, os.environ.get("TESTS", "/tests"))
import harbor as H


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    failed, by = [], {}
    try:
        a = json.load(open(os.path.join(APP, "answers.json")))
        assert isinstance(a, dict)
    except Exception as e:
        return H.reward(False, {"error": "answers.json unreadable: %s" % type(e).__name__})
    tol = KEY["tol"]
    for sec in ("contrasts", "values", "widths", "plans"):
        got = a.get(sec)
        by[sec] = {"n": len(KEY[sec]), "failed": 0}
        if not isinstance(got, dict):
            failed.append("%s:missing_section" % sec)
            by[sec]["failed"] = len(KEY[sec])
            continue
        for qid, want in sorted(KEY[sec].items()):
            g = got.get(qid)
            bad = None
            if sec == "contrasts":
                if not isinstance(g, dict) or not num(g.get("lo")) or not num(g.get("hi")):
                    bad = "shape"
                elif abs(g["lo"] - want["lo"]) > tol or abs(g["hi"] - want["hi"]) > tol:
                    bad = "value"
            elif sec == "values":
                if not isinstance(g, str):
                    bad = "shape"
                elif g.strip().lower() != want:
                    bad = "value"
            elif sec == "widths":
                if not num(g):
                    bad = "shape"
                elif abs(g - want) > tol:
                    bad = "value"
            else:
                if not isinstance(g, dict) or not isinstance(g.get("k"), int) \
                        or not isinstance(g.get("runs"), list):
                    bad = "shape"
                elif g["k"] != want["k"]:
                    bad = "k"
                elif len(g["runs"]) != want["k"] or len(set(g["runs"])) != want["k"]:
                    bad = "set_size"
                elif sorted(str(x) for x in g["runs"]) not in want["sets"]:
                    bad = "set"
            if bad:
                failed.append("%s:%s:%s" % (sec, qid, bad))
                by[sec]["failed"] += 1
    return H.reward(not failed, {"failed": failed[:40], "n_failed": len(failed), "by_section": by})


if __name__ == "__main__":
    sys.exit(main())
