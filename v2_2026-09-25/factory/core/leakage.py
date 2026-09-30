"""Leakage gate: the graded answer must not be *readable* anywhere the agent can reach.

Every arm here ships a world the agent explores with a shell, so the cheapest attack is never reasoning -
it is `grep`.  This gate is the mechanical version of that attack.  For each task it checks that:

  1. no file the agent can read names the hidden mechanism (the generators' truth-rule identifiers, the
     rival readings the self-checks sweep, or the authoring filenames);
  2. no key value is *readable* in the environment.  Whether a verbatim hit means anything depends on how
     coarse the published grid is: these worlds publish pass rates on a 0.0001 grid, so any one four-decimal
     answer collides with some surviving run's value about five percent of the time by chance, and the first
     version of this gate duly fired on such a collision.  So a hit counts as a leak when either
       * the value carries more decimals than the environment's own grid - it cannot have arisen by
         coincidence and must have been computed and then written down; or
       * the value sits on the grid *and* the line it appears on also names the run the key entry is about -
         that is a file associating the hidden quantity with its owner, which is what a leak looks like.
     Grid-valued hits on lines about other runs are reported as coincidences and do not fail the gate.
     This replaces an earlier design that exempted known-benign hits by name: a predicate that says *why* a
     hit is uninformative cannot quietly widen to cover a real leak, whereas a list of blessed hits can.
  3. the authoring material - reference solver, certificate, generator-side truth files - lives outside
     `environment/`, so a sandbox that mounts only `environment/app` cannot see it;
  4. the verifier and key are not inside `environment/` either, and the task's own `test.sh` reads them
     from `TESTS`, not from the app.

Usage: `python3 leakage.py [tasks_dir] [out.json]`.  Exit status is non-zero if any task leaks.
"""
import glob
import json
import os
import re
import sys

RULE_WORDS = re.compile(r"the_sink_finalises_the_bundle|the_screen_redacts_it|the_lost_shard_took|"
                        r"TRUTH_RULE|NAIVE_RULE|RECOV_RULES|grain\.json|ref_solve|certificate\.json")
SKIP_BIN = (".npz", ".pyc", ".png", ".gz", ".zip")
RUN_ID = re.compile(r"\br\d{4,6}\b")


def numbers_of(o, paths, path=""):
    """Every float leaf of the key, with the key paths it occurs at."""
    if isinstance(o, dict):
        for k, v in o.items():
            numbers_of(v, paths, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            numbers_of(v, paths, path + "/%d" % i)
    elif isinstance(o, float):
        paths.setdefault(o, []).append(path)
    return paths


def grid_decimals(text):
    """How many decimals the environment itself publishes - the resolution a coincidence can reach."""
    best = 0
    for t in text.values():
        for m in re.finditer(r"\d\.(\d+)", t):
            best = max(best, len(m.group(1)))
    return best


def owners(key_paths, queries):
    """The run ids the key entries at these paths are about, when they are about single runs."""
    out = set()
    if not isinstance(queries, dict):            # c-audit ships a bare list of query objects
        queries = {"": queries}
    for p in key_paths:
        bits = p.strip("/").split("/")
        if len(bits) < 2:
            continue
        sec, qid = bits[0], bits[1]
        for q in [x for x in queries.get(sec, []) if isinstance(x, dict)]:
            if q.get("id") == qid:
                out |= {q[k] for k in ("run_id",) if k in q} | set(q.get("recover", []))
    return out


def audit(task):
    name = os.path.basename(task)
    env = os.path.join(task, "environment")
    files = [p for p in glob.glob(env + "/**/*", recursive=True)
             if os.path.isfile(p) and not p.endswith(SKIP_BIN)]
    text = {p: open(p, errors="replace").read() for p in files}
    grid = grid_decimals(text)
    qp = os.path.join(env, "app", "queries.json")
    queries = json.load(open(qp)) if os.path.exists(qp) else {}

    named = sorted({(os.path.relpath(p, task), m) for p, t in text.items()
                    for m in RULE_WORDS.findall(t)})

    leaks, coincidences = [], []
    for kp in glob.glob(task + "/tests/key.json") + glob.glob(task + "/tests/truth.json"):
        key = json.load(open(kp))
        for v, where in sorted(numbers_of(key, {}).items()):
            s = ("%.6f" % v).rstrip("0")
            dec = len(s.split(".")[-1])
            if abs(v) < 1e-9 or dec < 3:
                continue                         # a value like 0.02 is published everywhere; not diagnostic
            own = owners(where, queries)
            for p, t in text.items():
                for line in (l for l in t.splitlines() if s in l):
                    rec = {"file": os.path.relpath(p, task), "value": s, "key_paths": where[:3],
                           "line": line.strip()[:90]}
                    if dec > grid:
                        rec["why"] = "finer than the environment's %d-decimal grid" % grid
                        leaks.append(rec)
                    elif own & set(RUN_ID.findall(line)):
                        rec["why"] = "on the grid, but this line names the run the key entry is about"
                        leaks.append(rec)
                    else:
                        rec["why"] = "on the grid and about another run - a collision, not a leak"
                        coincidences.append(rec)

    inside = sorted(os.path.relpath(p, task) for p in files
                    if re.search(r"(^|/)(solution|authoring|tests)(/|$)", os.path.relpath(p, task)))
    sh = os.path.join(task, "tests", "test.sh")
    shsrc = open(sh).read() if os.path.exists(sh) else ""
    reads_app = bool(re.search(r"(key|truth)\.json", shsrc)) and "TESTS" not in shsrc
    row = {"task": name, "n_readable_files": len(files), "environment_grid_decimals": grid,
           "mechanism_words_in_environment": named,
           "leaks": leaks[:8], "n_leaks": len(leaks),
           "grid_collisions": coincidences[:4], "n_grid_collisions": len(coincidences),
           "authoring_files_inside_environment": inside,
           "test_sh_reads_key_from_the_app": reads_app}
    row["clean"] = not named and not leaks and not inside and not reads_app
    return row


def main():
    tasks = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks"
    rows = [audit(t.rstrip("/")) for t in sorted(glob.glob(tasks + "/*/"))
            if os.path.isdir(t + "environment")]
    ok = all(r["clean"] for r in rows)
    for r in rows:
        print("%-4s %-18s files=%-3s grid=%d leaks=%-3d collisions=%-3d mech=%-2d authoring_in_env=%d"
              % ("OK" if r["clean"] else "LEAK", r["task"], r["n_readable_files"],
                 r["environment_grid_decimals"], r["n_leaks"], r["n_grid_collisions"],
                 len(r["mechanism_words_in_environment"]), len(r["authoring_files_inside_environment"])))
        for h in r["leaks"]:
            print("      LEAK", h)
        for m in r["mechanism_words_in_environment"]:
            print("      mech", m)
    print("LEAKAGE GATE", "PASS" if ok else "FAIL")
    if len(sys.argv) > 2:
        json.dump({"tasks": rows, "all_clean": ok}, open(sys.argv[2], "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
