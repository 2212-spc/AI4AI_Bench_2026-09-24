"""Grade finished runs (the verifier sees only declared artifacts + the lab ledger).

  python3 -m l15.grade <run_dir> [<run_dir> ...]      -> writes <run_dir>/grade.json, prints one line per run

Integrity: artifacts are read from the run's app/ on the host after the agent stopped; the hidden world lives
in <task_dir>/hidden (never mounted in the sandbox); the ledger is written by the lab service, not the agent."""
import json, os, sys
from .core import load_world


class StaleTask(Exception):
    """The built task and the live grading module disagree - the run is not gradeable."""


def check_consts(task_dir):
    """Refuse to grade a run whose instruction quoted different constants than the grader now enforces.

    See the note in build.py: a re-tuned constant silently re-grades old runs against a bar they were never
    shown, which turns a correct agent run into a failure.  Raising here means a stale task shows up as an
    integrity error in the sweep instead of as a model result."""
    fp = os.path.join(task_dir, "hidden", "grading_consts.json")
    if not os.path.exists(fp):
        raise StaleTask("built before constant-stamping; rebuild the task to grade it")
    import importlib
    cfg = json.load(open(os.path.join(task_dir, "hidden", "world.json")))
    mod = importlib.import_module("l15.tasks." + cfg["task"])
    was = json.load(open(fp))
    drift = {k: (v, getattr(mod, k)) for k, v in was.items()
             if hasattr(mod, k) and getattr(mod, k) != v}
    if drift:
        raise StaleTask("grading constants changed since build: "
                        + "; ".join("%s %r -> %r" % (k, a, b) for k, (a, b) in sorted(drift.items())))


def grade_run(rd):
    meta = json.load(open(os.path.join(rd, "lab.json")))
    check_consts(meta["task_dir"])
    w = load_world(meta["task_dir"])
    led = []
    lp = os.path.join(rd, "lab_ledger.jsonl")
    if os.path.exists(lp):
        led = [json.loads(l) for l in open(lp) if l.strip()]
    g = w.grade(os.path.join(rd, "app"), led)
    st = {}
    if os.path.exists(os.path.join(rd, "state.json")):
        s = json.load(open(os.path.join(rd, "state.json")))
        st = {"status": s.get("status"), "agent_min": round(s.get("agent_sec", 0) / 60, 1),
              "api_errors": s.get("api_errors") or 0}
        # DERIVE `truncated` here rather than trusting whatever an earlier grader happened to write.
        #
        # This field decides whether a run counts as a capability measurement (see explore/analysis.py):
        # a run the harness cut off never chose to stop, so its missing artifact says nothing about
        # whether the model could have produced one.  An earlier grade.py wrote it and this one had
        # stopped; the effect was silent and one-way - re-grading a held-out run promoted it into the
        # headline table with `truncated` absent, i.e. falsy.  Found 2026-09-28 while grading two runs
        # that had never been graded at all.  A flag that only the OLD code wrote is worse than no flag.
        st["truncated"] = s.get("status") in ("timeout", "max_calls", "api_failed")
        # A provider-side block is not a capability result either.  Fable's safeguard classifier refuses
        # the k2 instruction outright - all four k2d fable runs ended on turn 1 with "safeguards flagged
        # this message", zero lab calls, no artifact.  Rewording the task to get past a safety classifier
        # is out of scope, so these are recorded and held out rather than retried.
        _ft = s.get("final_text") or ""
        if "safeguards flagged" in _ft or "safeguards have flagged" in _ft:
            st["truncated"] = True
            st["truncate_reason"] = "provider_blocked"
        # `api_errors` is deliberately NOT a hold-out criterion, though it was one for a few hours on
        # 2026-09-28 and that was wrong.  The counter increments on every RETRIED transient (see
        # harness/gpt_agent.py:161); the harness only abandons a run at >40, with status `api_failed`.
        # So `done` plus a nonzero count means the retries were recovered from, not that the agent lost
        # its turns.  The rule was added on the strength of `k6f2__gpt55__r1` (11 api_errors, 3 lab
        # calls, read as a wasted run) and, once written, held out 9 runs of which three had PASSED -
        # `k2d99__gemflash__r1` with 144 model calls, 130 lab calls and a full scaling-law write-up, and
        # `k1c1__gpt6__r1` and `k4w2__gemflash__r1`.  Re-reading the state files, k6f2__gpt55__r1 made 24
        # model calls and shipped a py_compile-validated sched.py: it is a considered failure, and its
        # 3 lab calls are the finding, not an artifact of the transport.  Holding out runs on a signal
        # that correlates with being SLOW rather than being CUT OFF biases the tally towards whichever
        # models the gateway happened to serve badly - exactly the kind of quiet selection effect the
        # `truncated` field exists to prevent.  Terminal API failure is `status == "api_failed"`, which
        # the status test below already covers.
        elif st["truncated"]:
            st["truncate_reason"] = s.get("status")
    g.update({"run": os.path.basename(rd), "task": meta["task"], "model": meta["model"], "hint": meta["hint"],
              "lab_calls": len(led), "budget_spent_frac": round(sum(r["cost"] for r in led) / w.budget, 3), **st})
    json.dump(g, open(os.path.join(rd, "grade.json"), "w"), indent=1, default=float)
    return g


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        try:
            g = grade_run(rd.rstrip("/"))
        except StaleTask as e:
            print("%-40s STALE %s" % (os.path.basename(rd.rstrip("/")), e))
            continue
        fails = ",".join(k for k, v in g["items"].items() if not v["ok"])
        print("%-40s %-5s score=%s fail=[%s] calls=%d spent=%.2f" % (g["run"], "PASS" if g["pass"] else "fail", g["score"], fails,
                                                                  g["lab_calls"], g["budget_spent_frac"]))
