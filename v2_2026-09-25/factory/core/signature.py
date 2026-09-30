"""Heterogeneity gate: declared family signatures, plus a derived audit that checks the declaration.

A family is described by eight components: what the agent hands in, where truth comes from, which failure
modes the grader can separate, whether the world is noisy, whether a resource is metered, whether
abstention is a legal answer, what the agent has to interact with, and whether it may probe a black box.
Two families belong in the same benchmark only if their signatures differ in at least five of the eight.

Declared metadata is exactly the kind of thing an author gets wrong in his own favour, so five of the
eight are also *derived* from the shipped artefacts - the verifier's source, the answer key's vocabulary,
and the contents of `environment/` - and the gate fails if a declaration disagrees with what the task
actually does.  The derived fingerprint is reported per *task*, which also makes visible the thing the
declared table hides: several tasks are controlled arms of one family and are deliberately near-identical
on these axes.  Heterogeneity is a claim about families, never about task count.

Usage: `python3 signature.py [out.json] [tasks_dir]`.
"""
import glob
import itertools
import json
import os
import re
import sys

AXES = ["deliverable", "truth_source", "failure_vectors", "noisy_world", "metered_budget",
        "abstention_is_an_answer", "interaction", "probing"]

SIG = {
    "O": {"deliverable": "executable code (a module the grader imports and runs)",
          "truth_source": "execution of an independent specification implementation",
          "failure_vectors": "semantics|efficiency",
          "noisy_world": False, "metered_budget": "wall clock, calibrated",
          "abstention_is_an_answer": False, "interaction": "read a program, write a program",
          "probing": "unlimited - the reference implementation is public and runnable, so correctness is "
                     "never in doubt; only cost is"},
    "C": {"deliverable": "structured verdicts with numbers and a written audit",
          "truth_source": "analytic, from the generative model of the world",
          "failure_vectors": "identifiability|estimation|confounding",
          "noisy_world": True, "metered_budget": "none",
          "abstention_is_an_answer": True, "interaction": "read a fixed observational log",
          "probing": "none - no new experiment can be run"},
    "V": {"deliverable": "constructed counterexamples (inputs that break a claim)",
          "truth_source": "self-evidencing - the witness is re-checked from first principles",
          "failure_vectors": "false-negative|false-positive|out-of-scope",
          "noisy_world": False, "metered_budget": "none",
          "abstention_is_an_answer": False,
          "interaction": "read a code base, run it freely, hand back inputs",
          "probing": "unlimited - the modules are executable and copyable"},
    "S": {"deliverable": "point estimates plus a typed abstention code per item",
          "truth_source": "analytic, from the generative model of the world",
          "failure_vectors": "selection-on-outcome|over-abstention|nuisance-mechanism",
          "noisy_world": True, "metered_budget": "none",
          "abstention_is_an_answer": True, "interaction": "read a designed sweep and its launcher rules",
          "probing": "none - no new experiment can be run"},
    "B": {"deliverable": "sharp identified intervals plus ternary decisions at a stated margin",
          "truth_source": "exact arithmetic over the published records, cross-checked by a second "
                          "independent implementation",
          "failure_vectors": "non-sharpness|provenance-join|population-definition|decision-rule",
          "noisy_world": False, "metered_budget": "none",
          "abstention_is_an_answer": True,          # `cannot_tell` is a graded verdict, not a refusal
          "interaction": "reconcile several partly contradictory logs of one broken archive",
          "probing": "documentary - the mechanism must be recovered from incident notes and policy "
                     "versions, not from data"},
    "D": {"deliverable": "a purchase decision: guaranteed widths, marginal-value verdicts, and the "
                         "smallest set of measurements that reaches a stated target",
          "truth_source": "exact arithmetic over the records, hardened by an adversarial sweep over every "
                          "rival reading of the measurement service",
          "failure_vectors": "measurement-semantics|worst-case-over-outcomes|marginal-value|minimal-set",
          "noisy_world": False, "metered_budget": "none",
          "abstention_is_an_answer": False,
          "interaction": "choose what to measure next in an archive, and commit before seeing it",
          "probing": "counterfactual - the agent must predict what a measurement would return without "
                     "being allowed to take it"},
}

SHIP = ["O", "V", "S", "B", "D"]
RETIRED = {"C": "superseded by S: only 3 of 8 axes differ, and C is solved by frontier agents"}

# Which family each shipped task belongs to.  Arms of one family share a signature by construction.
FAMILY_OF = {"o-pack-a": "O", "v-falsify-a": "V", "s-censored-a": "S", "c-audit-a": "C",
             "b-bounds-a": "B", "b-bounds-sealed": "B",
             "d-design-a": "D", "d-design-sealed": "D", "d-design-open": "D"}

ABSTAIN = re.compile(r"abstain|unidentif|underdetermin|insufficient|cannot_tell|never_launched|"
                     r"no_surviving|unknown", re.I)
CLOCK = re.compile(r"perf_counter|time\.time|process_time|monotonic")


VOCAB_CONST = re.compile(r"^[A-Z][A-Z_0-9]*\s*=\s*[\{\(\[]([^\}\)\]]*)[\}\)\]]", re.M)


def _label_sets_in(src):
    """String literals that the verifier declares as an accepted vocabulary (ALL-CAPS constant sets)."""
    out = set()
    for body in VOCAB_CONST.findall(src):
        out |= set(re.findall(r'"([a-z][a-z_]{3,})"', body))
    return out


def _leaf_strings(o, out):
    if isinstance(o, dict):
        for v in o.values():
            _leaf_strings(v, out)
    elif isinstance(o, list):
        for v in o:
            _leaf_strings(v, out)
    elif isinstance(o, str):
        out.add(o)
    return out


def derive(task):
    """Read the five mechanically checkable axes off a task directory."""
    tests = os.path.join(task, "tests")
    ver = sorted(glob.glob(tests + "/verify_*.py"))
    src = "".join(open(p).read() for p in ver)
    env = os.path.join(task, "environment")
    envpy = [p for p in glob.glob(env + "/**/*.py", recursive=True)]
    keyp = os.path.join(tests, "key.json")
    key = json.load(open(keyp)) if os.path.exists(keyp) else None
    alt = [json.load(open(p)) for p in glob.glob(tests + "/truth.json")]

    if "answers.json" in src:
        kind = "records"                      # the agent hands in numbers and labels
    elif glob.glob(tests + "/*.npz"):
        kind = "code"                         # the agent hands in a module, graded against saved outputs
    else:
        kind = "inputs"                       # the agent hands in data that the app's own code re-runs on
    # the graded vocabulary: labels in the key, in any separate truth file, and any accepted-label set the
    # verifier declares as a constant.  A task whose labels live only in code is still audited.
    voc = set()
    for obj in ([key] if key is not None else []) + alt:
        _leaf_strings(obj, voc)
    voc |= _label_sets_in(src)
    vocab = sorted(voc)
    return {
        "family": FAMILY_OF.get(os.path.basename(task)),
        "deliverable_kind": kind,
        "answer_sections": sorted(k for k in key if k != "tol") if isinstance(key, dict) else [],
        "vocabulary": [v for v in vocab if len(v) < 40 and not re.fullmatch(r"r\d+", v)][:12],
        "abstention_in_vocabulary": any(ABSTAIN.search(v) for v in vocab),
        "metered_budget": "wall clock" if CLOCK.search(src) else "none",
        "executable_probing": bool(envpy),
        "n_executable_modules_in_environment": len(envpy),
    }


def distance(a, b):
    return [k for k in AXES if SIG[a][k] != SIG[b][k]]


def main():
    tasks_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/bench2/tasks"
    rows, ok = [], True
    for a, b in itertools.combinations(sorted(SIG), 2):
        d = distance(a, b)
        shipped = a in SHIP and b in SHIP
        rows.append({"pair": a + "-" + b, "n_differing_axes": len(d), "axes": d, "shipped_pair": shipped})
        if shipped:
            ok &= len(d) >= 5

    # --- derived audit: the declaration has to match what the shipped task does ------------------------
    derived, mismatch = {}, []
    for t in sorted(glob.glob(tasks_dir + "/*/")):
        name = os.path.basename(t.rstrip("/"))
        if name not in FAMILY_OF:
            continue
        dv = derive(t.rstrip("/"))
        derived[name] = dv
        fam = dv["family"]
        if fam not in SIG:
            continue
        want_ab = SIG[fam]["abstention_is_an_answer"]
        if dv["abstention_in_vocabulary"] != want_ab:
            mismatch.append("%s: declares abstention=%s, key vocabulary says %s"
                            % (name, want_ab, dv["abstention_in_vocabulary"]))
        if not SIG[fam]["metered_budget"].startswith(dv["metered_budget"].split()[0]):
            mismatch.append("%s: declares metered=%r, verifier clock use says %r"
                            % (name, SIG[fam]["metered_budget"], dv["metered_budget"]))
        # only the executable half of `probing` is observable from files; documentary vs counterfactual
        # probing is a prose distinction this audit cannot see and does not pretend to check.
        if SIG[fam]["probing"].startswith("unlimited") != dv["executable_probing"]:
            mismatch.append("%s: declares probing=%r, environment ships %d runnable modules"
                            % (name, SIG[fam]["probing"], dv["n_executable_modules_in_environment"]))
    # families must also be separable on the derived fingerprint alone, not only on prose
    fp = {}
    for name, dv in derived.items():
        if dv["family"] in SHIP:
            fp.setdefault(dv["family"], set()).add((dv["deliverable_kind"], dv["abstention_in_vocabulary"],
                                                   dv["metered_budget"], dv["executable_probing"],
                                                   tuple(dv["answer_sections"])))
    collisions = [(a, b) for a, b in itertools.combinations(sorted(fp), 2) if fp[a] & fp[b]]
    ok = ok and not mismatch and not collisions

    out = {"axes": AXES, "signatures": SIG, "shipped": SHIP, "retired": RETIRED, "pairs": rows,
           "min_required": 5, "family_of_task": FAMILY_OF, "derived_from_artefacts": derived,
           "declaration_mismatches": mismatch,
           "derived_fingerprint_collisions": ["%s~%s" % c for c in collisions],
           "all_shipped_pairs_pass": ok}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if len(sys.argv) > 1:
        os.makedirs(os.path.dirname(os.path.abspath(sys.argv[1])), exist_ok=True)
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
