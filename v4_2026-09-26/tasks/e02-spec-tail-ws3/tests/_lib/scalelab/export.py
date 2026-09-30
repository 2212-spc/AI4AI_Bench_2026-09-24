"""Export a built instance as a Harbor-style task directory.

<task>/
  instruction.md                 what the agent is asked (questions inline)
  task.toml                      metadata (cards, obstacles, claim, timeouts)
  environment/app/               everything the agent can read (bound to /app in the sandbox)
      manual.md  questions.json  answers.json (template)  notebook/runs.csv  notebook/lab_notes.md  bin/lab
      plus any files the blueprint ships (an audit task's analysis/ script, logs and report)
  environment/lab_service/       sidecar entrypoint note (the lab server holds hidden/world.json)
  hidden/world.json              world parameters, lab spec, noise salt  (server side only)
  tests/key.json                 keys, tolerances, witness schemas, prereg world sets
  tests/rows.json                the disclosed notebook rows (a witness must reproduce them)
  tests/_lib/scalelab/           frozen copy of the grading code
  tests/grade.py                 grader (builds the grading context and runs it)
  solution/oracle_answers.json   reference solver's answers on this instance's own noise salt
  solution/oracle_runs.jsonl     the reference solver's lab runs (proves solvability within budget)
  hints/H1.md hints/H2.md        hint ladder (appended to the instruction only for hint runs)
  cert.json                      calibration, kill matrix, gates
  review_card.md                 human review card (review_status: pending)

The three v4 forms are graded by *execution*, not by comparison with a stored answer, so tests/ carries
the code and the rows they need rather than a number.
"""
import json, os, shutil, math
from . import queries as Q
from . import labs
from .lab import caps_of
from .manual import manual_core, manual_tail, hint_text
from .common import fmt

HERE = os.path.dirname(os.path.abspath(__file__))

ROLE = {"pretrain": ("a pretraining team", "simulated lab"),
        "evallab": ("a model-evaluation team", "simulated evaluation service"),
        "rllab": ("a post-training team", "simulated post-training service"),
        "servelab": ("an inference-serving team", "simulated serving stack")}


def _fmt_knob(k, kd):
    if kd["type"] == "float":
        return "- `%s`: number in [%s, %s]%s" % (k, fmt(kd["min"]), fmt(kd["max"]), (" (default %s)" % kd["default"]) if "default" in kd else "")
    return "- `%s`: one of %s%s" % (k, ", ".join(map(str, kd["values"])), (" (default %s)" % kd["default"]) if "default" in kd else "")


def budget_line(sp):
    """One line in this lab's own cost unit; the unit and the charging rule come from the backend."""
    c = caps_of(sp); be = labs.backend(sp.get("lab", "pretrain"))
    return "Budget: at most %s %s per request, %s %s in total, at most %d requests.  Charging: %s." % (
        fmt(c["run_cost"], 3), be.COST_UNIT, fmt(c["total_cost"], 3), be.COST_UNIT, c["max_runs"], be.COST_TEXT)


def task_manual(w, items=None):
    bp, sp = w["bp"], w["spec"]
    lab = sp.get("lab", "pretrain")
    kinds = sorted({it["kind"] for it in items}) if items else None
    s = [manual_core(lab, kinds), "## 5. This lab", "", "Settable knobs:"]
    s += [_fmt_knob(k, kd) for k, kd in sp["knobs"].items()]
    if sp.get("fixed"):
        s += ["", "Fixed settings (cannot be changed here): " + ", ".join("`%s`=%s" % (k, fmt(v) if not isinstance(v, str) else v) for k, v in sp["fixed"].items())]
    s += ["", budget_line(sp)]
    if sp.get("metrics"):
        s += ["Reported by `lab run`: " + ", ".join(sp["metrics"]) + "."]
    for line in getattr(bp, "LAB_EXTRA", lambda w: [])(w):
        s.append(line)
    kus = bp.known_unknowns(w["p"])
    s += ["", "### Known unknowns", ""]
    if kus:
        for ku in kus:
            s.append("- **%s.** %s" % (ku["name"], ku["text"]))
    else:
        s.append("- None: everything the questions ask about is determined by the lab plus Section 2.")
    tail = manual_tail(kinds)
    if tail:
        s += ["", tail]
    return "\n".join(s) + "\n"


def _question_md(q):
    L = ["- **%s** %s  %s%s" % (q["id"], q["question"], ("[unit: %s] " % q["unit"]) if "unit" in q else "",
                               "Answer format: `%s`" % json.dumps(q["answer_format"]))]
    if "claim" in q:
        L.append("    - Claim under test: *%s*" % q["claim"])
    if "limits" in q:
        L.append("    - Plan limits: at most %d runs, %s in total; allowed conclusions: %s."
                 % (q["limits"]["max_runs"], fmt(q["limits"]["budget"], 3),
                    ", ".join("`%s`" % s for s in q["limits"]["allowed_labels"])))
    return "\n".join(L)


def instruction(w, items, hint_level=0):
    bp = w["bp"]
    lab = w["spec"].get("lab", "pretrain")
    team, service = ROLE.get(lab, ROLE["pretrain"])
    qs = [Q.public_view(it) for it in items]
    kinds = {it["kind"] for it in items}
    s = ["# Task: %s" % bp.TITLE, "",
         "You are a research engineer in %s.  The team's notebook (`/app/notebook/`) holds earlier runs and "
         "the team's notes.  You have a budget on the team's %s (`lab` command; read `/app/manual.md` "
         "first - it defines the service, its guarantees and how answers are graded)." % (team, service)]
    if "audit" in kinds:
        s += ["", "`/app/analysis/` holds a teammate's analysis script.  It reads the logs in "
              "`/app/notebook/`, and the number it printed is the one quoted in the team's notes."]
    s += ["", "Answer every question below by writing `/app/answers.json` (a template with the right keys is "
          "already there).  You may run any experiments within the budget.  Work carefully: the notes may be "
          "wrong, and some questions may not be fully answerable from anything the lab can measure - the manual "
          "says how to answer those.", "", "## Questions", ""]
    s += [_question_md(q) for q in qs]
    s += ["", "When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id."]
    ht = hint_text(hint_level, bp.CARDS)
    if ht:
        s += ["", ht]
    return "\n".join(s) + "\n"


def answers_template(items):
    out = {}
    for it in items:
        if it["kind"] in ("point", "set"):
            out[it["id"]] = {"lo": None, "hi": None}
        elif it["kind"] == "decision":
            out[it["id"]] = {"choice": None}
        elif it["kind"] == "cex":
            out[it["id"]] = {"verdict": None, "witness": {k: None for k in it["schema"]["params"]}}
        elif it["kind"] == "prereg":
            out[it["id"]] = {"runs": [], "rule": {"expr": "", "cuts": []}}
        elif it["kind"] == "audit":
            out[it["id"]] = {"defect": None, "site": None, "corrected": {"lo": None, "hi": None}}
        else:
            out[it["id"]] = {"verdict": None}
    return out


def runs_csv(w):
    rows = w["rows"]
    cfg_keys = []
    for r in rows:
        for k in r["config"]:
            if k not in cfg_keys and k not in ("ckpts", "cooldowns"):
                cfg_keys.append(k)
    met = []
    for r in rows:
        for k, v in r.items():
            if k not in ("config", "status", "note", "checkpoints", "cooldown_branches") and k not in met and not isinstance(v, (dict, list)):
                met.append(k)
    lines = [",".join(["run"] + cfg_keys + ["status"] + met)]
    for i, r in enumerate(rows, 1):
        vals = [str(i)] + [fmt(r["config"].get(k), 6) if not isinstance(r["config"].get(k), str) else r["config"][k] for k in cfg_keys]
        vals.append(r["status"])
        vals += ["" if r.get(k) is None else (fmt(r[k], 6) if not isinstance(r[k], str) else r[k]) for k in met]
        lines.append(",".join(vals))
    txt = "\n".join(lines) + "\n"
    extra = [r for r in rows if r.get("checkpoints") or r.get("cooldown_branches")]
    return txt, extra


def shipped_text(w, items):
    """Everything the agent can read, concatenated - the leakage scanner's input."""
    csv, extra = runs_csv(w)
    parts = [task_manual(w, items), instruction(w, items), w["notes"], csv, json.dumps(extra),
             json.dumps([Q.public_view(it) for it in items])]
    parts += list((w.get("files") or {}).values())
    return "\n".join(parts)


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


def review_card(inst, task_id):
    w, cal, g = inst["w"], inst["cal"], inst["gates"]
    bp = w["bp"]
    L = ["---", "task: %s" % task_id, "blueprint: %s" % bp.ID, "world_seed: %s" % w["ws"], "review_status: pending",
         "reviewer: ", "---", "", "# Review card: %s" % bp.TITLE, "",
         "**Claim tested.** %s" % bp.CLAIM, "",
         "**Lab.** %s" % w["spec"].get("lab", "pretrain"), "",
         "**Cards.** " + ", ".join("%s (%s)" % (c, __import__("scalelab.cards", fromlist=["CARDS"]).CARDS[c]["name"]) for c in bp.CARDS),
         "", "**Obstacles.** " + ", ".join(bp.OBSTACLES), "", "**Design note.** " + (bp.__doc__ or "").strip(), "",
         "## Items, keys, tolerances", "", "| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |",
         "|---|---|---|---|---|---|"]
    for it in cal["items"]:
        k = it["key"]
        ks = ("[%.4g, %.4g]" % (k["lo"], k["hi"])) if it["kind"] in ("point", "set") else json.dumps(k)
        tol = ("%.3g" % it["tol"]) if "tol" in it else ("regret<=%.3g" % it.get("r_tol", 0))
        L.append("| %s | %s | %s | %s | %.2f | %s |" % (it["id"], it["kind"], ks, tol, cal["ver_pass"][it["id"]],
                                                     ", ".join(g["G10_item_useful"]["item_kills"][it["id"]]) or "-"))
    L += ["", "## Rivals", "", "| rival | score | killed with margin on | gated |", "|---|---|---|---|"]
    for name, v in inst["mat"].items():
        L.append("| %s | %.2f | %s | %s |" % (name, v["score"], ", ".join(v["margin_items"]) or "-", "yes" if v["must_kill"] else "info"))
    L += ["", "## Gates", ""]
    for gn, gv in g.items():
        L.append("- %s: **%s**" % (gn, "pass" if gv["pass"] else "FAIL"))
    L += ["", "## Human checklist (tick each; any 'no' -> reject or fix the blueprint)", "",
          "- [ ] The notebook and lab notes read like a plausible team artefact; nothing in them states the answer.",
          "- [ ] Each question is unambiguous given manual.md Section 3 (point vs. set vs. verdict semantics).",
          "- [ ] Each known unknown is truly undeterminable in this lab (see cert.json G4) and its documented range is the one used for the key.",
          "- [ ] The keys follow from the cards' stated forms (spot-check one numeric key by hand from hidden/world.json).",
          "- [ ] The oracle's runs (solution/oracle_runs.jsonl) are a design a competent researcher could think of.",
          "- [ ] The rival that encodes the notes' reading fails for the reason the design note says.",
          "- [ ] Grounding references for each active card are appropriate (cards.py).",
          "- [ ] For a counterexample item: the free parameters are the only ones that could flip the claim, and the oracle's witness is not the only one that works.",
          "- [ ] For a plan item: every world in the world set is genuinely consistent with the notebook, and its label is the conclusion a careful analyst would reach.",
          "- [ ] For an audit item: the script reads as a teammate's work, the defect is silent (it runs and prints a plausible number), and every red herring is provably harmless.",
          "", "Reviewer notes:", ""]
    return "\n".join(L) + "\n"


def _copy_lib(dst):
    """Freeze the grading code next to the key, so a task can be graded years later by its own copy."""
    shutil.copytree(HERE, os.path.join(dst, "scalelab"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def export(inst, out_root, task_id):
    w, cal = inst["w"], inst["cal"]
    items = cal["items"]
    d = os.path.join(out_root, task_id)
    if os.path.exists(d):
        shutil.rmtree(d)
    app = os.path.join(d, "environment", "app")
    os.makedirs(os.path.join(app, "notebook")); os.makedirs(os.path.join(app, "bin"))
    for sub in ("hidden", "tests", "solution", "hints", "environment/lab_service"):
        os.makedirs(os.path.join(d, sub))
    open(os.path.join(app, "manual.md"), "w").write(task_manual(w, items))
    open(os.path.join(app, "questions.json"), "w").write(json.dumps([Q.public_view(it) for it in items], indent=1))
    open(os.path.join(app, "answers.json"), "w").write(json.dumps(answers_template(items), indent=1))
    csv, extra = runs_csv(w)
    open(os.path.join(app, "notebook", "runs.csv"), "w").write(csv)
    if extra:
        open(os.path.join(app, "notebook", "checkpoints.json"), "w").write(json.dumps(_jsonable(
            [{"run": w["rows"].index(r) + 1, "checkpoints": r.get("checkpoints"), "cooldown_branches": r.get("cooldown_branches")} for r in extra]), indent=1))
    open(os.path.join(app, "notebook", "lab_notes.md"), "w").write(w["notes"])
    for rel, text in (w.get("files") or {}).items():
        p = os.path.join(app, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    shutil.copy(os.path.join(HERE, "lab_cli.py"), os.path.join(app, "bin", "lab"))
    os.chmod(os.path.join(app, "bin", "lab"), 0o755)
    open(os.path.join(d, "instruction.md"), "w").write(instruction(w, items, 0))
    for h in (1, 2):
        open(os.path.join(d, "hints", "H%d.md" % h), "w").write(hint_text(h, w["bp"].CARDS))
    open(os.path.join(d, "hidden", "world.json"), "w").write(json.dumps(_jsonable(
        {"params": w["p"], "spec": w["spec"], "salt": w["salt"], "blueprint": w["bp"].ID,
         "module": w["bp"].__name__.rsplit(".", 1)[-1], "world_seed": w["ws"]}), indent=1))
    open(os.path.join(d, "tests", "key.json"), "w").write(json.dumps(_jsonable(items), indent=1))
    open(os.path.join(d, "tests", "rows.json"), "w").write(json.dumps(_jsonable(w["rows"]), indent=1))
    _copy_lib(os.path.join(d, "tests", "_lib"))
    open(os.path.join(d, "tests", "grade.py"), "w").write(GRADE_PY)
    open(os.path.join(d, "solution", "oracle_answers.json"), "w").write(json.dumps(_jsonable(cal["ship_ans"]), indent=1))
    with open(os.path.join(d, "solution", "oracle_runs.jsonl"), "w") as f:
        for r in cal["ship_log"]:
            f.write(json.dumps(_jsonable(r)) + "\n")
    cert = {"gates": inst["gates"], "ok": inst["ok"], "kill_matrix": inst["mat"],
            "calibration": {"tol": cal["tol"], "cal_errs": cal["cal_errs"], "ver_pass": cal["ver_pass"],
                            "ver_allpass": cal["ver_allpass"], "oracle_usage": cal["ship_usage"]}}
    if "difficulty" in inst:
        open(os.path.join(d, "difficulty.json"), "w").write(json.dumps(_jsonable(inst["difficulty"]), indent=1))
    open(os.path.join(d, "cert.json"), "w").write(json.dumps(_jsonable(cert), indent=1))
    open(os.path.join(d, "review_card.md"), "w").write(review_card(inst, task_id))
    bp = w["bp"]
    open(os.path.join(d, "task.toml"), "w").write(
        '[task]\nid = "%s"\nblueprint = "%s"\nlab = "%s"\nworld_seed = %s\ntitle = "%s"\ncards = %s\nobstacles = %s\nclaim = "%s"\n'
        'n_items = %d\nkinds = %s\n\n[agent]\ntimeout_sec = 7200\n\n[environment]\napp_dir = "environment/app"\n'
        'lab_service = "scalelab.lab_server (holds hidden/world.json; the agent reaches it only through bin/lab)"\n\n'
        '[verifier]\ncommand = "python3 tests/grade.py <answers.json> [lab_ledger.jsonl]"\n' % (
            task_id, bp.ID, w["spec"].get("lab", "pretrain"), w["ws"], bp.TITLE, json.dumps(bp.CARDS),
            json.dumps(bp.OBSTACLES), bp.CLAIM.replace('"', "'"), len(items),
            json.dumps(sorted({it["kind"] for it in items}))))
    open(os.path.join(d, "environment", "lab_service", "README.md"), "w").write(
        "The lab service is `scalelab/lab_server.py`.  It loads `hidden/world.json`, keeps the budget ledger outside the "
        "agent's filesystem, and executes every request through `scalelab.lab.Session` - the same code the builder's "
        "oracle, rivals and gates used.\n")
    return d


GRADE_PY = r'''"""Grade an answers.json against this task's key.

    usage: python3 grade.py <answers.json> [lab_ledger.jsonl]

The three v4 forms are graded by execution rather than by comparison with a stored answer: a
counterexample witness is rebuilt into a world and checked against the evidence, and a pre-registered
plan is executed in every world the notebook leaves open.  That needs the lab code, the hidden world and
the disclosed rows, so this grader imports the frozen copy of the package in `_lib/`.

Pass the agent's lab ledger (`<run>/lab_ledger.jsonl`) as the second argument, or set $LAB_LEDGER, so
that a counterexample witness must also reproduce the runs the agent made itself.
"""
import importlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_lib"))
from scalelab import queries as Q
from scalelab import ctx as C

items = json.load(open(os.path.join(HERE, "key.json")))
world = json.load(open(os.path.join(HERE, os.pardir, "hidden", "world.json")))
rows = json.load(open(os.path.join(HERE, "rows.json")))
ledger = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("LAB_LEDGER")

claim_fn = None
if any(it["kind"] == "cex" for it in items):
    bp = importlib.import_module("scalelab.bp." + world["module"])
    claim_fn = getattr(bp, "claim_fn", None)

gctx = C.make_ctx(world["params"], world["spec"], world["salt"], rows, items, claim_fn, ledger)
try:
    ans = json.load(open(sys.argv[1]))
except Exception as e:
    ans = {}
    print("unreadable answers:", e)
print(json.dumps(Q.grade(items, ans, gctx), indent=1))
'''
