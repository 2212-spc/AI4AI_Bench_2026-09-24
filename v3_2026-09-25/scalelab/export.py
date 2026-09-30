"""Export a built instance as a Harbor-style task directory.

<task>/
  instruction.md                 what the agent is asked (questions inline)
  task.toml                      metadata (cards, obstacles, claim, timeouts)
  environment/app/               everything the agent can read (bound to /app in the sandbox)
      manual.md  questions.json  answers.json (template)  notebook/runs.csv  notebook/lab_notes.md  bin/lab
  environment/lab_service/       sidecar entrypoint note (the lab server holds hidden/world.json)
  hidden/world.json              world parameters, lab spec, noise salt  (server side only)
  tests/key.json  tests/grade.py grader (keys, tolerances)
  solution/oracle_answers.json   reference solver's answers on this instance's own noise salt
  solution/oracle_runs.jsonl     the reference solver's lab runs (proves solvability within budget)
  hints/H1.md hints/H2.md        hint ladder (appended to the instruction only for hint runs)
  cert.json                      calibration, kill matrix, gates
  review_card.md                 human review card (review_status: pending)
"""
import json, os, shutil, math
from . import queries as Q
from .manual import MANUAL_CORE, LAB_CLI_HELP, hint_text
from .common import fmt

HERE = os.path.dirname(os.path.abspath(__file__))


def _fmt_knob(k, kd):
    if kd["type"] == "float":
        return "- `%s`: number in [%s, %s]%s" % (k, fmt(kd["min"]), fmt(kd["max"]), (" (default %s)" % kd["default"]) if "default" in kd else "")
    return "- `%s`: one of %s%s" % (k, ", ".join(map(str, kd["values"])), (" (default %s)" % kd["default"]) if "default" in kd else "")


def task_manual(w):
    bp, sp = w["bp"], w["spec"]
    s = [MANUAL_CORE, LAB_CLI_HELP, "## 5. This lab", "", "Settable knobs:"]
    s += [_fmt_knob(k, kd) for k, kd in sp["knobs"].items()]
    if sp.get("fixed"):
        s += ["", "Fixed settings (cannot be changed here): " + ", ".join("`%s`=%s" % (k, fmt(v) if not isinstance(v, str) else v) for k, v in sp["fixed"].items())]
    c = sp["caps"]
    s += ["", "Compute: at most %s FLOPs per run, %s FLOPs in total, at most %d runs." % (fmt(c["run_flops"], 3), fmt(c["total_flops"], 3), c["max_runs"]),
          "Metrics reported by `lab run`: " + ", ".join(sp.get("metrics", ["loss"])) + "."]
    for line in getattr(bp, "LAB_EXTRA", lambda w: [])(w):
        s.append(line)
    kus = bp.known_unknowns(w["p"])
    s += ["", "### Known unknowns", ""]
    if kus:
        for ku in kus:
            s.append("- **%s.** %s" % (ku["name"], ku["text"]))
    else:
        s.append("- None: everything the questions ask about is determined by the lab plus Section 2.")
    return "\n".join(s) + "\n"


def instruction(w, items, hint_level=0):
    bp = w["bp"]
    qs = [Q.public_view(it) for it in items]
    s = ["# Task: %s" % bp.TITLE, "",
         "You are a research engineer in a pretraining team.  The team's notebook (`/app/notebook/`) holds "
         "earlier runs and the team's notes.  You have a compute budget on the team's simulated lab "
         "(`lab` command; read `/app/manual.md` first - it defines the lab, its guarantees and how answers are graded).",
         "", "Answer every question below by writing `/app/answers.json` (a template with the right keys is "
         "already there).  You may run any experiments within the budget.  Work carefully: the notes may be "
         "wrong, and some questions may not be fully answerable from anything the lab can measure - the manual "
         "says how to answer those.", "", "## Questions", ""]
    for q in qs:
        fmt_s = json.dumps(q["answer_format"])
        s.append("- **%s** %s  %s%s" % (q["id"], q["question"], ("[unit: %s] " % q["unit"]) if "unit" in q else "", "Answer format: `%s`" % fmt_s))
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
    csv, extra = runs_csv(w)
    return "\n".join([task_manual(w), instruction(w, items), w["notes"], csv, json.dumps(extra),
                      json.dumps([Q.public_view(it) for it in items])])


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


def review_card(inst, task_id):
    w, cal, g = inst["w"], inst["cal"], inst["gates"]
    bp = w["bp"]
    L = ["---", "task: %s" % task_id, "blueprint: %s" % bp.ID, "world_seed: %s" % w["ws"], "review_status: pending",
         "reviewer: ", "---", "", "# Review card: %s" % bp.TITLE, "",
         "**Claim tested.** %s" % bp.CLAIM, "",
         "**Cards.** " + ", ".join("%s (%s)" % (c, __import__("scalelab.cards", fromlist=["CARDS"]).CARDS[c]["name"]) for c in bp.CARDS),
         "", "**Obstacles.** " + ", ".join(bp.OBSTACLES), "", "**Design note.** " + (bp.__doc__ or "").strip(), "",
         "## Items, keys, tolerances", "", "| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |",
         "|---|---|---|---|---|---|"]
    for it in cal["items"]:
        k = it["key"]
        ks = ("[%.4g, %.4g]" % (k["lo"], k["hi"])) if it["kind"] in ("point", "set") else (k.get("verdict") or k.get("choice"))
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
          "", "Reviewer notes:", ""]
    return "\n".join(L) + "\n"


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
    open(os.path.join(app, "manual.md"), "w").write(task_manual(w))
    open(os.path.join(app, "questions.json"), "w").write(json.dumps([Q.public_view(it) for it in items], indent=1))
    open(os.path.join(app, "answers.json"), "w").write(json.dumps(answers_template(items), indent=1))
    csv, extra = runs_csv(w)
    open(os.path.join(app, "notebook", "runs.csv"), "w").write(csv)
    if extra:
        open(os.path.join(app, "notebook", "checkpoints.json"), "w").write(json.dumps(_jsonable(
            [{"run": w["rows"].index(r) + 1, "checkpoints": r.get("checkpoints"), "cooldown_branches": r.get("cooldown_branches")} for r in extra]), indent=1))
    open(os.path.join(app, "notebook", "lab_notes.md"), "w").write(w["notes"])
    shutil.copy(os.path.join(HERE, "lab_cli.py"), os.path.join(app, "bin", "lab"))
    os.chmod(os.path.join(app, "bin", "lab"), 0o755)
    open(os.path.join(d, "instruction.md"), "w").write(instruction(w, items, 0))
    for h in (1, 2):
        open(os.path.join(d, "hints", "H%d.md" % h), "w").write(hint_text(h, w["bp"].CARDS))
    open(os.path.join(d, "hidden", "world.json"), "w").write(json.dumps(_jsonable(
        {"params": w["p"], "spec": w["spec"], "salt": w["salt"], "blueprint": w["bp"].ID, "world_seed": w["ws"]}), indent=1))
    open(os.path.join(d, "tests", "key.json"), "w").write(json.dumps(_jsonable(items), indent=1))
    shutil.copy(os.path.join(HERE, "queries.py"), os.path.join(d, "tests", "queries.py"))
    open(os.path.join(d, "tests", "grade.py"), "w").write(GRADE_PY)
    open(os.path.join(d, "solution", "oracle_answers.json"), "w").write(json.dumps(_jsonable(cal["ship_ans"]), indent=1))
    with open(os.path.join(d, "solution", "oracle_runs.jsonl"), "w") as f:
        for r in cal["ship_log"]:
            f.write(json.dumps(_jsonable(r)) + "\n")
    cert = {"gates": inst["gates"], "ok": inst["ok"], "kill_matrix": inst["mat"],
            "calibration": {"tol": cal["tol"], "cal_errs": cal["cal_errs"], "ver_pass": cal["ver_pass"],
                            "ver_allpass": cal["ver_allpass"], "oracle_usage": cal["ship_usage"]}}
    open(os.path.join(d, "cert.json"), "w").write(json.dumps(_jsonable(cert), indent=1))
    open(os.path.join(d, "review_card.md"), "w").write(review_card(inst, task_id))
    bp = w["bp"]
    open(os.path.join(d, "task.toml"), "w").write(
        '[task]\nid = "%s"\nblueprint = "%s"\nworld_seed = %s\ntitle = "%s"\ncards = %s\nobstacles = %s\nclaim = "%s"\n'
        'n_items = %d\n\n[agent]\ntimeout_sec = 7200\n\n[environment]\napp_dir = "environment/app"\n'
        'lab_service = "scalelab.lab_server (holds hidden/world.json; the agent reaches it only through bin/lab)"\n\n'
        '[verifier]\ncommand = "python3 tests/grade.py <answers.json>"\n' % (
            task_id, bp.ID, w["ws"], bp.TITLE, json.dumps(bp.CARDS), json.dumps(bp.OBSTACLES), bp.CLAIM.replace('"', "'"), len(items)))
    open(os.path.join(d, "environment", "lab_service", "README.md"), "w").write(
        "The lab service is `scalelab/lab_server.py`.  It loads `hidden/world.json`, keeps the budget ledger outside the "
        "agent's filesystem, and executes every request through `scalelab.lab.Session` - the same code the builder's "
        "oracle, rivals and gates used.\n")
    return d


GRADE_PY = r'''"""Grade an answers.json against this task's key.  usage: python3 grade.py answers.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queries import grade
items = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.json")))
try:
    ans = json.load(open(sys.argv[1]))
except Exception as e:
    ans = {}
    print("unreadable answers:", e)
print(json.dumps(grade(items, ans), indent=1))
'''
