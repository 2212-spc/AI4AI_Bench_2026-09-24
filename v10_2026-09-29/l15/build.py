"""Package one L1.5 task instance (TB-style directory).

  python3 -m l15.build <task_module> <seed> <out_task_dir>

Writes
  instruction.md            what the agent sees (+ hints/H*.md appended at launch for hint ablations)
  environment/app/          the agent's /app: bin/lab (lab client), docs/, starter files
  hidden/world.json         the instance: params, secret noise salt, budget, cached truth  (never inside a sandbox)
  task.toml                 declared artifacts, grader, budget
The module supplies: sample_params(seed), BUDGET, instruction(params, truth), docs(params) -> {relpath: text},
hints(params) -> {1: text, 2: text}, optional starter(params) -> {relpath: text}, instance_gate(params)."""
import importlib, json, os, secrets, shutil, stat, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def build(modname, seed, out):
    mod = importlib.import_module("l15.tasks." + modname)
    p = mod.sample_params(seed)
    ok, info = mod.instance_gate(p)
    truth = {k: v for k, v in info.items()}
    os.makedirs(os.path.join(out, "hidden"), exist_ok=True)
    app = os.path.join(out, "environment", "app")
    os.makedirs(os.path.join(app, "bin"), exist_ok=True)
    cfg = {"task": modname, "seed": seed, "params": p, "salt": secrets.token_hex(12), "budget": mod.BUDGET, "truth": truth}
    if hasattr(mod, "instance_truth"):
        truth.update(mod.instance_truth(cfg))   # salt-dependent reference numbers quoted in the instruction
    if hasattr(mod, "instance_budget"):
        # tasks whose budget unit is the instance's own cost scale (K6: GPU-seconds) set it per instance,
        # so that "the lab costs a fraction of the real run" holds for every instance, not just the median one
        cfg["budget"] = float(mod.instance_budget(p, truth))
    json.dump(cfg, open(os.path.join(out, "hidden", "world.json"), "w"), indent=1, default=float)
    open(os.path.join(out, "instruction.md"), "w").write(mod.instruction(p, truth))
    lab = os.path.join(app, "bin", "lab")
    shutil.copy(os.path.join(HERE, "cli.py"), lab)
    os.chmod(lab, os.stat(lab).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for rel, txt in {**mod.docs(p), **(mod.starter(p) if hasattr(mod, "starter") else {})}.items():
        fp = os.path.join(app, rel); os.makedirs(os.path.dirname(fp), exist_ok=True); open(fp, "w").write(txt)
    if hasattr(mod, "starter_files"):
        mod.starter_files(p, app)            # binary starter files (v10 scaleup: sample.npz / dev.npz)
    os.makedirs(os.path.join(out, "hints"), exist_ok=True)
    for k, txt in mod.hints(p).items():
        open(os.path.join(out, "hints", "H%d.md" % k), "w").write(txt)
    # Stamp the module's grading constants into the built task and check them at grade time.
    #
    # Why: the grader imports the LIVE module while the instruction is frozen at build time, so re-tuning a
    # constant silently re-grades every run that was built before the change against a bar it was never
    # shown.  This bit us for real: three k3-edge-w1 runs were built and graded under WIDTH_CAP=0.40, the
    # cap was later measured down to 0.20, and a re-grade turned a passing gpt-6 run into "R0_artifact:
    # interval width 0.3800 exceeds the cap 0.20".  The agent had done nothing wrong.  A stale task is a
    # data-integrity bug, not a model result, so it has to fail loudly rather than score low.
    consts = {k: getattr(mod, k) for k in dir(mod)
              if k.isupper() and isinstance(getattr(mod, k), (int, float, str)) and not k.startswith("_")}
    json.dump(consts, open(os.path.join(out, "hidden", "grading_consts.json"), "w"), indent=1, default=float)
    open(os.path.join(out, "task.toml"), "w").write(
        '[task]\nname = "%s"\nmodule = "%s"\nseed = %d\n\n[verifier]\nartifacts = %s\ngrader = "python3 -m l15.grade <run_dir>"\n'
        'main_score = "pass = all rubric items R* true (mechanical, against the simulator)"\n\n[lab]\nbudget = %g\nunit = "%s"\n'
        % (os.path.basename(out), modname, seed, json.dumps(mod.World.ARTIFACTS), cfg["budget"], mod.World.BUDGET_UNIT))
    return out, ok, info


if __name__ == "__main__":
    out, ok, info = build(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    print(out, "screen_ok=%s" % ok)
