"""Difficulty calibration: let a coding agent attempt a task, then grade it with the host-side grader.

    python3 harness/run_agent.py --agent codex  --tasks all --runs 3 --jobs 3
    python3 harness/run_agent.py --agent claude --tasks sched_law,dedup_cv --runs 3

Per run: copy ONLY tasks/<t>/workdir to a fresh temp dir (the agent never sees hidden/, reference/, decoys/,
grade.py), launch the agent CLI there with a fixed prompt, wait (timeout), then grade the agent's workdir with
tasks/<t>/grade.py.  Results -> harness/results/<agent>/<task>/run<k>.json (+ agent log).  Summarise with
harness/summarize.py.  The agent command lines are templates: edit AGENTS below if your CLI flags differ.
"""
import argparse, concurrent.futures as cf, importlib.util, json, os, shutil, subprocess, tempfile, time, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = ("You are working in this directory on an ML engineering task. Read task.md carefully and complete it. "
          "You can run code (CPU only). Only the file(s) task.md names are graded, on hidden settings you cannot see; "
          "the dev data here is for your own checks. Work autonomously until you are confident, then stop.")
AGENTS = {  # {dir} = task workdir copy, {prompt} = PROMPT.  Model names via env so they are easy to change.
    "codex": ["codex", "exec", "--model", os.environ.get("CODEX_MODEL", "gpt-6"), "--skip-git-repo-check",
              "--sandbox", "workspace-write", "--cd", "{dir}", "{prompt}"],
    "claude": ["claude", "-p", "{prompt}", "--model", os.environ.get("CLAUDE_MODEL", "claude-opus-5"),
               "--dangerously-skip-permissions"],
}


def load_grader(task):
    spec = importlib.util.spec_from_file_location(f"grade_{task}", os.path.join(ROOT, "tasks", task, "grade.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def one_run(agent, task, k, timeout):
    out_dir = os.path.join(ROOT, "harness", "results", agent, task); os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"v11_{task}_{agent}_")
    d = os.path.join(tmp, "workdir")
    shutil.copytree(os.path.join(ROOT, "tasks", task, "workdir"), d, ignore=shutil.ignore_patterns("__pycache__"))
    cmd = [a.replace("{dir}", d).replace("{prompt}", PROMPT) for a in AGENTS[agent]]
    t0 = time.time(); status = "ok"
    with open(os.path.join(out_dir, f"run{k}.log"), "w") as log:
        try:
            subprocess.run(cmd, cwd=d, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "agent_timeout"
    agent_secs = time.time() - t0
    try:
        res = load_grader(task).grade(d)
    except Exception as e:
        res = {"pass": False, "error": repr(e)[:1000], "settings": {}}
    res.update(agent=agent, task=task, run=k, status=status, agent_secs=round(agent_secs))
    for f in ["NOTES.md"] + glob.glob(os.path.join(d, "*.py")):  # keep what the agent wrote, for inspection
        src = f if os.path.isabs(f) else os.path.join(d, f)
        if os.path.exists(src): shutil.copy(src, os.path.join(out_dir, f"run{k}_{os.path.basename(src)}"))
    with open(os.path.join(out_dir, f"run{k}.json"), "w") as f: json.dump(res, f, indent=1, default=float)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{agent} {task} run{k}: pass={res['pass']} {status} {agent_secs/60:.0f}min", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(AGENTS))
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3 * 3600)
    a = ap.parse_args()
    tasks = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "tasks", "*", "certificate.json"))
                   if json.load(open(p))["ok"]) if a.tasks == "all" else a.tasks.split(",")
    jobs = [(a.agent, t, k, a.timeout) for t in tasks for k in range(a.runs)]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: one_run(*j), jobs))


if __name__ == "__main__":
    main()
