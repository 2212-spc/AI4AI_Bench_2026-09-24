"""Create one agent run for one task instance (v7 / L1.5).
  python3 launch.py <task_dir> <cc|gpt> <model> <hint 0|1|2> <run_name> [effort]
The run dir (RUN_ROOT/<run_name>) gets: app/ (copy of the task's environment/app + .lab_token + bin/lab),
lab.json (token, task_dir - read by the lab server, outside the sandbox), state.json (agent state).
The hint level appends hints/H<k>.md to the instruction; nothing else differs between hint levels."""
import json, os, secrets, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v7/runs")


def launch(task_dir, agent, model, hint, name, effort="high"):
    task_dir = os.path.abspath(task_dir)
    rd = os.path.join(RUN_ROOT, name)
    if os.path.exists(os.path.join(rd, "state.json")):
        raise SystemExit("run exists: " + rd)
    os.makedirs(rd, exist_ok=True)
    instr = open(os.path.join(task_dir, "instruction.md")).read()
    if int(hint) > 0:
        instr += "\n" + open(os.path.join(task_dir, "hints", "H%d.md" % int(hint))).read()
    ip = os.path.join(rd, "instruction.md"); open(ip, "w").write(instr)
    app_src = os.path.join(task_dir, "environment", "app")
    if agent == "cc":
        subprocess.check_call(["python3", os.path.join(HERE, "cc_agent.py"), "init", rd, app_src, ip, model])
    else:
        subprocess.check_call(["python3", os.path.join(HERE, "gpt_agent.py"), "init", rd, app_src, ip, model, effort])
    tok = secrets.token_hex(16)
    open(os.path.join(rd, "app", ".lab_token"), "w").write(tok)
    json.dump({"token": tok, "task_dir": task_dir, "agent": agent, "model": model, "hint": int(hint),
               "task": os.path.basename(task_dir)}, open(os.path.join(rd, "lab.json"), "w"))
    return rd


if __name__ == "__main__":
    a = sys.argv[1:]
    print(launch(*a[:5], *(a[5:6] or ["high"])))
