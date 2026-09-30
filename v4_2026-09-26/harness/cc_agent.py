"""Resumable Claude Code (headless) runner inside bwrap (v3: model is a per-run parameter).
  python cc_agent.py init <run_dir> <app_src_dir> <instruction.md> <model>
  python cc_agent.py tick <run_dir> <deadline_epoch>"""
import json, os, subprocess, sys, time, uuid, shutil, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gateway import PLACEHOLDER   # the only credential string the sandbox ever sees

CONT = ("Continue working on the task from where you left off (your previous turn was interrupted by an "
        "infrastructure time slice, nothing else changed). If you are already completely done, reply with a brief final summary.")
PREAMBLE = ("You are working autonomously in a Linux sandbox (Python 3.10 + numpy; no package installs, no internet needed). "
            "No human is available: never ask questions, make reasonable decisions and keep working until the task is fully complete.\n\n")


def load(rd): return json.load(open(os.path.join(rd, "state.json")))


def save(rd, st):
    json.dump(st, open(os.path.join(rd, "state.json.tmp"), "w")); os.replace(os.path.join(rd, "state.json.tmp"), os.path.join(rd, "state.json"))


def init(rd, app_src, instr_path, model):
    os.makedirs(rd, exist_ok=True)
    if os.path.exists(rd + "/app"): shutil.rmtree(rd + "/app")
    shutil.copytree(app_src, rd + "/app")
    os.makedirs(rd + "/home", exist_ok=True); os.makedirs(rd + "/sbx_tmp", exist_ok=True)
    save(rd, {"status": "running", "model": model, "session": str(uuid.uuid4()), "n_ticks": 0, "cost": 0.0, "started": time.time(),
              "instr": PREAMBLE + open(instr_path).read(), "turns": 0})


def tick(rd, deadline):
    st = load(rd)
    if st["status"] != "running": return
    rem = deadline - time.time() - 4
    if rem < 40: return
    # Fix 2026-09-25: the sandbox gets the gateway placeholder, never the real key.  The previous line read
    # the real key from SEC_KEY and exported it into bwrap as ANTHROPIC_API_KEY, so it was readable from
    # /proc/self/environ inside the sandbox.  The gateway overwrites Authorization/x-api-key unconditionally.
    if not os.environ.get("GATEWAY_URL", "").startswith("http://127.0.0.1"):
        raise SystemExit("cc_agent: GATEWAY_URL not set - start harness/gateway.py first (drive.py does)")
    key = PLACEHOLDER
    first = st["n_ticks"] == 0
    prompt = st["instr"] if first else CONT
    cc = ["claude", "-p", prompt, "--model", st["model"], "--output-format", "stream-json", "--verbose",
          "--dangerously-skip-permissions", "--disallowedTools", "WebSearch", "WebFetch"]
    cc += (["--session-id", st["session"]] if first else ["--resume", st["session"]])
    env_keep = {k: v for k, v in os.environ.items() if k.lower().endswith("_proxy") or k in ("NO_PROXY", "no_proxy")}
    bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
          "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
          "--bind", rd + "/app", "/app", "--bind", rd + "/home", "/home/agent", "--bind", rd + "/sbx_tmp", "/tmp",
          "--proc", "/proc", "--dev", "/dev", "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--chdir", "/app",
          "--clearenv", "--setenv", "PATH", "/app/bin:/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", "/home/agent",
          "--setenv", "LANG", "C.UTF-8", "--setenv", "ANTHROPIC_BASE_URL", os.environ["GATEWAY_URL"],
          "--setenv", "ANTHROPIC_API_KEY", key, "--setenv", "DISABLE_AUTOUPDATER", "1",
          "--setenv", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1", "--setenv", "DISABLE_TELEMETRY", "1",
          "--setenv", "BASH_DEFAULT_TIMEOUT_MS", "120000", "--setenv", "BASH_MAX_TIMEOUT_MS", "150000"]
    for k, v in env_keep.items(): bw += ["--setenv", k, v]
    out_path = rd + "/cc_stream_%03d.jsonl" % st["n_ticks"]
    st["n_ticks"] += 1; save(rd, st)
    with open(out_path, "wb") as fo:
        p = subprocess.Popen(bw + ["--"] + cc, stdout=fo, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            p.wait(timeout=rem)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGINT)
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
    result = None
    for line in open(out_path, errors="replace"):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "assistant": st["turns"] += 1
        if ev.get("type") == "result": result = ev
    if result:
        st["cost"] += result.get("total_cost_usd") or 0.0
        if not timed_out and result.get("subtype") == "success":
            st["status"] = "done"; st["final_text"] = (result.get("result") or "")[:4000]
        elif not timed_out:
            st["last_result_subtype"] = result.get("subtype")
    st["last_tick_timed_out"] = timed_out
    save(rd, st)


if __name__ == "__main__":
    if sys.argv[1] == "init": init(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else: tick(sys.argv[2], float(sys.argv[3]))
