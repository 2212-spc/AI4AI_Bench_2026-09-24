"""Resumable Responses-API agent loop for gpt-6-astra (codex-like shell tool).
Each invocation advances a run until the deadline, persisting state to disk, because the
host kills all processes after each ~170 s tool call.
  python gpt_agent.py init <run_dir> <app_src_dir> <instruction.md> [--effort high]
  python gpt_agent.py tick <run_dir> <deadline_epoch>
Agent commands run in bwrap with only <run_dir>/app visible (as /app); no API key inside."""
import json, os, subprocess, sys, time, urllib.request, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gateway import PLACEHOLDER

API = os.environ.get("GATEWAY_URL", "http://localhost:8080") + "/v1/responses"
MODEL = "gpt-6-astra"
MAX_CALLS = 150
OUT_LIMIT = 12000
TOOLS = [{
    "type": "function", "name": "shell",
    "description": ("Runs a command in the sandbox and returns stdout+stderr and exit code. "
                    "Pass argv, e.g. [\"bash\",\"-lc\",\"ls -la\"]. Working directory is /app. "
                    "An `apply_patch` command is available: [\"apply_patch\", \"*** Begin Patch\\n*** Update File: path\\n@@\\n-old\\n+new\\n*** End Patch\"]. "
                    "Each call is a fresh process (no persistent shell state). Max 150 s per call."),
    "parameters": {"type": "object", "properties": {
        "command": {"type": "array", "items": {"type": "string"}},
        "workdir": {"type": "string"},
        "timeout_ms": {"type": "number"}}, "required": ["command"]}}]

SYSTEM = ("You are an autonomous software/ML engineering agent working in a Linux sandbox (Python 3.10 + numpy; "
          "no internet package installs). There is no human available: never ask questions, make reasonable "
          "decisions and keep working until the task is fully complete. Use the `shell` tool to inspect files, "
          "run experiments and edit code. When you are completely done, reply with a brief final summary and "
          "do not call any tool in that final message.")


def key():
    # 2026-09-25: this loop talks only to the loopback gateway, which overwrites the credential, so it no
    # longer reads the real key at all - the gateway is now literally the only process that holds it.
    if not os.environ.get("GATEWAY_URL", "").startswith("http://127.0.0.1"):
        raise SystemExit("gpt_agent: GATEWAY_URL not set - start harness/gateway.py first (drive.py does)")
    return PLACEHOLDER


def load(rd):
    return json.load(open(os.path.join(rd, "state.json")))


def save(rd, st):
    tmp = os.path.join(rd, "state.json.tmp")
    json.dump(st, open(tmp, "w")); os.replace(tmp, os.path.join(rd, "state.json"))


def log(rd, rec):
    rec["t"] = time.time()
    with open(os.path.join(rd, "traj.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_cmd(rd, argv, workdir, timeout):
    app = os.path.join(rd, "app")
    wd = workdir or "/app"
    if not wd.startswith("/app") and not wd.startswith("/tmp"):
        wd = "/app"
    if argv and argv[0] == "apply_patch":
        argv = ["/opt/tools/apply_patch"] + argv[1:]
    bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
          "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
          "--bind", app, "/app", "--ro-bind", "/tmp/bench2/harness/bin", "/opt/tools",
          "--bind", os.path.join(rd, "sbx_tmp"), "/tmp", "--tmpfs", "/home", "--proc", "/proc", "--dev", "/dev",
          "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--chdir", wd,
          "--clearenv", "--setenv", "PATH", "/opt/tools:/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", "/tmp",
          "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--"]
    t = time.time()
    try:
        p = subprocess.run(bw + argv, capture_output=True, timeout=timeout)
        out = p.stdout.decode(errors="replace") + p.stderr.decode(errors="replace")
        code = p.returncode
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")) + "\n[timed out after %ds]" % timeout
        code = 124
    except Exception as e:
        out = "failed to execute: %r" % e; code = 127
    if len(out) > OUT_LIMIT:
        out = out[:OUT_LIMIT // 2] + "\n...[%d chars truncated]...\n" % (len(out) - OUT_LIMIT) + out[-OUT_LIMIT // 2:]
    return {"exit_code": code, "duration_s": round(time.time() - t, 1), "output": out}


def effort_now(st):
    """Harness limit: a single API call cannot outlive one ~155 s host tool call, and gpt-6-astra at high
    effort exceeds that once the transcript passes ~20k tokens.  A read timeout loses the whole call, so
    after the first one the run is *stickily* downgraded to medium effort rather than burning a slice per
    retry.  Every downgrade is logged; the report states which fraction of calls ran at which effort."""
    if st["effort"] != "high":
        return st["effort"]
    return "medium" if (st.get("forced_medium") or st.get("consec_timeouts", 0) >= 1) else "high"


def call_api(st, timeout):
    body = {"model": MODEL, "input": st["items"], "tools": TOOLS, "store": False,
            "reasoning": {"effort": effort_now(st)}, "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pending_calls(st):
    done = {it["call_id"] for it in st["items"] if it.get("type") == "function_call_output"}
    return [it for it in st["items"] if it.get("type") == "function_call" and it["call_id"] not in done]


def tick(rd, deadline):
    st = load(rd)
    while st["status"] == "running":
        now = time.time()
        # 1) execute outstanding tool calls
        pc = pending_calls(st)
        if pc:
            for it in pc:
                rem = deadline - time.time()
                if rem < 15:
                    save(rd, st); return
                try:
                    args = json.loads(it.get("arguments") or "{}")
                except Exception:
                    args = {}
                argv = args.get("command") or []
                if isinstance(argv, str): argv = ["bash", "-lc", argv]
                to = min(150, max(5, (args.get("timeout_ms") or 120000) / 1000), rem - 10)
                res = run_cmd(rd, argv, args.get("workdir"), to) if argv else {"exit_code": 2, "output": "empty command", "duration_s": 0}
                st["items"].append({"type": "function_call_output", "call_id": it["call_id"],
                                    "output": json.dumps(res)})
                log(rd, {"kind": "tool", "argv": argv, "exit": res["exit_code"], "dur": res["duration_s"], "out": res["output"][:4000]})
                save(rd, st)
            continue
        # 2) model call
        if st["n_calls"] >= MAX_CALLS:
            st["status"] = "max_calls"; save(rd, st); break
        rem = deadline - time.time()
        if rem < 75:
            save(rd, st); return
        t0 = time.time()
        try:
            resp = call_api(st, timeout=rem - 5)
        except Exception as e:
            st["api_errors"] += 1
            if "timed out" in repr(e): st["consec_timeouts"] = st.get("consec_timeouts", 0) + 1
            log(rd, {"kind": "api_error", "err": repr(e)[:300], "elapsed": round(time.time() - t0, 1)})
            save(rd, st)
            if st["api_errors"] > 40:
                st["status"] = "api_failed"; save(rd, st)
            return
        st["n_calls"] += 1
        if effort_now(st) != st["effort"]:
            st["forced_medium"] = True
            st["effort_downgrades"] = st.get("effort_downgrades", 0) + 1; log(rd, {"kind": "effort_downgrade", "call": st["n_calls"]})
        st["consec_timeouts"] = 0
        u = resp.get("usage") or {}
        st["usage"]["input"] += u.get("input_tokens", 0); st["usage"]["output"] += u.get("output_tokens", 0)
        st["usage"]["reasoning"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        st["usage"]["cached"] += (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
        outs = resp.get("output") or []
        for o in outs:
            o.pop("status", None)
            st["items"].append(o)
        texts = [c.get("text", "") for o in outs if o.get("type") == "message" for c in (o.get("content") or [])]
        calls = [o for o in outs if o.get("type") == "function_call"]
        log(rd, {"kind": "model", "latency": round(time.time() - t0, 1), "n_calls": st["n_calls"], "text": "\n".join(texts)[:4000],
                 "calls": [c.get("arguments", "")[:2000] for c in calls], "usage": u.get("output_tokens")})
        if not calls:
            st["status"] = "done"; st["final_text"] = "\n".join(texts)
        save(rd, st)


def init(rd, app_src, instr_path, effort="high"):
    os.makedirs(rd, exist_ok=True)
    if os.path.exists(os.path.join(rd, "app")): shutil.rmtree(os.path.join(rd, "app"))
    shutil.copytree(app_src, os.path.join(rd, "app"))
    os.makedirs(os.path.join(rd, "sbx_tmp"), exist_ok=True)
    instr = open(instr_path).read()
    st = {"status": "running", "effort": effort, "n_calls": 0, "api_errors": 0, "started": time.time(),
          "usage": {"input": 0, "output": 0, "reasoning": 0, "cached": 0},
          "items": [{"role": "developer", "content": SYSTEM},
                    {"role": "user", "content": instr}]}
    save(rd, st)


if __name__ == "__main__":
    if sys.argv[1] == "init":
        a = [x for x in sys.argv[5:] if not x.startswith("--")]
        init(sys.argv[2], sys.argv[3], sys.argv[4], *(a[:1] or ["high"]))
    elif sys.argv[1] == "tick":
        tick(sys.argv[2], float(sys.argv[3]))
