"""Resumable Messages-API agent loop for Claude models (v10).

Why this replaces cc_agent.py (the Claude Code CLI driven through SIGINT):
  every host tool call is capped at ~178 s and kills all child processes, so a run is advanced in ticks.
  The CLI could only be interrupted by SIGINT at the tick deadline, which threw away whatever the model
  was thinking at that moment; in v8, 22 of 36 Fable ticks on r_crit produced no assistant event at all.
  This loop instead executes WHOLE steps: it only starts a model call when the rest of the tick can hold
  it, and persists every completed step (messages, pending tool calls, usage) to state.json.

Harness condition that must be reported with every result: a single model call has to fit into one tick,
so extended thinking is capped at THINK_BUDGET tokens per call (default 4000) and max_tokens at MAX_TOK.
Thinking is interleaved (the model can think again after every tool result), so the cap is per step,
not per task.

  python claude_agent.py init <run_dir> <app_src_dir> <instruction.md> <model>
  python claude_agent.py tick <run_dir> <deadline_epoch>

Tools: Claude Code names/schemas (Bash, Read, Write, Edit, Glob, Grep) because the upstream channel injects
the Claude Code tool list; Bash runs in a bwrap sandbox with only /app and a private /tmp visible.
Calls are streamed and hedged (a duplicate request is started if no response headers arrive within 20 s /
45 s; `hedges` in state.json counts them).  No credential is ever visible inside the sandbox; the loop
talks only to the loopback gateway."""
import copy, json, os, subprocess, sys, time, urllib.request, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gateway import PLACEHOLDER

API = os.environ.get("GATEWAY_URL", "http://localhost:8080") + "/v1/messages"
TOOLS_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
MAX_CALLS = 150
OUT_LIMIT = 12000
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "4000"))
MAX_TOK = int(os.environ.get("CLAUDE_MAX_TOK", "7000"))
CTX_SOFT = 150000          # input tokens above which old tool outputs are elided

# The Claude channel behind the gateway injects the Claude Code tool set whenever a request carries tools
# (observed 2026-09-29: +~19k cached tokens per call and the model calling `Bash` with a `description`
# field although we had declared `bash`).  So the harness declares - and implements - exactly the Claude
# Code names and input schemas; whichever copy of the definitions the model reads, its calls work.
TOOL_NAMES = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
TOOLS = [
    {"name": "Bash", "description": (
        "Run a bash command in the sandbox (Linux, Python 3.10 + numpy, no network installs). Working directory "
        "is /app. Each call is a fresh process: no shell state persists between calls. Returns stdout+stderr "
        "(truncated to 12000 chars) and the exit code. Max runtime per call 150000 ms (`timeout`, in ms)."),
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string"}, "timeout": {"type": "number"}, "description": {"type": "string"}},
         "required": ["command"]}},
    {"name": "Read", "description": "Read a text file (absolute path under /app or /tmp). Optional 1-based `offset` line and `limit` lines.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "offset": {"type": "number"},
                                                       "limit": {"type": "number"}}, "required": ["file_path"]}},
    {"name": "Write", "description": "Create or overwrite a text file under /app or /tmp.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["file_path", "content"]}},
    {"name": "Edit", "description": ("Replace `old_string` with `new_string` in a file under /app or /tmp. Fails unless "
                                     "`old_string` occurs exactly once, or set `replace_all`."),
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                                                       "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
                      "required": ["file_path", "old_string", "new_string"]}},
    {"name": "Glob", "description": "List files matching a glob pattern (e.g. **/*.py) under `path` (default /app).",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                      "required": ["pattern"]}},
    {"name": "Grep", "description": "Search file contents with a regular expression (grep -rnE) under `path` (default /app).",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                       "glob": {"type": "string"}}, "required": ["pattern"]}},
]

SYSTEM = ("You are an autonomous software/ML research agent working in a Linux sandbox (Python 3.10 + numpy; "
          "no internet, no package installs). There is no human available: never ask questions, make reasonable "
          "decisions and keep working until the task is fully complete. Use the tools to inspect files, run "
          "experiments and edit code. When you are completely done, reply with a brief final summary and do not "
          "call any tool in that final message.")

SBX_HOST_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}


def load(rd):
    return json.load(open(os.path.join(rd, "state.json")))


def save(rd, st):
    tmp = os.path.join(rd, "state.json.tmp")
    json.dump(st, open(tmp, "w")); os.replace(tmp, os.path.join(rd, "state.json"))


def log(rd, rec):
    rec["t"] = time.time()
    with open(os.path.join(rd, "traj.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_cmd(rd, cmd, timeout):
    app = os.path.join(rd, "app")
    bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
          "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
          "--bind", app, "/app", "--ro-bind", TOOLS_BIN, "/opt/tools",
          "--bind", os.path.join(rd, "sbx_tmp"), "/tmp", "--tmpfs", "/home", "--proc", "/proc", "--dev", "/dev",
          "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--chdir", "/app",
          "--clearenv", "--setenv", "PATH", "/app/bin:/opt/tools:/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", "/tmp",
          "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--", "bash", "-lc", cmd]
    t = time.time()
    try:
        # bwrap is PID 1 in the sandbox and /proc/1/environ shows ITS env, so it gets a minimal one (v8 fix)
        p = subprocess.run(bw, capture_output=True, timeout=timeout, env=SBX_HOST_ENV)
        out = p.stdout.decode(errors="replace") + p.stderr.decode(errors="replace"); code = p.returncode
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")) + "\n[timed out after %ds]" % timeout
        code = 124
    except Exception as e:
        out = "failed to execute: %r" % e; code = 127
    if len(out) > OUT_LIMIT:
        out = out[:OUT_LIMIT // 2] + "\n...[%d chars truncated]...\n" % (len(out) - OUT_LIMIT) + out[-OUT_LIMIT // 2:]
    return {"exit_code": code, "duration_s": round(time.time() - t, 1), "output": out}


def host_path(rd, p):
    """Map a sandbox path (/app/..., /tmp/..., or relative to /app) to the host, refusing escapes."""
    if not p.startswith("/"):
        p = "/app/" + p
    for pre, sub in (("/app", "app"), ("/tmp", "sbx_tmp")):
        if p == pre or p.startswith(pre + "/"):
            base = os.path.realpath(os.path.join(rd, sub))
            hp = os.path.realpath(os.path.join(base, p[len(pre):].lstrip("/")))
            if hp == base or hp.startswith(base + os.sep):
                return hp
    raise ValueError("path must be under /app or /tmp")


def _sh(x):
    return "'" + str(x).replace("'", "'\\''") + "'"


def run_tool(rd, name, inp, rem):
    try:
        if name == "Bash":
            t = inp.get("timeout")
            t = float(t) / 1000.0 if t else 120.0
            to = min(150, max(5, t), rem - 10)
            r = run_cmd(rd, str(inp.get("command", "")), to)
            return "exit_code=%d (%.1fs)\n%s" % (r["exit_code"], r["duration_s"], r["output"]), r["exit_code"] != 0
        if name == "Read":
            hp = host_path(rd, inp["file_path"]); lines = open(hp, errors="replace").read().split("\n")
            off = max(1, int(inp.get("offset") or 1)); lim = int(inp.get("limit") or 2000)
            out = "\n".join("%6d\t%s" % (i, l) for i, l in enumerate(lines[off - 1:off - 1 + lim], start=off))
            if len(out) > OUT_LIMIT:
                out = out[:OUT_LIMIT] + "\n...[truncated; use offset/limit]"
            return out, False
        if name == "Write":
            hp = host_path(rd, inp["file_path"]); os.makedirs(os.path.dirname(hp), exist_ok=True)
            open(hp, "w").write(inp.get("content", ""))
            return "wrote %d chars to %s" % (len(inp.get("content", "")), inp["file_path"]), False
        if name == "Edit":
            hp = host_path(rd, inp["file_path"]); s = open(hp).read(); n = s.count(inp["old_string"])
            if n == 0 or (n > 1 and not inp.get("replace_all")):
                return "edit failed: old_string occurs %d times in %s" % (n, inp["file_path"]), True
            s = s.replace(inp["old_string"], inp["new_string"]) if inp.get("replace_all") else s.replace(inp["old_string"], inp["new_string"], 1)
            open(hp, "w").write(s)
            return "edited %s (%d replacement%s)" % (inp["file_path"], n if inp.get("replace_all") else 1, "s" if n > 1 and inp.get("replace_all") else ""), False
        if name == "Glob":
            base = inp.get("path") or "/app"
            r = run_cmd(rd, "cd %s && python3 -c 'import glob,sys; print(\"\\n\".join(sorted(glob.glob(sys.argv[1], recursive=True))[:500]))' %s" % (_sh(base), _sh(inp["pattern"])), min(30, rem - 10))
            return r["output"] or "(no matches)", r["exit_code"] != 0
        if name == "Grep":
            base = inp.get("path") or "/app"
            inc = (" --include=%s" % _sh(inp["glob"])) if inp.get("glob") else ""
            r = run_cmd(rd, "grep -rnE%s -- %s %s | head -300" % (inc, _sh(inp["pattern"]), _sh(base)), min(30, rem - 10))
            return r["output"] or "(no matches)", False
        return ("tool %r is not available in this environment. All available tools are already loaded: %s. "
                "Use Bash for everything else." % (name, ", ".join(TOOL_NAMES))), True
    except Exception as e:
        return "tool error: %s: %s" % (type(e).__name__, str(e)[:300]), True


def with_cache(msgs):
    """Put ephemeral cache breakpoints on the last block of the last two user messages."""
    m = copy.deepcopy(msgs); k = 0
    for msg in reversed(m):
        if msg["role"] == "user" and isinstance(msg["content"], list) and msg["content"]:
            msg["content"][-1]["cache_control"] = {"type": "ephemeral"}; k += 1
            if k == 2:
                break
    return m


def elide(st):
    """Keep the context under CTX_SOFT: blank the oldest tool outputs (never the instruction)."""
    n = 0
    for msg in st["messages"][1:-20]:
        if msg["role"] != "user" or not isinstance(msg["content"], list):
            continue
        for b in msg["content"]:
            if b.get("type") == "tool_result" and not b.get("_elided"):
                b["content"] = "[old output elided by the harness to save context; re-run the command if needed]"
                b["_elided"] = True; n += 1
    return n


def clean(msgs):
    out = copy.deepcopy(msgs)
    for msg in out:
        if isinstance(msg["content"], list):
            for b in msg["content"]:
                b.pop("_elided", None)
    return out


def _open_stream(body):
    import http.client, urllib.parse
    u = urllib.parse.urlparse(API)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=175)
    conn.request("POST", u.path, body=body, headers={
        "x-api-key": PLACEHOLDER, "anthropic-version": "2023-06-01", "content-type": "application/json",
        "anthropic-beta": "interleaved-thinking-2025-05-14"})
    return conn


def call_api(st, deadline, hedge_after=(20, 45), log_fn=None):
    """Streaming call with request hedging.

    The Claude channel intermittently accepts a request and never sends a byte (2026-09-29: 2 of 3 parallel
    Fable probes silent for 160 s while the third answered in 8.8 s).  A silent request costs a whole tick,
    so if no response headers arrive within hedge_after[k] seconds another identical request is started;
    the first one to answer wins and the rest are closed.  Returns the assembled message dict."""
    import threading, queue
    body = json.dumps({"model": st["model"], "max_tokens": MAX_TOK, "stream": True,
                       "thinking": {"type": "enabled", "budget_tokens": THINK_BUDGET},
                       "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                       "tools": TOOLS, "messages": with_cache(clean(st["messages"]))}).encode()
    q = queue.Queue(); conns = []; t0 = time.time()

    def attempt(k):
        try:
            c = _open_stream(body); conns.append(c)
            r = c.getresponse()
            q.put((k, c, r, None))
        except Exception as e:
            q.put((k, None, None, e))

    started = 0; errs = []; win = None
    threading.Thread(target=attempt, args=(0,), daemon=True).start(); started = 1
    while win is None:
        now = time.time()
        if now > deadline - 2:
            break
        nxt = hedge_after[started - 1] if started - 1 < len(hedge_after) else None
        wait = (t0 + nxt - now) if nxt is not None else (deadline - 2 - now)
        try:
            k, c, r, e = q.get(timeout=max(0.05, min(wait, deadline - 2 - now)))
            if e is not None or r.status != 200:
                errs.append(repr(e) if e else "HTTP %s %s" % (r.status, r.read()[:300]))
                if len(errs) >= 3 or (started >= 1 + len(hedge_after) and len(errs) >= started):
                    raise RuntimeError("all attempts failed: " + " | ".join(errs)[:600])
                continue
            win = (k, c, r)
        except queue.Empty:
            if nxt is not None and time.time() >= t0 + nxt:
                threading.Thread(target=attempt, args=(started,), daemon=True).start(); started += 1
    st["hedges"] = st.get("hedges", 0) + (started - 1)
    if win is None:
        for c in conns:
            try: c.sock and c.sock.close()
            except Exception: pass
        raise TimeoutError("timed out: no response headers from %d attempt(s)" % started)
    k, c, r = win
    for c2 in conns:
        if c2 is not c:
            try: c2.sock and c2.sock.close()
            except Exception: pass
    c.sock.settimeout(max(5, min(75, deadline - time.time())))   # a stream silent for 75 s is treated as hung
    msg = {"content": [], "usage": {}, "stop_reason": None}
    blocks = {}; pj = {}
    for raw in r:
        if time.time() > deadline:
            c.sock.close(); raise TimeoutError("timed out: stream still running at the tick deadline")
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        d = json.loads(line[5:])
        ty = d.get("type")
        if ty == "message_start":
            msg["usage"].update(d["message"].get("usage") or {})
        elif ty == "content_block_start":
            blocks[d["index"]] = dict(d["content_block"])
            if blocks[d["index"]].get("type") == "tool_use":
                pj[d["index"]] = ""
        elif ty == "content_block_delta":
            b = blocks[d["index"]]; dl = d["delta"]; dt = dl.get("type")
            if dt == "text_delta": b["text"] = b.get("text", "") + dl["text"]
            elif dt == "thinking_delta": b["thinking"] = b.get("thinking", "") + dl["thinking"]
            elif dt == "signature_delta": b["signature"] = b.get("signature", "") + dl["signature"]
            elif dt == "input_json_delta": pj[d["index"]] += dl.get("partial_json", "")
        elif ty == "content_block_stop":
            i = d["index"]
            if i in pj:
                try: blocks[i]["input"] = json.loads(pj[i]) if pj[i] else {}
                except Exception: blocks[i]["input"] = {"_unparseable": pj[i][:2000]}
        elif ty == "message_delta":
            msg["stop_reason"] = (d.get("delta") or {}).get("stop_reason")
            msg["usage"].update({k2: v for k2, v in (d.get("usage") or {}).items() if v is not None})
        elif ty == "error":
            raise RuntimeError("stream error: %s" % json.dumps(d)[:400])
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    msg["attempt_won"] = k
    return msg


def pending(st):
    last = st["messages"][-1]
    if last["role"] != "assistant":
        return []
    done = set(st.get("done_ids", []))
    return [b for b in last["content"] if b.get("type") == "tool_use" and b["id"] not in done]


def tick(rd, deadline):
    if not os.environ.get("GATEWAY_URL", "").startswith("http://127.0.0.1"):
        raise SystemExit("claude_agent: GATEWAY_URL not set - drive.py starts the gateway")
    st = load(rd)
    while st["status"] == "running":
        pc = pending(st)
        if pc:
            for b in pc:
                rem = deadline - time.time()
                if rem < 15:
                    save(rd, st); return
                out, err = run_tool(rd, b["name"], b.get("input") or {}, rem)
                st.setdefault("partial_results", []).append(
                    {"type": "tool_result", "tool_use_id": b["id"], "content": out, **({"is_error": True} if err else {})})
                st.setdefault("done_ids", []).append(b["id"])
                log(rd, {"kind": "tool", "name": b["name"], "input": json.dumps(b.get("input"))[:3000], "err": err, "out": out[:4000]})
                save(rd, st)
            st["messages"].append({"role": "user", "content": st.pop("partial_results")})
            st["done_ids"] = []
            save(rd, st)
            continue
        if st["n_calls"] >= MAX_CALLS:
            st["status"] = "max_calls"; save(rd, st); break
        rem = deadline - time.time()
        lat = sorted(st.get("recent_lat", []))[-5:]
        need = min(150, max(45, 1.25 * (lat[-1] if lat else 60) + 8))
        if rem < need:
            st["idle_skips"] = st.get("idle_skips", 0) + 1
            save(rd, st); return
        if st.get("last_in_tokens", 0) > CTX_SOFT:
            n = elide(st); log(rd, {"kind": "elide", "n": n})
        t0 = time.time()
        try:
            resp = call_api(st, deadline - 1)
        except Exception as e:
            body = ""
            if hasattr(e, "read"):
                try: body = e.read().decode(errors="replace")[:500]
                except Exception: pass
            st["api_errors"] += 1
            if "timed out" in repr(e) or "Timeout" in repr(e): st["timeouts"] = st.get("timeouts", 0) + 1
            log(rd, {"kind": "api_error", "err": repr(e)[:300], "body": body, "elapsed": round(time.time() - t0, 1)})
            if st["api_errors"] > 40:
                st["status"] = "api_failed"
            save(rd, st); return
        dt = round(time.time() - t0, 1)
        st["n_calls"] += 1
        st["recent_lat"] = (st.get("recent_lat", []) + [dt])[-8:]
        u = resp.get("usage") or {}
        st["last_in_tokens"] = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        for k_src, k_dst in (("input_tokens", "input"), ("output_tokens", "output"),
                             ("cache_read_input_tokens", "cached"), ("cache_creation_input_tokens", "cache_write")):
            st["usage"][k_dst] = st["usage"].get(k_dst, 0) + (u.get(k_src) or 0)
        content = resp.get("content") or []
        stop = resp.get("stop_reason")
        if stop is None:
            # 2026-09-29: the relay sometimes holds a request ~120 s and then closes a 200 stream with no
            # message_delta (no stop_reason, usually no content).  That is a transport failure, not the model
            # ending its turn - the first version of this loop recorded it as a final empty answer and ended
            # four Fable runs at call 1.  Retry instead; nothing from a cut stream is kept.
            st["n_calls"] -= 1
            st["api_errors"] += 1
            st["cut_streams"] = st.get("cut_streams", 0) + 1
            log(rd, {"kind": "api_error", "err": "stream closed without stop_reason (cut stream)",
                     "elapsed": dt, "n_blocks": len(content)})
            if st["api_errors"] > 40:
                st["status"] = "api_failed"
            save(rd, st); return
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        uses = [b for b in content if b.get("type") == "tool_use"]
        think_chars = sum(len(b.get("thinking", "")) for b in content if b.get("type") == "thinking")
        log(rd, {"kind": "model", "latency": dt, "n_calls": st["n_calls"], "stop": stop, "text": "\n".join(texts)[:4000],
                 "calls": [json.dumps(b.get("input"))[:2000] for b in uses], "out_tok": u.get("output_tokens"),
                 "think_chars": think_chars})
        if stop == "max_tokens" and not uses:
            # truncated mid-step: keep visible text only (a cut thinking block cannot be replayed) and nudge
            st["truncations"] = st.get("truncations", 0) + 1
            keep = [b for b in content if b.get("type") == "text" and b.get("text")]
            if keep:
                st["messages"].append({"role": "assistant", "content": keep})
                st["messages"].append({"role": "user", "content": [{"type": "text", "text":
                    "[harness] Your last message hit the per-step output limit. Continue, in smaller steps (write long files in parts)."}]})
            elif st["truncations"] > 6:
                st["status"] = "done"; st["final_text"] = "[harness] repeated output-limit truncation"
            save(rd, st); continue
        if stop == "max_tokens" and uses:
            # the last tool_use may be cut; drop incomplete ones (they would fail to parse anyway)
            pass
        st["messages"].append({"role": "assistant", "content": content})
        if not uses:
            st["status"] = "done"; st["final_text"] = "\n".join(texts); st["stop_reason"] = stop
        save(rd, st)


def init(rd, app_src, instr_path, model):
    os.makedirs(rd, exist_ok=True)
    if os.path.exists(os.path.join(rd, "app")): shutil.rmtree(os.path.join(rd, "app"))
    shutil.copytree(app_src, os.path.join(rd, "app"))
    os.makedirs(os.path.join(rd, "sbx_tmp"), exist_ok=True)
    instr = open(instr_path).read()
    st = {"status": "running", "agent_kind": "claude_api", "model": model, "n_calls": 0, "api_errors": 0,
          "started": time.time(), "think_budget": THINK_BUDGET, "max_tokens": MAX_TOK,
          "usage": {"input": 0, "output": 0, "cached": 0, "cache_write": 0},
          "messages": [{"role": "user", "content": [{"type": "text", "text": instr}]}]}
    save(rd, st)


if __name__ == "__main__":
    if sys.argv[1] == "init":
        init(*sys.argv[2:6])
    elif sys.argv[1] == "tick":
        tick(sys.argv[2], float(sys.argv[3]))
