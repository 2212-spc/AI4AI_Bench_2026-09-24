"""Sandboxed runner for submitted algorithms (policy/search code).

The submitted code runs in bwrap with no network, no /app, no run dir: only a private copy of the
submission directory (read-only) and the task's client library.  It talks to a host-side *driver* (which
holds the hidden world) over JSON lines on stdin/stdout.  The client library moves the user's print() to
stderr so stray output cannot corrupt the protocol.

  run(sub_dir, entry_module, client_lib_path, driver, timeout_s) -> {"ok", "error", "stderr_tail", "n_msgs", "elapsed"}

driver.handle(msg: dict) -> dict   (reply);  driver.finished -> bool (after a terminal 'submit' message)
"""
import json, os, resource, select, shutil, subprocess, tempfile, time

BOOT = r'''
import sys, json, importlib, traceback
sys.path.insert(0, "/sub"); sys.path.insert(0, "/l15lib")
import l15env
try:
    m = importlib.import_module(%r)
    fn = getattr(m, %r)
    fn(l15env.Env())
    l15env._finish()
except SystemExit:
    raise
except BaseException as e:
    traceback.print_exc(file=sys.stderr)
    l15env._send({"op": "__crash__", "error": "%%s: %%s" %% (type(e).__name__, str(e)[:300])})
    sys.exit(1)
'''


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (1536 << 20, 1536 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.setsid()


def run(sub_dir, entry_module, entry_fn, client_lib_path, driver, timeout_s=60.0):
    tmp = tempfile.mkdtemp(prefix="l15run_")
    try:
        sub = os.path.join(tmp, "sub"); lib = os.path.join(tmp, "lib")
        shutil.copytree(sub_dir, sub, ignore=shutil.ignore_patterns("*.pyc", "__pycache__", ".lab_token", "*.jsonl"),
                        symlinks=False, ignore_dangling_symlinks=True)
        os.makedirs(lib)
        shutil.copy(client_lib_path, os.path.join(lib, "l15env.py"))
        open(os.path.join(lib, "_boot.py"), "w").write(BOOT % (entry_module, entry_fn))
        bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
              "--symlink", "usr/bin", "/bin", "--ro-bind", sub, "/sub", "--ro-bind", lib, "/l15lib",
              "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev", "--unshare-all", "--die-with-parent",
              "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
              "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "OMP_NUM_THREADS", "1",
              "--setenv", "OPENBLAS_NUM_THREADS", "1", "--", "python3", "-u", "/l15lib/_boot.py"]
        t0 = time.time()
        errf = open(os.path.join(tmp, "stderr.txt"), "wb")
        p = subprocess.Popen(bw, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, preexec_fn=_limits)
        buf = b""; n = 0; err = None
        try:
            while True:
                rem = timeout_s - (time.time() - t0)
                if rem <= 0:
                    err = "timeout after %.0fs" % timeout_s; break
                r, _, _ = select.select([p.stdout], [], [], min(rem, 5.0))
                if not r:
                    if p.poll() is not None:
                        err = err or "exited without submitting (code %s)" % p.returncode; break
                    continue
                chunk = os.read(p.stdout.fileno(), 1 << 16)
                if not chunk:
                    if not driver.finished:
                        err = "exited without submitting (code %s)" % p.poll()
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    n += 1
                    try:
                        msg = json.loads(line)
                    except Exception:
                        msg = {"op": "__garbage__"}
                    if msg.get("op") == "__crash__":
                        err = "crashed: " + str(msg.get("error"))[:300]; break
                    if msg.get("op") == "__done__":
                        if not driver.finished:
                            err = "returned without submitting"
                        break
                    try:
                        rep = driver.handle(msg)
                    except Exception as e:
                        rep = {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
                    try:
                        p.stdin.write((json.dumps(rep, default=float) + "\n").encode()); p.stdin.flush()
                    except BrokenPipeError:
                        break
                if err or (msg if n else {}).get("op") == "__done__":
                    break
        finally:
            try:
                os.killpg(p.pid, 9)
            except Exception:
                pass
            p.wait()
            errf.close()
        tail = open(os.path.join(tmp, "stderr.txt"), "rb").read()[-3000:].decode(errors="replace")
        ok = driver.finished and err is None
        return {"ok": ok, "error": err if not ok else None, "stderr_tail": tail, "n_msgs": n,
                "elapsed": round(time.time() - t0, 2)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
