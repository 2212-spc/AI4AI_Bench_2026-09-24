"""Run a command with a directory bound at /app and another read-only at /tests, as harbor does."""
import subprocess, tempfile


def bwrap(app, tests, argv, timeout=900):
    logs = tempfile.mkdtemp()
    bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
          "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
          "--bind", app, "/app", "--bind", logs, "/logs", "--tmpfs", "/home",
          "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/app",
          "--clearenv", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
          "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8"]
    bw += (["--ro-bind", tests, "/tests"] if tests else [])
    p = subprocess.run(bw + ["--"] + argv, capture_output=True, text=True, timeout=timeout)
    return p, logs
