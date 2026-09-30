"""Call a function from an agent-written solution file in a separate process.

The solution never shares a process with hidden generators or answers: the host
pickles the *inputs it is allowed to see* to a temp dir, a child python loads the
solution file from a scratch copy, calls the function, and pickles the result back.
"""
import os, pickle, shutil, subprocess, sys, tempfile, textwrap, time

_CHILD = textwrap.dedent(r'''
import importlib.util, pickle, sys, os
sol_path, fn_name, in_path, out_path = sys.argv[1:5]
os.chdir(os.path.dirname(sol_path))
sys.path.insert(0, os.path.dirname(sol_path))
spec = importlib.util.spec_from_file_location("solution", sol_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
with open(in_path, "rb") as f: args, kwargs = pickle.load(f)
res = getattr(mod, fn_name)(*args, **kwargs)
with open(out_path, "wb") as f: pickle.dump(res, f)
''')


def call(solution_dir, sol_file, fn_name, args=(), kwargs=None, timeout=600, extra_files=()):
    """Copy solution_dir (the agent's editable dir, e.g. workdir) to scratch and call fn."""
    kwargs = kwargs or {}
    tmp = tempfile.mkdtemp(prefix="v11call_")
    try:
        work = os.path.join(tmp, "w")
        shutil.copytree(solution_dir, work, ignore=shutil.ignore_patterns("__pycache__", "*.npz", "data"))
        for src in extra_files:
            shutil.copy(src, work)
        sol_path = os.path.join(work, sol_file)
        in_path, out_path = os.path.join(tmp, "in.pkl"), os.path.join(tmp, "out.pkl")
        with open(in_path, "wb") as f:
            pickle.dump((args, kwargs), f)
        t0 = time.time()
        env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", LOKY_MAX_CPU_COUNT="4")
        p = subprocess.run([sys.executable, "-c", _CHILD, sol_path, fn_name, in_path, out_path],
                           capture_output=True, text=True, timeout=timeout, env=env)
        if p.returncode != 0:
            raise RuntimeError(f"solution failed (rc={p.returncode}):\n{p.stderr[-3000:]}")
        with open(out_path, "rb") as f:
            return pickle.load(f), time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
