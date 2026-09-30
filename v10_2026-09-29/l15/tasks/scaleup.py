"""SCALEUP (level L1): ship a training job, prototyped on a 5% sample, to the full corpus.  MLS-style:
the deliverable is CODE + CONFIG, graded by EXECUTING it on data the agent never sees.

The situation.  A team trains a small numpy MLP classifier on crowd-labelled data (20-30% of labels are
wrong).  Everything - train.py, config.json - was developed on a 5% uniform sample of the corpus, with
hyper-parameters picked on a 1,000-row gold dev set; their sweep log is shipped.  Production trains on the
full corpus (20x the rows) for the SAME number of epochs (20x the steps).  The agent may change anything
in the repo; production runs it twice (seed 0, seed 1) on the full corpus and scores gold test accuracy.

Difficulty mechanism (the family's signature: the standard hygiene - "tune on the dev set, ship the best
dev config" - is exactly what fails, and nothing available at sample scale says so directly):

  P1  REGULARISATION TUNED AT SAMPLE SCALE DOES NOT TRANSFER.  With AdamW's decoupled weight decay the
      weights forget their past on a timescale of 1/(lr*wd) steps; what matters for generalisation is that
      timescale measured in EPOCHS (Wang & Aitchison 2024, "How to set AdamW's weight decay as you scale model
      and dataset size").  At equal epochs, 20x the data = 20x the steps per epoch, so the sample-optimal wd
      is ~5-20x too strong at production.  Equivalently (Bayesian view): the right prior strength shrinks
      with N.  With noisy labels the sample-optimal wd is LARGE (it is fighting memorisation), so shipping
      it over-regularises production by a lot.
  P2  THE SAMPLE CANNOT SHOW IT BY TUNING, ONLY BY A SCALING EXPERIMENT.  Every dev-set comparison at
      sample scale (wd, augmentation, label smoothing, width, lr) ranks configs for N=4,000.  The agent must
      design an experiment ACROSS N (subsample 1k/2k/4k at equal epochs, watch wd* move) or apply the
      scaling rule; a better dev score at N=4,000 buys nothing - more sample-scale tuning of regularisers
      makes production worse.
  (Escape routes that also pass and are legitimate: scale wd by N_sample/N_full; extrapolate wd*(N) from
  subsamples; tune inside the production job on a held-out slice of the corpus within the CPU cap.)

Ground truth is EXECUTION, not a model: the grader runs the shipped repo on the hidden corpus in a sandbox
(bwrap, no network, one core, CPU-time and memory caps) and computes accuracy against gold labels.  The
pass bar T is certified per instance by l15/tasks/scaleup_cert.py: the scaling-rule reference must clear T
by >= MARGIN and every decoy (ship as is; re-tune on the sample, several variants) must miss it by >= MARGIN,
all measured with this very grader code path.
"""
import csv, io, json, os, shutil, subprocess, tempfile
import numpy as np
from ..core import World as _W

LEVEL = "L1"
BUDGET = 1.0                 # no lab operations; the budget exists only for the ledger bookkeeping
D_IN = 32
K_CLS = 10
TEACH_H = 64
N_VIS = 4000
N_DEV = 1000
N_TEST = 20000
RATIO = 20
N_FULL = N_VIS * RATIO       # 80,000
BS = 128
EPOCHS = 32                  # v10: was 64; one production replica must fit the harness's ~170 s call window
S_VIS = N_VIS * EPOCHS // BS  # 1,000
S_FULL = N_FULL * EPOCHS // BS  # 20,000
CPU_CAP_S = 120              # per production replica (replicas run in parallel, one core each)
WALL_CAP_S = CPU_CAP_S + 40
MEM_CAP = 2000000000
REPO_CAP_BYTES = 1000000
MARGIN = 0.015               # certification margin both ways (accuracy units)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.join(HERE, "scaleup_repo")
CERT_PATH = os.path.join(HERE, "scaleup_cert.json")
CERT_PATH_L2 = os.path.join(HERE, "scaleup_l2_cert.json")    # level L2 (duplicated corpus), see scaleup_l2.py
BASE_CFG = {"hidden": 256, "lr": 0.003, "batch_size": BS, "aug_sigma": 0.0, "label_smoothing": 0.0,
            "warmup_frac": 0.02}
TEAM_WD_GRID = (0.3, 1.0, 2.0, 3.0, 5.0, 8.0)
TEAM_LR_GRID = (0.001, 0.003)


# ------------------------------------------------------------------ data (deterministic from params)
def _teacher(ts):
    r = np.random.default_rng(ts)
    W1 = r.standard_normal((D_IN, TEACH_H)) / np.sqrt(D_IN)
    b1 = r.standard_normal(TEACH_H) * 0.5
    W2 = r.standard_normal((TEACH_H, K_CLS)) / np.sqrt(TEACH_H) * 3
    A = np.diag(np.exp(r.uniform(-1.5, 1.5, D_IN))) @ (np.eye(D_IN) + 0.3 * r.standard_normal((D_IN, D_IN)) / np.sqrt(D_IN))
    off = r.uniform(-3, 3, D_IN)
    return W1, b1, W2, A, off


def _draw(p, n, stream, noise):
    W1, b1, W2, A, off = _teacher(p["ts"])
    r = np.random.default_rng([p["data_seed"], stream])
    Z = r.standard_normal((n, D_IN))
    y = np.argmax(np.maximum(Z @ W1 + b1, 0) @ W2, 1).astype(np.int64)
    X = (Z @ A + off).astype(np.float32)
    yn = y.copy()
    m = r.random(n) < noise
    yn[m] = r.integers(0, K_CLS, int(m.sum()))
    return X, yn, y


_DATA = {}


def data(p):
    """corpus (N_FULL rows, crowd labels), the 5% sample (a subset of the corpus), dev and test (gold).

    L2 (scaleup_dup): p["dup"] = k > 1 builds the corpus from N_FULL/k distinct labelled rows, each present k
    times (exact copies, same crowd label), in random order.  dup absent or 1 is the L1 corpus, bit for bit."""
    k = json.dumps({x: p.get(x) for x in ("ts", "data_seed", "noise", "dup")}, sort_keys=True)
    if k not in _DATA:
        dup = int(p.get("dup") or 1)
        if dup == 1:
            Xc, yc, yc_gold = _draw(p, N_FULL, 1, p["noise"])
        else:
            nu = N_FULL // dup
            Xu, yu, yu_gold = _draw(p, nu, 1, p["noise"])
            src = np.tile(np.arange(nu), dup)[np.random.default_rng([p["data_seed"], 5]).permutation(nu * dup)]
            Xc, yc, yc_gold = Xu[src], yu[src], yu_gold[src]
        idx = np.sort(np.random.default_rng([p["data_seed"], 2]).choice(N_FULL, N_VIS, replace=False))
        Xd, _, yd = _draw(p, N_DEV, 3, 0.0)
        Xt, _, yt = _draw(p, N_TEST, 4, 0.0)
        _DATA.clear()
        _DATA[k] = {"corpus": (Xc, yc), "sample": (Xc[idx], yc[idx]), "dev": (Xd, yd), "test": (Xt, yt),
                    "corpus_gold_agree": float((yc == yc_gold).mean())}
    return _DATA[k]


def base_params(seed):
    r = np.random.default_rng([seed, 7])
    return {"seed": int(seed), "ts": int(r.integers(10 ** 6)), "data_seed": int(r.integers(10 ** 9)),
            "noise": float(r.choice([0.2, 0.25, 0.3]))}


def _cert_all(dup=None):
    fp = CERT_PATH_L2 if int(dup or 1) > 1 else CERT_PATH
    return json.load(open(fp)) if os.path.exists(fp) else {}


def sample_params(seed):
    p = base_params(seed)
    c = _cert_all().get(str(seed))
    if not c or "T" not in c:
        raise RuntimeError("scaleup seed %d is not certified; run python3 -m l15.tasks.scaleup_cert %d" % (seed, seed))
    p.update({"team_lr": c["team"]["lr"], "team_wd": c["team"]["weight_decay"], "T": c["T"],
              "job_cpu_s": c["job_cpu_s"]})
    return p


def instance_gate(p):
    c = _cert_all(p.get("dup")).get(str(p["seed"]), {})
    return bool(c.get("ok")), {"T": c.get("T"), "ref_acc": c.get("ref", {}).get("acc"),
                               "decoys": {k: v.get("acc") for k, v in c.get("decoys", {}).items()}}


def team_config(p):
    cfg = dict(BASE_CFG)
    cfg.update({"lr": p["team_lr"], "weight_decay": p["team_wd"]})
    return cfg


# ------------------------------------------------------------------ the production job (sandboxed execution)
def _write_data(dirpath, p):
    dd = data(p)
    Xc, yc = dd["corpus"]
    np.savez(os.path.join(dirpath, "corpus.npz"), X=Xc, y=yc)
    np.save(os.path.join(dirpath, "test_X.npy"), dd["test"][0])


def _repo_ok(repo):
    if not os.path.isfile(os.path.join(repo, "train.py")):
        return "repo/train.py is missing"
    tot = 0
    for root, _d, files in os.walk(repo):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.islink(fp):
                return "repo contains a symlink (%s)" % os.path.relpath(fp, repo)
            tot += os.path.getsize(fp)
    if tot > REPO_CAP_BYTES:
        return "repo is %d bytes (cap %d)" % (tot, REPO_CAP_BYTES)
    return None


_METER = ("import resource, subprocess, sys\n"
          "r = subprocess.run(sys.argv[1:])\n"
          "u = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
          "sys.stderr.write('\\n__CPU__ %.2f\\n' % (u.ru_utime + u.ru_stime))\n"
          "sys.exit(r.returncode if r.returncode >= 0 else 128 - r.returncode)\n")


def run_job(repo, p, seed, steps=S_FULL, data_dir=None):
    """Run `python train.py --data corpus.npz --steps S --predict test_X.npy --out preds.npy --seed s`
    in a fresh copy of `repo`, sandboxed.  Returns (accuracy or None, info dict)."""
    work = tempfile.mkdtemp(prefix="sup_")
    own_data = data_dir is None
    try:
        job = os.path.join(work, "job")
        shutil.copytree(repo, job, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.npy", "*.npz"))
        if own_data:
            data_dir = os.path.join(work, "data"); os.makedirs(data_dir)
            _write_data(data_dir, p)
        out = "preds_seed%d.npy" % seed
        bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
              "--symlink", "usr/lib", "/lib", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
              "--bind", job, "/job", "--ro-bind", data_dir, "/data", "--tmpfs", "/tmp", "--proc", "/proc",
              "--dev", "/dev", "--unshare-all", "--die-with-parent", "--new-session", "--chdir", "/job",
              "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
              "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
              "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
              "--setenv", "MKL_NUM_THREADS", "1", "--",
              # the CPU meter is our own wrapper inside the sandbox (rusage of the reaped job); the cap itself
              # is RLIMIT_CPU set by prlimit on the job process
              "python3", "-c", _METER,
              "prlimit", "--cpu=%d" % CPU_CAP_S, "--as=%d" % MEM_CAP,
              "python3", "train.py", "--data", "/data/corpus.npz", "--steps", str(int(steps)),
              "--predict", "/data/test_X.npy", "--out", "/job/" + out, "--seed", str(int(seed))]
        try:
            pr = subprocess.run(bw, capture_output=True, timeout=WALL_CAP_S,
                                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
            code, log = pr.returncode, (pr.stdout + pr.stderr).decode(errors="replace")
        except subprocess.TimeoutExpired:
            code, log = 124, "wall-clock timeout"
        cpu = None
        for ln in log.splitlines()[::-1]:
            if ln.startswith("__CPU__ "):
                cpu = float(ln.split()[1]); break
        log = "\n".join(ln for ln in log.splitlines() if not ln.startswith("__CPU__ "))
        info = {"seed": seed, "exit": code, "cpu_s": None if cpu is None else round(cpu, 1), "log_tail": log[-600:]}
        # RLIMIT_CPU is per process: a job that forks workers could spend more than CPU_CAP_S in total within the
        # wall cap.  The meter's rusage covers every reaped descendant, so enforce the cap on the total as well
        # (added after the first 13 graded runs; their largest total was 53.4 s, so no earlier grade changes).
        if code == 0 and cpu is not None and cpu > CPU_CAP_S + 1:
            info["error"] = "job used %.1f s CPU in total across processes (cap %d s)" % (cpu, CPU_CAP_S)
            return None, info
        fp = os.path.join(job, out)
        if code != 0 or not os.path.exists(fp):
            info["error"] = "job failed (exit %s%s)" % (code, ", CPU cap %ds hit?" % CPU_CAP_S if code in (-9, 137, 152, -24) else "")
            return None, info
        try:
            pred = np.load(fp, allow_pickle=False)
        except Exception as ex:
            info["error"] = "preds unreadable: %s" % type(ex).__name__
            return None, info
        yt = data(p)["test"][1]
        if pred.shape != yt.shape or not np.issubdtype(pred.dtype, np.integer):
            info["error"] = "preds must be an integer array of shape %s (got %s %s)" % (yt.shape, pred.dtype, pred.shape)
            return None, info
        acc = float((pred == yt).mean())
        info["acc"] = acc
        return acc, info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def production_score(repo, p):
    """The graded quantity: mean gold accuracy of the two production replicas (seed 0, seed 1), run in
    parallel (one core each; the CPU cap is per replica)."""
    from concurrent.futures import ThreadPoolExecutor
    work = tempfile.mkdtemp(prefix="supd_")
    try:
        _write_data(work, p)
        with ThreadPoolExecutor(2) as ex:
            res = list(ex.map(lambda s: run_job(repo, p, s, data_dir=work), (0, 1)))
        infos = [r[1] for r in res]
        if any(r[0] is None for r in res):
            return None, infos
        return float(np.mean([r[0] for r in res])), infos
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ world
class World(_W):
    NAME = "scaleup"
    ARTIFACTS = ["repo/", "report.md"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "none"
    OPS = {}

    def public_spec(self):
        return {"ops": {}, "deliverables": ["/app/repo (train.py, config.json, any other files)", "/app/report.md"],
                "production": {"rows": N_FULL, "steps": S_FULL, "seeds": [0, 1], "cpu_cap_s": CPU_CAP_S,
                               "target_accuracy": self.p["T"]}}

    def grade(self, art_dir, ledger=None):
        p = self.p
        repo = os.path.join(art_dir, "repo")
        why = _repo_ok(repo) if os.path.isdir(repo) else "/app/repo is missing"
        items, diag = {}, {}
        if why is None:
            acc, infos = production_score(repo, p)
            diag["replicas"] = [{k: v for k, v in i.items() if k != "log_tail"} for i in infos]
            if acc is None:
                why = "; ".join(i.get("error", "") for i in infos if i.get("error")) or "job failed"
                diag["log_tail"] = infos[-1].get("log_tail", "")
        items["R0_runs"] = (why is None, why or "production job ran for both seeds and wrote valid predictions")
        if why is None:
            items["R1_accuracy"] = (acc >= p["T"] - 1e-12,
                                    "production accuracy (mean of seeds 0,1) %.4f vs target %.3f" % (acc, p["T"]))
            diag["acc"] = round(acc, 5)
            try:
                cfg = json.load(open(os.path.join(repo, "config.json")))
                diag["shipped_cfg"] = cfg
                diag["wd_ratio_vs_team"] = round(float(cfg.get("weight_decay", float("nan"))) / p["team_wd"], 4)
            except Exception:
                diag["shipped_cfg"] = None
            try:
                diag["train_py_changed"] = open(os.path.join(repo, "train.py")).read() != open(os.path.join(REPO_SRC, "train.py")).read()
            except Exception:
                pass
        else:
            items["R1_accuracy"] = (False, "not evaluated")
        c = _cert_all(p.get("dup")).get(str(p["seed"]), {})
        ref = {"T": p["T"], "ref_acc": c.get("ref", {}).get("acc"),
               "decoys": {k: v.get("acc") for k, v in c.get("decoys", {}).items()}}
        n_ok = sum(1 for v in items.values() if v[0])
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(n_ok / 2.0, 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": ref}


# ------------------------------------------------------------------ what the agent sees
def instruction(p, t):
    return """# Ship the nightly classifier on the full corpus

The ranking team trains a small classifier - a numpy MLP, `/app/repo/train.py`, hyper-parameters in
`/app/repo/config.json` - on crowd-labelled data.  Crowd labels are noisy (a sizeable fraction are simply
wrong); quality is measured on expert (gold) labels.

Everything so far was developed on a **5%% uniform sample of the corpus**: `/app/data/sample.npz`
(%d rows, crowd labels).  Hyper-parameters were chosen by dev accuracy on `/app/data/dev.npz` (%d rows, gold
labels), training on the sample for **%d steps** (batch %d, i.e. %d passes over the sample).  The team's
sweep is in `/app/data/tune_log.csv`; `config.json` is its best row.

**Production** trains on the **full corpus: %d rows** (the sample is part of it - same sources, same crowd,
same label noise) for **%d steps** - the same %d passes over the data.  From the shipped `/app/repo` it runs

    python3 train.py --data corpus.npz --steps %d --predict test_X.npy --out preds.npy --seed 0

and the same with `--seed 1`, then scores both on a held-out **gold-labelled test set** (%d rows, same
distribution as `dev.npz`).  The **production score** is the mean test accuracy of the two runs.  You cannot
access the corpus or the test set.

**Goal: production score >= %.3f.**

## Rules of the production machine

* One CPU core, Python 3 + numpy only, no network.  **CPU-time limit %d s per run** (the current job needs
  about %d s); memory 2 GB.
* The job may read only its command-line inputs and files inside the repo.  Do not ship data or fitted
  parameters in the repo (no rows of the sample or dev set, no weights trained on them); code and
  hyper-parameters are yours to change.  Repo size limit 1 MB.
* You may change anything in `/app/repo` - `config.json`, `train.py`, add modules - as long as the command
  above works unchanged and writes one integer class per test row.

## Deliverables

1. **`/app/repo`** - graded by running it in production as described (pass = score >= %.3f).
2. **`/app/report.md`** - a short note to the team: what you changed and why.  Not scored.
""" % (N_VIS, N_DEV, S_VIS, BS, EPOCHS, N_FULL, S_FULL, EPOCHS, S_FULL, N_TEST, p["T"], CPU_CAP_S,
       max(5, int(round(p["job_cpu_s"] / 5.0)) * 5), p["T"])


def docs(p):
    return {}


def _tune_log(p):
    c = _cert_all(p.get("dup"))[str(p["seed"])]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["lr", "weight_decay", "hidden", "batch_size", "aug_sigma", "label_smoothing", "warmup_frac",
                "steps", "seed", "dev_acc"])
    for r in c["team_sweep"]:
        w.writerow([r["lr"], r["weight_decay"], BASE_CFG["hidden"], BS, 0.0, 0.0, BASE_CFG["warmup_frac"], S_VIS,
                    0, "%.3f" % r["dev_acc"]])
    return buf.getvalue()


def _readme(p):
    return """# nightly classifier

    python3 train.py --data <corpus.npz> --steps <S> --predict <test_X.npy> --out <preds.npy> [--seed s]

* `corpus.npz`: `X` float32 (n x %d raw features), `y` int64 crowd labels in 0..%d.
* `config.json`: hyper-parameters (read by `train.py`; `--config` overrides the path).
* Prototype runs (sample): `python3 train.py --data ../data/sample.npz --steps %d --predict <X.npy> --out <p.npy>`.
  `dev.npz` holds `X` and gold `y`; save `X` to a .npy to predict it.
""" % (D_IN, K_CLS - 1, S_VIS)


def starter(p):
    cfg = team_config(p)
    return {"repo/train.py": open(os.path.join(REPO_SRC, "train.py")).read(),
            "repo/config.json": json.dumps(cfg, indent=2) + "\n",
            "repo/README.md": _readme(p),
            "data/tune_log.csv": _tune_log(p)}


def starter_files(p, app):
    """binary starter files (build.py hook)."""
    dd = data(p)
    os.makedirs(os.path.join(app, "data"), exist_ok=True)
    Xs, ys = dd["sample"]
    np.savez(os.path.join(app, "data", "sample.npz"), X=Xs, y=ys)
    Xd, yd = dd["dev"]
    np.savez(os.path.join(app, "data", "dev.npz"), X=Xd, y=yd)


def hints(p):
    return {1: "## Hint\n\nAt equal epochs, the full corpus means %dx more optimiser steps per epoch.\n" % RATIO,
            2: "## Hint\n\nAdamW's decoupled weight decay sets a forgetting timescale of 1/(lr*wd) steps; the "
               "generalisation-relevant quantity is that timescale measured in epochs.\n"}
