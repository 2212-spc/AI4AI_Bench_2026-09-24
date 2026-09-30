"""Export generated instances to Harbor task directories.

  python3 export.py e1 <instance_id> [<instance_id> ...]
  python3 export.py a1 <instance_id> [...]
Output: /sessions/.../outputs/harbor_tasks/<instance_id>/ (OUT env overrides).

Layout (Harbor / TBS conventions; items marked [unverified] could not be checked against a live harbor install):
  task.toml, instruction.md, README.md,
  environment/{Dockerfile, docker-compose.yaml, app/..., lab/...}   (E1: agent image + lab sidecar)
  solution/{solve.sh, ...}                                         (blind reference solution, computes its answer)
  tests/{Dockerfile, test.sh, verify*.py, <hidden truth>}          (separate verifier, reads artifacts only)
  authoring/{certificate.json, provenance.md}                      (generator evidence, not a runtime dependency)"""
import json, os, shutil, subprocess, sys

BENCH = "/tmp/bench"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", "/sessions/dazzling-admiring-lovelace/mnt/outputs/harbor_tasks")
AGENT_TIMEOUT = 3600
SUFFIX = ("\n\nYou have %d seconds to complete this task. Do not cheat by using online solutions or hints specific to "
          "this task.\n" % AGENT_TIMEOUT)
NUMPY_PIN = "numpy==2.2.6"


def w(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    if mode: os.chmod(path, mode)


TOML_E1 = """version = "1.0"
artifacts = ["/app/decision.json", "/app/REPORT.md"]

[metadata]
author_name = "independent-benchmark (generated)"
difficulty = "hard"
category = "ai4ai"
tags = ["experiment-design", "interaction-effects", "budgeted-lab", "generated", "{family}"]
expert_time_estimate_hours = 1.5

[agent]
timeout_sec = {agent_timeout}.0

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"
cpus = 1
memory_mb = 512

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 2048
storage_mb = 4096
"""

COMPOSE_E1 = """# Agent container `main` + budgeted lab sidecar `lab` on an internal network (no internet for either).
# [unverified] service name `main` and how harbor merges this file with its own compose must be checked on a live install.
services:
  main:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - LAB_URL=http://lab:18777/
    depends_on:
      - lab
    networks: [labnet]
  lab:
    build:
      context: ./lab
    environment:
      - LAB_BUDGET={budget}
    expose:
      - "18777"
    networks: [labnet]
networks:
  labnet:
    internal: true
"""

DOCKER_AGENT_E1 = """FROM python:3.11-slim
RUN pip install --no-cache-dir {numpy}
WORKDIR /app
COPY app/ /app/
RUN chmod +x /app/lab
ENV LAB_URL=http://lab:18777/
"""

DOCKER_LAB_E1 = """FROM python:3.11-slim
RUN pip install --no-cache-dir {numpy}
WORKDIR /srv/lab
COPY engine.py e1_world.py lab_service.py world.json /srv/lab/
RUN mkdir -p /var/lab
EXPOSE 18777
CMD ["python3", "/srv/lab/lab_service.py"]
"""

DOCKER_TESTS_E1 = """FROM python:3.11-slim
RUN mkdir -p /app /logs/verifier /tests
COPY verify_e1.py scoring.json test.sh /tests/
"""

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/{verify} || echo 0 > /logs/verifier/reward.txt
"""

SOLVE_E1 = """#!/bin/bash
set -euo pipefail
# Blind reference solution: factorial screen of c1-c5 on common seeds, replicate the leaders, ship the best.
python3 /solution/expert_e1.py
"""


def cli_for_harbor(src):
    s = open(src).read()
    s = s.replace('TOKEN = open(os.path.join(HERE, ".lab_token")).read().strip()',
                  'TOKEN = open(os.path.join(HERE, ".lab_token")).read().strip() if os.path.exists(os.path.join(HERE, ".lab_token")) else ""')
    s = s.replace('URL = os.environ.get("LAB_URL", "http://127.0.0.1:18777/")', 'URL = os.environ.get("LAB_URL", "http://lab:18777/")')
    return s


def export_e1(iid):
    src = f"{BENCH}/instances/e1/{iid}"; dst = f"{OUT}/{iid}"
    if os.path.exists(dst): shutil.rmtree(dst)
    sc = json.load(open(src + "/hidden/scoring.json")); cert = json.load(open(src + "/hidden/certificate.json"))
    instr = open(src + "/instruction.md").read().rstrip()
    instr = instr.replace("`/app/lab` (run `/app/lab --help`)", "`/app/lab` (run `python3 /app/lab --help`)")
    w(dst + "/instruction.md", instr + SUFFIX)
    w(dst + "/task.toml", TOML_E1.format(family="E1", agent_timeout=AGENT_TIMEOUT))
    # environment: agent image
    shutil.copytree(src + "/public", dst + "/environment/app", ignore=shutil.ignore_patterns("lab", ".lab_token"))
    w(dst + "/environment/app/lab", cli_for_harbor(f"{BENCH}/lab/lab_cli.py"), 0o755)
    w(dst + "/environment/Dockerfile", DOCKER_AGENT_E1.format(numpy=NUMPY_PIN))
    w(dst + "/environment/docker-compose.yaml", COMPOSE_E1.format(budget=sc["budget"]))
    # environment: lab sidecar (hidden world lives only here)
    for sub in ["environment/lab", "solution", "tests", "authoring"]: os.makedirs(f"{dst}/{sub}", exist_ok=True)
    shutil.copy(f"{BENCH}/lab/engine.py", dst + "/environment/lab/")
    w(dst + "/environment/lab/e1_world.py", open(f"{BENCH}/gen/e1_world.py").read().replace('import sys; sys.path.insert(0, "/tmp/bench/lab")\n', ""))
    shutil.copy(HERE + "/lab_service.py", dst + "/environment/lab/")
    shutil.copy(src + "/hidden/world.json", dst + "/environment/lab/world.json")
    w(dst + "/environment/lab/Dockerfile", DOCKER_LAB_E1.format(numpy=NUMPY_PIN))
    # solution
    w(dst + "/solution/solve.sh", SOLVE_E1, 0o755); shutil.copy(HERE + "/expert_e1.py", dst + "/solution/")
    # tests (separate verifier)
    shutil.copy(HERE + "/verify_e1.py", dst + "/tests/"); shutil.copy(src + "/hidden/scoring.json", dst + "/tests/")
    w(dst + "/tests/test.sh", TEST_SH.format(verify="verify_e1.py"), 0o755)
    w(dst + "/tests/Dockerfile", DOCKER_TESTS_E1)
    # authoring evidence
    os.makedirs(dst + "/authoring", exist_ok=True)
    shutil.copy(src + "/hidden/certificate.json", dst + "/authoring/certificate.json")
    pol = cert["policies"]
    rows = "\n".join(f"| {k} | {v['pass']:.3f} | {', '.join(f'{a} ({b:.2f})' for a, b in v['typical_end'][:2])} |" for k, v in pol.items())
    w(dst + "/README.md", f"""# {iid} (family E1: budgeted recipe-change triage with interacting changes)

Not shown to the agent.

## Difficulty
Six candidate recipe changes for a small MLP trainer interact: some pairs diverge together, some changes only help
in combination, one change (c6) is a pure RNG-stream decoy. The budget ({sc['budget']} lab runs) is far below what
naive per-arm replication needs, the teammate note is an honest single-seed one-factor-at-a-time ablation whose plan
is wrong, and the textbook path (single ablations, then fix instabilities one knob at a time) ends in a local optimum.
The optimum is {'+'.join(cert['opt'])} (mean val MSE {cert['opt_mu']:.5f}); accept set within {100*sc['delta_rel']:.1f}%:
{', '.join(cert['accept_set'])}.

Simulated policies against the truth table (pass rate, typical end states):

| policy | pass | typical end |
|---|---|---|
{rows}

## Reference solution
`solution/expert_e1.py`: screens all 32 subsets of c1-c5 on common seeds (c6 dropped by reasoning: it only changes
the minibatch RNG stream), replicates the leaders on fresh common seeds with the remaining budget, ships the best
and predicts its mean. It uses only public information and the lab CLI. Certified pass rate
{pol['expert_prune_c6']['pass']:.3f} (simulation over the truth table).

## Verification
`tests/verify_e1.py` reads only `/app/decision.json` (artifact) and the baked-in truth `tests/scoring.json`
(64 subsets x {cert['n_truth_seeds']} seeds, common random numbers, computed by executing the trainer). Reward 1 iff the
shipped subset's regret <= {100*sc['delta_rel']:.1f}% and the predicted val MSE is within {100*sc['pred_tol']:.0f}% of its true mean.
The accept boundary sits in a gap of the sorted regrets ({cert['boundary_clearance_pairedSE']:.1f} paired SE clearance).
The budget is enforced by the lab sidecar (HTTP 429), whose ledger is not visible to the agent.
Evidence: `authoring/certificate.json` (C1 truth resolution, C2 structure, C3 blind solvability, C4 no shortcut, C5 estimation).
""")
    w(dst + "/authoring/provenance.md", f"""# Provenance
Generated by gen/gen_e1.py from world parameters `environment/lab/world.json` (seeded, fully reproducible);
truth table by gen/e1_truth.py; world selected by the adversarial search gen/e1_search.py when applicable.
Teammate note seed: {cert.get('teammate_seed')}. Generator code is not a runtime dependency.
""")
    return dst


TOML_A1 = """version = "1.0"
artifacts = ["/app/minilab", "/app/FIX_NOTES.md"]

[metadata]
author_name = "independent-benchmark (generated)"
difficulty = "medium"
category = "ai4ai"
tags = ["debugging", "training-loop", "masked-bugs", "generated", "A1"]
expert_time_estimate_hours = 1.0

[agent]
timeout_sec = {agent_timeout}.0

[verifier]
timeout_sec = 900.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"
cpus = 2
memory_mb = 2048

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 2048
storage_mb = 4096
network_mode = "no-network"
"""


def export_a1(iid):
    src = f"{BENCH}/instances/a1/{iid}"; dst = f"{OUT}/{iid}"
    if os.path.exists(dst): shutil.rmtree(dst)
    w(dst + "/instruction.md", open(src + "/instruction.md").read().rstrip() + SUFFIX)
    w(dst + "/task.toml", TOML_A1.format(agent_timeout=AGENT_TIMEOUT))
    shutil.copytree(src + "/public", dst + "/environment/app")
    w(dst + "/environment/Dockerfile", f"FROM python:3.11-slim\nRUN pip install --no-cache-dir {NUMPY_PIN}\nWORKDIR /app\nCOPY app/ /app/\n")
    # solution = patch from public to the generator's oracle variant (applied, not echoed answers)
    os.makedirs(dst + "/solution", exist_ok=True)
    diff = subprocess.run(["diff", "-ruN", "public/minilab", "variants/oracle/minilab"], cwd=src, capture_output=True, text=True).stdout
    diff = diff.replace("--- public/minilab", "--- a/minilab").replace("+++ variants/oracle/minilab", "+++ b/minilab")
    w(dst + "/solution/fix.patch", diff)
    w(dst + "/solution/solve.sh", "#!/bin/bash\nset -euo pipefail\ncd /app\npatch -p1 < /solution/fix.patch\n"
      "printf '# Fix notes\\n\\nSee /solution/fix.patch (reference fix).\\n' > /app/FIX_NOTES.md\n", 0o755)
    os.makedirs(dst + "/tests", exist_ok=True); shutil.copy(f"{BENCH}/gen/a1_verify_core.py", dst + "/tests/")
    shutil.copy(src + "/hidden/verify_spec.json", dst + "/tests/")
    w(dst + "/tests/verify_a1.py", """import json, os, sys
sys.path.insert(0, "/tests")
import a1_verify_core as V
d = json.load(open(os.environ.get("SPEC", "/tests/verify_spec.json")))
ok, rep = V.evaluate(os.environ.get("APP", "/app"), d["spec"], V.teacher_from(d["teacher"]), verbose=False)
rd = os.environ.get("REWARD_DIR", "/logs/verifier"); os.makedirs(rd, exist_ok=True)
open(os.path.join(rd, "reward.txt"), "w").write("%d\\n" % int(ok)); print(json.dumps(rep)[:2000])
""")
    w(dst + "/tests/test.sh", TEST_SH.format(verify="verify_a1.py"), 0o755)
    w(dst + "/tests/Dockerfile", f"FROM python:3.11-slim\nRUN pip install --no-cache-dir {NUMPY_PIN}\nRUN mkdir -p /app /logs/verifier /tests\n"
      "COPY a1_verify_core.py verify_a1.py verify_spec.json test.sh /tests/\n")
    os.makedirs(dst + "/authoring", exist_ok=True); shutil.copy(src + "/hidden/certificate.json", dst + "/authoring/")
    w(dst + "/README.md", f"""# {iid} (family A1: masked convention-conflict regression)

Not shown to the agent.

## Difficulty
Two convention bugs in the v2 trainer partially compensate each other (masking): fixing only one makes results worse or
diverge, and a learning-rate rescale can match the canonical number while failing held-out configurations.
Empirically this family is saturated by frontier agents (4/4 passes) and is kept as a calibrated negative control.

## Reference solution
`solution/fix.patch` restores the reference semantics described in the README (applied by solve.sh).

## Verification
The verifier imports the agent's `minilab.trainer.train` (the artifact), trains on the canonical and held-out
configurations with fresh data from the teacher and compares mean held-out MSE with reference statistics computed by
the generator's reference implementation (`authoring/certificate.json` lists the near-miss variants that must fail:
single fixes, lr-compensation hacks, decoy edits).
""")
    return dst


if __name__ == "__main__":
    fam = sys.argv[1]
    for iid in sys.argv[2:]:
        print("exported", (export_e1 if fam == "e1" else export_a1)(iid))
