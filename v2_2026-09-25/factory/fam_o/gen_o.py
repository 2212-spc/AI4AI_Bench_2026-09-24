"""Family O generator: exact-equivalence acceleration of a spec-as-code packer.

Capability probed: extract an exact behavioural contract from code whose rules interact, then redesign the
algorithm (O(n*open) -> O(n log n)) without losing a single rule.  Truth is outsourced to the reference
implementation, so the verifier needs no rubric and no statistics.

Gates (all executed, see certificate.json):
  G-truth      expected outputs are produced by the specification implementation itself
  G-solvable   a blind reference solution (fast_pack.py) matches on every hidden case and meets the time gate
  G-shortcut   15 plausible-but-wrong fast packers are each falsified by at least one hidden case (mined)
  G-timing     the specification implementation itself misses the time gate by a wide margin
  G-verifier   nop / partial / off-by-one-rule submissions all score 0 (selfcheck.py)
usage: gen_o.py <instance_seed> <out_task_dir>
"""
import json, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "core"))
import ref_pack, fast_pack, candidates                      # noqa: E402
from gen_inputs import make_lengths, PROFILES               # noqa: E402
import harbor                                               # noqa: E402

AGENT_TIMEOUT = 5400


def params_for(seed):
    r = np.random.default_rng(seed)
    return {"capacity": int(r.choice([2048, 4096, 8192])),
            "close_below": int(r.choice([4, 8, 16])),
            "max_open": int(r.choice([5000, 6000, 7000])),
            "flush_every": int(r.choice([48, 64, 96, 128]))}


def _try(params, prof, n, s, found, cases):
    L = make_lengths(prof, n, params["capacity"], s)
    ref = ref_pack.pack(L, params)
    hit = [v for v in candidates.VARIANTS if v not in found and candidates.candidate(v)(L, params) != ref]
    if hit:
        cases.append({"kind": "mined", "profile": prof, "n": n, "seed": s, "falsifies": hit})
        for v in hit:
            found[v] = len(cases) - 1
    return hit


def mine(params, rng, budget=120):
    """For every wrong-candidate packer find a small input on which it differs from the spec.
    Phase 1 uses cheap inputs; phase 2 raises n until the open-bin memory guard can bind."""
    found, cases = {}, []
    for _ in range(budget):
        _try(params, str(rng.choice(PROFILES)), int(rng.integers(60, 700)), int(rng.integers(0, 10 ** 6)), found, cases)
        if len(found) == len(candidates.VARIANTS):
            return cases, found
    for n in (12000, 30000, 60000):
        for prof in ("bimodal", "wide", "oversize"):
            _try(params, prof, n, int(rng.integers(0, 10 ** 6)), found, cases)
            if len(found) == len(candidates.VARIANTS):
                return cases, found
    return cases, found


def edge_cases(params):
    C = params["capacity"]
    return [("empty", []), ("one_oversized", [C + 1]), ("all_zero", [0] * 40),
            ("exact_cap", [C, C, 1, C - 1, 0, C + 5, 2]),
            ("zero_after_close", [C, 0, 0, C - 1, 0]),
            ("oversize_burst", [C + 1] * 7 + [1, 2, 3] + [C + 9] * 3)]


def main():
    seed = int(sys.argv[1]); out = sys.argv[2]
    cache = os.environ.get("CACHE", "/tmp/bench2/cache_o_%d" % seed)
    os.makedirs(cache, exist_ok=True)
    rng = np.random.default_rng(seed)
    params = params_for(seed)
    t0 = time.time()

    # ---- stage 1: mine falsifying inputs for every wrong-candidate packer (cached) ----
    if os.path.exists(cache + "/mined.json"):
        d = json.load(open(cache + "/mined.json")); mined, found = d["mined"], d["found"]
    else:
        mined, found = mine(params, rng)
        json.dump({"mined": mined, "found": found}, open(cache + "/mined.json", "w"))
        print("stage1 mined %d cases, %d/%d variants, %.0fs" % (len(mined), len(found), len(candidates.VARIANTS),
                                                                time.time() - t0), file=sys.stderr)
    unfalsified = sorted(set(candidates.VARIANTS) - set(found))
    rng = np.random.default_rng([seed, 2])          # suite seeds independent of how long mining ran

    suite = []
    for prof in PROFILES:
        for k in range(3):
            suite.append({"kind": "base", "profile": prof, "n": int(rng.integers(400, 3000)),
                          "seed": int(rng.integers(0, 10 ** 6)), "falsifies": []})
    for prof in ("bimodal", "wide", "oversize"):          # large enough for the memory guard to bind
        suite.append({"kind": "base_large", "profile": prof, "n": 40000,
                      "seed": int(rng.integers(0, 10 ** 6)), "falsifies": []})
    n_base = len(suite)
    suite += mined
    for name, L in edge_cases(params):
        suite.append({"kind": "edge", "name": name, "lengths": L, "falsifies": []})

    big = {"kind": "timing", "profile": "bimodal", "n": 400000, "seed": int(rng.integers(0, 10 ** 6))}

    def inputs_of(c):
        return c["lengths"] if "lengths" in c else make_lengths(c["profile"], c["n"], params["capacity"], c["seed"])

    # ---- stage 2: expected outputs of the small/medium suite (cached) ----
    if os.path.exists(cache + "/expected_small.npz"):
        z = np.load(cache + "/expected_small.npz"); expected = {k: z[k] for k in z.files}
    else:
        expected = {}
        for i, c in enumerate(suite):
            expected["c%03d" % i] = np.asarray(ref_pack.pack(inputs_of(c), params), dtype=np.int64)
            print("stage2 %d/%d" % (i + 1, len(suite)), file=sys.stderr)
        np.savez_compressed(cache + "/expected_small.npz", **expected)

    # ---- stage 3: the 400k timing shard (cached; this is the slow part) ----
    Lbig = inputs_of(big)
    if os.path.exists(cache + "/big.npz"):
        d = np.load(cache + "/big.npz"); exp_big = d["y"].tolist(); ref_big_s = float(d["s"])
    else:
        t = time.time(); exp_big = ref_pack.pack(Lbig, params); ref_big_s = time.time() - t
        np.savez_compressed(cache + "/big.npz", y=np.asarray(exp_big, dtype=np.int64), s=np.array(ref_big_s))
        print("stage3 spec impl on 400k: %.1fs" % ref_big_s, file=sys.stderr)
    expected["big"] = np.asarray(exp_big, dtype=np.int64)

    # ---- stage 4: blind reference solution must match everywhere and be fast ----
    t = time.time(); got_big = fast_pack.pack(Lbig, params); fast_big_s = time.time() - t
    bad = [i for i, c in enumerate(suite) if fast_pack.pack(inputs_of(c), params) != expected["c%03d" % i].tolist()]
    ok_sol = (not bad) and got_big == exp_big

    eqv = {v: all(candidates.candidate(v)(inputs_of(c), params) == expected["c%03d" % i].tolist()
                  for i, c in enumerate(suite)) for v in candidates.EQUIVALENT}
    calib = calibrate()
    limit = 12.0                                 # seconds on a host with this calibration time
    cert = {"family": "O", "instance_seed": seed, "params": params,
            "n_cases": len(suite), "n_mined": len(mined), "mine_seconds": round(time.time() - t0, 1),
            "candidates_falsified": {v: "c%03d" % (n_base + i) for v, i in found.items()},
            "candidates_unfalsified": unfalsified,
            "G_no_false_rejection_equivalent_variants": eqv,
            "G_solvable_blind_reference": bool(ok_sol), "blind_reference_mismatch_cases": bad,
            "G_timing_limit_s": limit, "G_timing_calibration_loop_s": calib,
            "spec_impl_big_s": round(ref_big_s, 2), "blind_reference_big_s": round(fast_big_s, 2),
            "timing_ratio": round(ref_big_s / max(fast_big_s, 1e-6), 1),
            "G_timing_spec_impl_fails": bool(ref_big_s > 2.0 * limit),
            "G_timing_reference_passes": bool(fast_big_s < 0.35 * limit)}
    cert["accepted"] = bool(cert["G_solvable_blind_reference"] and cert["G_timing_spec_impl_fails"]
                            and cert["G_timing_reference_passes"] and not unfalsified and all(eqv.values()))
    write_task(out, seed, params, suite, big, expected, limit, cert, calib)
    print(json.dumps(cert, indent=1))


def calibrate():
    """CPU seconds for a fixed pure-python loop; the verifier repeats it to rescale the time limit.
    process_time is used so that load on the host does not inflate the measurement."""
    t = time.process_time()
    x = 0
    for i in range(20_000_000):
        x += i % 7
    return round(time.process_time() - t, 3)


INSTR = """# Make the training-data packer fast without changing a single packed batch

`/app/pack_ref.py` is the reference sequence packer used by MiniStack's data loader. It is the
specification: whatever `pack(lengths, params)` returns for a given input is by definition the correct
packing. `/app/params.json` holds the parameters of this deployment. `/app/sample/` contains one small
input and the packing the reference produces for it, so you can sanity-check your work.

The reference is too slow: at production scale (hundreds of thousands of sequences per shard) it takes
minutes per shard and has become the bottleneck of the data pipeline.

Write `/app/solution.py` exposing

    def pack(lengths: list[int], params: dict) -> list[int]

that returns **exactly** the same bin id for every sequence as `pack_ref.py` does, for every input, and
that is fast enough to pack a 400,000-sequence shard well inside the time limit below.

Rules:
* `/app/solution.py` must be self-contained: it may import the Python standard library and numpy, but it
  must not import, exec, read or otherwise call `pack_ref.py` (a copy of it does not help - it is the slow
  implementation you are replacing).
* Your `pack` will be called in a fresh process, several times, with inputs you have not seen, including
  inputs specifically built around the corner cases of the reference rules.
* Grading is all-or-nothing: every hidden input must match the reference output element for element,
  and the 400,000-sequence shard must be packed in under {limit:.0f} seconds (measured on a machine
  calibrated against the authoring host; the reference implementation needs roughly {refs:.0f} s there).

Deliverable: `/app/solution.py`. Also write `/app/NOTES.md` listing the behavioural details of the
reference that your implementation had to preserve.
"""


def write_task(out, seed, params, suite, big, expected, limit, cert, calib):
    if os.path.exists(out):
        shutil.rmtree(out)
    app = out + "/environment/app"
    os.makedirs(app + "/sample")
    shutil.copy(os.path.join(HERE, "ref_pack.py"), app + "/pack_ref.py")
    json.dump(params, open(app + "/params.json", "w"), indent=1)
    sl = make_lengths("mix", 120, params["capacity"], 999_001)
    json.dump({"params": params, "lengths": sl, "expected_bin_of_item": ref_pack.pack(sl, params)},
              open(app + "/sample/sample_00.json", "w"))
    harbor.w(out + "/environment/Dockerfile",
             "FROM python:3.11-slim\nRUN pip install --no-cache-dir %s\nWORKDIR /app\nCOPY app/ /app/\n" % harbor.NUMPY)

    harbor.w(out + "/instruction.md",
             INSTR.format(limit=limit, refs=cert["spec_impl_big_s"])
             + harbor.SUFFIX_T.format(t=AGENT_TIMEOUT))
    harbor.w(out + "/task.toml", harbor.task_toml(["/app/solution.py", "/app/NOTES.md"], "O",
                                                  ["algorithm-equivalence", "spec-extraction", "complexity"],
                                                  AGENT_TIMEOUT, 2.5, verifier_timeout=900))
    # ---- tests (separate verifier, no network) ----
    os.makedirs(out + "/tests")
    shutil.copy(os.path.join(HERE, "gen_inputs.py"), out + "/tests/gen_inputs.py")
    shutil.copy(os.path.join(HERE, "verify_o.py"), out + "/tests/verify_o.py")
    json.dump({"params": params, "suite": suite, "big": big, "limit_s": limit,
               "calib_ref_s": calib}, open(out + "/tests/spec.json", "w"))
    np.savez_compressed(out + "/tests/expected.npz", **expected)
    harbor.w(out + "/tests/test.sh", harbor.TEST_SH.format(verify="verify_o.py"), 0o755)
    harbor.w(out + "/tests/Dockerfile",
             "FROM python:3.11-slim\nRUN pip install --no-cache-dir %s\nRUN mkdir -p /app /logs/verifier /tests\n"
             "COPY verify_o.py gen_inputs.py spec.json expected.npz test.sh /tests/\n" % harbor.NUMPY)
    # ---- solution ----
    os.makedirs(out + "/solution")
    shutil.copy(os.path.join(HERE, "fast_pack.py"), out + "/solution/solution.py")
    harbor.w(out + "/solution/solve.sh",
             "#!/bin/bash\nset -euo pipefail\ncp /solution/solution.py /app/solution.py\n"
             "cp /solution/NOTES.md /app/NOTES.md\n", 0o755)
    harbor.w(out + "/solution/NOTES.md", NOTES)
    # ---- authoring evidence ----
    os.makedirs(out + "/authoring")
    json.dump(cert, open(out + "/authoring/certificate.json", "w"), indent=1)
    harbor.w(out + "/authoring/provenance.md", PROV.format(seed=seed, canary=harbor.CANARY, **cert))
    harbor.w(out + "/README.md", README.format(seed=seed, **cert))


NOTES = """Behavioural details of `pack_ref.py` that the fast implementation must preserve:
1. first fit in **bin creation order**, not best fit (leftmost open bin whose remaining capacity >= L);
2. a sequence with L > capacity consumes a bin id and that bin is never open;
3. the closure test is `remaining < close_below`, strictly;
4. the memory guard fires only when the number of open bins is strictly greater than max_open, and it
   closes the **fullest** open bin (smallest remaining), ties broken by smallest bin id;
5. rule order after a placement is: closure, then memory guard, then latency guard;
6. the latency guard uses the index of the sequence in the input (oversized sequences advance that index
   but are skipped by the guard because the loop body returns early for them);
7. the latency guard needs at least two open bins and closes the oldest one;
8. zero-length sequences take the leftmost open bin and open a new bin when none is open.
"""

PROV = """# Provenance - family O instance (seed {seed})

Truth source: the specification implementation `pack_ref.py` executed at authoring time; no human wrote any
expected output. Hidden expected outputs are in `tests/expected.npz`.

Seed of the design (what was borrowed, and what was changed): sequence packing in LLM data pipelines
(first-fit/best-fit bin packing over token lengths, as used in Megatron/HF packing utilities) supplies the
*mechanism*. Everything a public implementation would share was then transformed: three interacting
closure rules were added (capacity threshold, open-bin memory guard, periodic latency flush), the flush is
driven by the input index rather than the placement count, and the numeric parameters are drawn per
instance. No public artifact asks for bit-exact reproduction of such a packer under a complexity budget.

Difficulty evidence: {n_mined} plausible fast packers (one rule dropped or reordered each) were each
falsified by a mined hidden input; the specification implementation itself needs {spec_impl_big_s}s on the
timing shard versus {blind_reference_big_s}s for the blind reference solution (ratio {timing_ratio}x).

Canary: {canary}
"""

README = """# Family O instance (seed {seed}) - exact-equivalence acceleration of a spec-as-code packer

Not shown to the agent.

## Difficulty
The agent must turn an O(n * open_bins) first-fit loop into an O(n log n) structure while preserving nine
interacting behaviours, three of which are invisible on ordinary inputs (oversized sequences still consume
a bin id; the latency guard is driven by the input index, so oversized sequences advance it but never
trigger it; the memory guard closes the *fullest* bin with ties broken by smallest id). A leftmost-index
query with deletions needs a segment tree, and the memory guard needs a second order on the same set, so
the obvious "heap of open bins" answer is wrong twice over. {n_mined} single-rule variants of the reference
were each falsified by a mined input that is part of the hidden suite, so dropping any one rule scores 0.
Timing: specification implementation {spec_impl_big_s}s vs blind reference {blind_reference_big_s}s on the
400k shard (ratio {timing_ratio}x); the limit is {G_timing_limit_s}s at calibration factor 1.

## Reference solution
`solution/solution.py`: two segment trees over bin creation index (max-remaining for the leftmost-fit
descent, min-remaining for the memory guard) plus a monotone pointer for the oldest open bin. Written from
the specification only.

## Verification
`tests/verify_o.py` imports `/app/solution.py` in a fresh process and compares its output element for
element against `tests/expected.npz`, which was produced by executing `pack_ref.py` at authoring time on
{n_cases} hidden inputs (8 length profiles x 3 seeds, {n_mined} mined corner-case inputs, 6 hand-built edge
inputs) plus one 400k timing shard. Reward 1 iff every case matches and the timing shard is packed inside
the calibrated limit. No rubric, no statistics, no judgement.
"""

if __name__ == "__main__":
    main()
