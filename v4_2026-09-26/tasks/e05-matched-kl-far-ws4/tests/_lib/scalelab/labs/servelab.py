"""ServeLab: a simulated *inference serving* stack (batching, quantization, speculative decoding,
queueing, test-time compute).

Cost unit: **accelerator-seconds**.  A request is charged for the accelerator time it actually occupies,
so a long high-batch benchmark costs far more than a short one and a quality evaluation costs what its
decoding costs.  Budget therefore buys very different evidence depending on the design.

Services (`svc`)
  bench    steady-state decode benchmark: throughput, per-token latency, KV footprint, acceptance
  quality  accuracy of the served configuration on a fixed held-out set
  load     closed-loop load test at an arrival rate: utilisation, p50 / p99 end-to-end latency
  ttc      test-time compute: coverage and selected accuracy at k samples

Mechanism cards implemented here: S1 (roofline / critical batch), S2 (quantization as an effective-
parameter reduction), S3 (speculative decoding with positional acceptance decay), S4 (M/G/1 queueing
where utilisation is *not* a knob), S5 (KV-cache capacity and the batch ceiling), S6 (test-time compute:
coverage versus selection, verifier precision ceiling).
"""
import hashlib, math
import numpy as np

NAME = "servelab"
COST_UNIT = "accelerator-seconds"
COST_TEXT = "accelerator-seconds = measured wall time x replicas (a refused request is not charged)"
EXTRA_FIELDS = ()

BASE = dict(
    # model / hardware
    N=7.0e9, L=32, H_kv=8, d_head=128, kv_bytes=2.0, act_bytes=2.0e9,
    P_peak=4.0e14, BW=3.35e12, M_gpu=80.0e9, mem_util=0.90, eff=0.70,
    prefill_flop_eff=0.55,
    # S2 quantization
    quant_g=None,               # bits -> g_x in N_eff = N * (1 - exp(-bits/g))
    A_loss=1.9, alpha=0.076, B_loss=0.0, beta=0.10, D_tok=2.0e12, E_loss=1.62,
    acc_k=2.6, acc_l0=1.95, acc_max=0.92, delta_ptq=None,   # bits -> extra loss
    # S3 speculative decoding
    drafts=None,                # name -> {"a": acceptance at position 1, "c": relative cost, "rho": positional decay}
    # S4 queueing
    cs2=1.0, gen_tok=256.0, prompt_tok=512.0,
    # S6 test-time compute
    tt_alpha=0.6, tt_beta=1.2, tt_mode=0.55, ver_tp=0.9, ver_fp=0.1, ver_cost=0.05,
    # noise
    sig_tput=0.012, sig_lat=0.02, n_eval=500, n_tt=200,
)

DUR_REF = 20.0      # reference benchmark duration: the noise levels in BASE are quoted at this `dur`
N_ACC = 2000        # proposals behind each reported per-position acceptance rate, at DUR_REF


def _dur(cfg):
    return max(float(cfg.get("dur", DUR_REF)), 1e-9) / DUR_REF


def sig_of(p, cfg, key):
    """A longer benchmark averages more decode steps, so its run-to-run noise falls as 1/sqrt(dur).

    Duration is therefore the precision-for-cost knob of this service, and a blueprint that wants a
    quantity measured tightly has to pay for it - the v3 lesson that a cost knob buying nothing is a knob
    the agent should always set to its minimum."""
    return float(p[key]) / math.sqrt(_dur(cfg))


def n_acc_of(p, cfg):
    """Proposals behind one reported acceptance rate: proportional to how long the benchmark ran."""
    return max(N_ACC * _dur(cfg), 1.0)


def full(p):
    out = dict(BASE); out.update(p or {})
    out["quant_g"] = dict(out["quant_g"] or {"16": 1.0e-9, "8": 3.0, "4": 6.0})
    out["delta_ptq"] = dict(out["delta_ptq"] or {"16": 0.0, "8": 0.0, "4": 0.0})
    out["drafts"] = dict(out["drafts"] or {"none": {"a": 0.0, "c": 0.0, "rho": 1.0}})
    return out


def _h(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def _key(bits):
    return str(int(round(float(bits))))


# --------------------------------------------------------------------------------- S5 memory / S1 roofline
def weight_bytes(p, bits):
    return p["N"] * float(bits) / 8.0


def kv_per_token(p):
    """S5: bytes of KV cache per token of context, for one sequence."""
    return 2.0 * p["L"] * p["H_kv"] * p["d_head"] * p["kv_bytes"]


def batch_max(p, bits, seq):
    """S5: the largest batch whose weights + activations + KV cache fit in memory."""
    free = p["M_gpu"] * p["mem_util"] - weight_bytes(p, bits) - p["act_bytes"]
    if free <= 0:
        return 0
    return int(free // (kv_per_token(p) * float(seq)))


def step_time(p, bits, seq, batch):
    """S1: one decode step is the max of the memory-traffic time and the arithmetic time."""
    mem = (weight_bytes(p, bits) + batch * kv_per_token(p) * float(seq)) / p["BW"]
    comp = 2.0 * p["N"] * batch / (p["P_peak"] * p["eff"])
    return max(mem, comp)


def b_crit(p, bits, seq):
    """S1: the batch at which the decode step stops being memory-bound (ignoring KV traffic)."""
    lhs = weight_bytes(p, bits) / p["BW"]
    return lhs * p["P_peak"] * p["eff"] / (2.0 * p["N"])


# ----------------------------------------------------------------------------------- S3 speculation
def accept_profile(p, draft, g):
    """S3: acceptance decays with draft position.

    A deployment only ever measures the positions its own proposal length exercises.  `rho_tail` continues
    the decay beyond position `tail_from` (0-based) and defaults to `rho`, so a lab that leaves it unset has
    one decay constant everywhere; a lab that sets it has a profile whose tail cannot be measured from
    proposals shorter than `tail_from + 2`.
    """
    d = p["drafts"][draft]
    a0 = float(d["a"]); rho = float(d.get("rho", 1.0))
    k = int(d.get("tail_from", 0) or 0)
    rt = rho if d.get("rho_tail") is None else float(d["rho_tail"])
    out = []
    for i in range(int(g)):
        a = a0 * rho ** i if (k <= 0 or i < k) else a0 * rho ** (k - 1) * rt ** (i - k + 1)
        out.append(min(max(a, 0.0), 1.0))
    return out


def exp_tokens(p, draft, g):
    """S3: expected accepted tokens per verification step (1 + prefix products of acceptance)."""
    if g <= 0:
        return 1.0
    e = 1.0; pref = 1.0
    for a in accept_profile(p, draft, g):
        pref *= a; e += pref
    return float(e)


def spec_step(p, bits, seq, batch, draft, g):
    """Wall time of one verification step: g draft steps at relative cost c, then one target pass."""
    base = step_time(p, bits, seq, batch)
    c = float(p["drafts"][draft].get("c", 0.0))
    return base * (1.0 + c * int(g))


def tokens_per_s(p, bits, seq, batch, draft, g):
    t = spec_step(p, bits, seq, batch, draft, g)
    return float(batch * exp_tokens(p, draft, g) / t)


# ----------------------------------------------------------------------------------- S2 quality
def n_eff(p, bits):
    g = float(p["quant_g"][_key(bits)])
    return p["N"] * (1.0 - math.exp(-float(bits) / max(g, 1e-12)))


def loss_of(p, bits):
    L = p["A_loss"] * n_eff(p, bits) ** (-p["alpha"]) + p["E_loss"] + float(p["delta_ptq"][_key(bits)])
    if p["B_loss"]:
        L += p["B_loss"] * p["D_tok"] ** (-p["beta"])
    return float(L)


def acc_of(p, bits):
    return float(p["acc_max"] / (1.0 + math.exp(p["acc_k"] * (loss_of(p, bits) - p["acc_l0"]))))


def acc_l0_for(p, bits, acc_target):
    """Blueprint helper: the acc_l0 that makes accuracy at `bits` equal `acc_target`."""
    r = p["acc_max"] / float(acc_target) - 1.0
    return float(loss_of(p, bits) - math.log(max(r, 1e-12)) / p["acc_k"])


# ----------------------------------------------------------------------------------- S4 queueing
def service_time(p, bits, seq, batch):
    """Amortised accelerator time one request occupies: generation at the batched step rate."""
    t = step_time(p, bits, seq, batch)
    return float(p["gen_tok"] * t / max(batch, 1) + p["prompt_tok"] * 2.0 * p["N"]
                 / (p["P_peak"] * p["prefill_flop_eff"] * max(batch, 1)))


def queue_stats(p, bits, seq, batch, rate):
    """S4: utilisation is not a knob -- rho = rate * E[S] and E[S] is set by the batch.

    Mean waiting time is the Pollaczek-Khinchine formula; the sojourn-time distribution is modelled as
    exponential with that mean, so p50 = ln(2) E[T] and p99 = ln(100) E[T]."""
    ES = service_time(p, bits, seq, batch)
    rho = float(rate) * ES
    if rho >= 1.0:
        return {"util": round(rho, 4), "stable": False}
    Wq = rho * ES * (1.0 + p["cs2"]) / (2.0 * (1.0 - rho))
    ET = Wq + ES
    return {"util": round(rho, 6), "stable": True, "E_S_ms": 1e3 * ES, "E_T_ms": 1e3 * ET,
            "p50_ms": 1e3 * math.log(2.0) * ET, "p99_ms": 1e3 * math.log(100.0) * ET}


# ----------------------------------------------------------------------------------- S6 test-time compute
def _logbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def coverage(p, k):
    """S6: beta-binomial pass@k -- 1 - pass@k decays as a power law in k with exponent alpha."""
    a, b = p["tt_alpha"], p["tt_beta"]
    return float(1.0 - math.exp(_logbeta(a, b + float(k)) - _logbeta(a, b)))


def selected(p, k):
    """S6: what a noisy verifier can actually pick out, bounded by its precision on the sampled set."""
    cov = coverage(p, k); mode = p["tt_mode"]
    tp, fp = p["ver_tp"], p["ver_fp"]
    if tp <= 0:
        return float(mode)
    frac = cov * tp / max(cov * tp + (1.0 - cov) * fp, 1e-9)     # precision of the verifier's pick
    return float(min(cov, max(mode, frac * cov)))


# ----------------------------------------------------------------------------------- backend protocol
def check(sess, cfg):
    from ..lab import LabError
    p = sess.p; svc = cfg.get("svc", "bench")
    if _key(cfg.get("bits", 16)) not in p["quant_g"]:
        raise LabError("bits must be one of %s" % sorted(int(k) for k in p["quant_g"]))
    if cfg.get("draft", "none") not in p["drafts"]:
        raise LabError("draft must be one of %s" % sorted(p["drafts"]))
    if cfg.get("draft", "none") == "none" and int(cfg.get("spec_g", 0)) > 0:
        raise LabError("spec_g>0 requires a draft model")
    if svc in ("bench", "load"):
        bmax = batch_max(p, cfg["bits"], cfg["seq"])
        if int(cfg["batch"]) > bmax:
            raise LabError("out of memory: batch=%d exceeds the KV-cache capacity %d at bits=%d seq=%d"
                           % (int(cfg["batch"]), bmax, int(cfg["bits"]), int(cfg["seq"])))


def extras(sess, req, cfg):
    return {}


def cost(sess, cfg, extra):
    p = sess.p; svc = cfg.get("svc", "bench")
    if svc == "bench":
        return float(cfg.get("dur", 20.0))
    if svc == "load":
        return float(cfg.get("dur", 60.0))
    if svc == "quality":
        b = int(cfg.get("batch", 32))
        return float(p["n_eval"] * service_time(p, cfg["bits"], cfg["seq"], b))
    k = float(cfg.get("k", 1))
    b = int(cfg.get("batch", 32))
    return float(p["n_tt"] * k * service_time(p, cfg["bits"], cfg["seq"], b) * (1.0 + p["ver_cost"]))


def _rng(sess, cfg, seed, tag):
    return np.random.default_rng(_h(sess.salt, repr(sorted((k, str(v)) for k, v in cfg.items())), int(seed), tag))


def execute(sess, cfg, seed, extra=None):
    p = sess.p; svc = cfg.get("svc", "bench")
    out = {"svc": svc, "config": dict(cfg, seed=seed), "status": "ok"}
    rng = _rng(sess, cfg, seed, svc)
    bits = cfg.get("bits", 16); seq = cfg.get("seq", 1024)
    if svc == "bench":
        B = int(cfg["batch"]); draft = cfg.get("draft", "none"); g = int(cfg.get("spec_g", 0))
        tp = tokens_per_s(p, bits, seq, B, draft, g)
        t = spec_step(p, bits, seq, B, draft, g)
        prof = accept_profile(p, draft, g)
        s_tp, s_lat = sig_of(p, cfg, "sig_tput"), sig_of(p, cfg, "sig_lat")
        out.update({"batch": B, "tokens_per_s": round(tp * math.exp(s_tp * rng.normal()), 3),
                    "ms_per_token": round(1e3 * t / max(exp_tokens(p, draft, g), 1e-9)
                                          * math.exp(s_lat * rng.normal()), 4),
                    "kv_bytes_per_seq": kv_per_token(p) * float(seq),
                    "batch_max": batch_max(p, bits, seq)})
        if g > 0:
            # Per-position acceptance is *estimated* from the proposals the benchmark actually made, so it
            # carries sampling noise and a longer run estimates it better.  The expected accepted-tokens-
            # per-step is deliberately NOT reported: it is a noise-free functional of the hidden acceptance
            # profile, and printing it would both contradict the noise model above and make the profile
            # exactly identifiable from one request.
            nacc = n_acc_of(p, cfg)
            obs = [round(min(1.0, max(0.0, a + math.sqrt(max(a * (1 - a), 1e-9) / nacc)
                                      * float(rng.normal()))), 4) for a in prof]
            out["accept_by_position"] = obs
            out["accept_rate"] = round(sum(obs) / len(obs), 4)
        return out
    if svc == "quality":
        a = acc_of(p, bits); n = p["n_eval"]
        out.update({"n_items": n, "acc": round(min(1.0, max(0.0,
                    a + math.sqrt(max(a * (1 - a), 1e-9) / n) * float(rng.normal()))), 5)})
        return out
    if svc == "load":
        B = int(cfg["batch"]); st = queue_stats(p, bits, seq, B, cfg["rate"])
        if not st["stable"]:
            out.update({"status": "unstable", "util": st["util"],
                        "note": "arrival rate exceeds the service rate at this batch; the queue grows without bound"})
            return out
        s_lat = sig_of(p, cfg, "sig_lat")
        out.update({"util": st["util"], "rate": cfg["rate"],
                    "p50_ms": round(st["p50_ms"] * math.exp(s_lat * float(rng.normal())), 2),
                    "p99_ms": round(st["p99_ms"] * math.exp(s_lat * float(rng.normal())), 2)})
        return out
    k = int(cfg.get("k", 1)); n = p["n_tt"]
    cov = coverage(p, k); sel = selected(p, k)
    # coupled draw: a verifier can only select a correct answer on an item that has one
    covered = rng.random(n) < cov
    picked = covered & (rng.random(n) < (sel / cov if cov > 0 else 0.0))
    out.update({"k": k, "n_items": n,
                "cov_at_k": round(float(covered.mean()), 5),
                "acc_selected": round(float(picked.mean()), 5)})
    return out


def predict_row(p, row):
    cfg = row.get("config", {}); svc = row.get("svc", "bench")
    bits = cfg.get("bits", 16); seq = cfg.get("seq", 1024)
    out = {}
    if svc == "bench":
        B = int(cfg["batch"]); draft = cfg.get("draft", "none"); g = int(cfg.get("spec_g", 0))
        tp = tokens_per_s(p, bits, seq, B, draft, g)
        out["tokens_per_s"] = (tp, sig_of(p, cfg, "sig_tput") * tp)
        t = 1e3 * spec_step(p, bits, seq, B, draft, g) / max(exp_tokens(p, draft, g), 1e-9)
        out["ms_per_token"] = (t, sig_of(p, cfg, "sig_lat") * t)
        # Each reported acceptance position is its own observable; a counterexample world has to
        # reproduce the whole profile, not just the throughput it implies.  `accept_rate` is the mean
        # of these and is deliberately left out, so that it is not counted twice.  The sd is the one
        # that row's own `dur` bought, so a long run pins the profile down harder than a short one.
        nacc = n_acc_of(p, cfg)
        for i, a in enumerate(accept_profile(p, draft, g)):
            out["accept_pos%d" % i] = (a, math.sqrt(max(a * (1 - a), 1e-9) / nacc))
    elif svc == "quality":
        a = acc_of(p, bits)
        out["acc"] = (a, math.sqrt(max(a * (1 - a), 1e-9) / p["n_eval"]))
    elif svc == "load":
        st = queue_stats(p, bits, seq, int(cfg["batch"]), cfg["rate"])
        if st["stable"] and "p99_ms" in row:
            out["p99_ms"] = (st["p99_ms"], sig_of(p, cfg, "sig_lat") * st["p99_ms"])
            out["util"] = (st["util"], 1e-6)
    else:
        k = int(cfg.get("k", 1)); n = p["n_tt"]
        cov = coverage(p, k); sel = selected(p, k)
        out["cov_at_k"] = (cov, math.sqrt(max(cov * (1 - cov), 1e-9) / n))
        out["acc_selected"] = (sel, math.sqrt(max(sel * (1 - sel), 1e-9) / n))
    return out


def observed(row):
    keys = ("tokens_per_s", "ms_per_token", "acc", "p99_ms", "util", "cov_at_k", "acc_selected")
    out = {k: row[k] for k in keys if k in row}
    for i, a in enumerate(row.get("accept_by_position", []) or []):
        out["accept_pos%d" % i] = a
    return out


MANUAL = r"""## 1. What the service models (abstraction boundary)

ServeLab is a **simulated** inference-serving stack.  No real kernel is run; every request is answered by
a simulator of one fixed model on one fixed accelerator type.  The simulator is not a replica of any real
deployment: **its hardware and model constants were drawn fresh for this lab**, so published roofline
numbers, acceptance rates and quantization results describe other systems.  Measure; do not recall.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Arithmetic versus memory.*  A decode step costs the larger of its memory-traffic time and its
  arithmetic time, so throughput per accelerator rises with batch up to a point and then stops.
- *KV cache.*  Each sequence holds a cache proportional to its context length; weights, activations and
  caches share a fixed memory, which puts a hard ceiling on batch.  A request above the ceiling is refused.
- *Numeric precision.*  Serving at fewer bits changes both the memory traffic and the served quality.
- *Speculative decoding.*  A draft model proposes several tokens which the target verifies; how many are
  accepted can depend on the position within the proposal, and a decay measured over one proposal length
  does not have to continue at the same rate beyond it.  Drafting costs time per proposed token.
- *Queueing.*  Under a Poisson arrival process, mean end-to-end latency follows from utilisation and the
  variability of service times; latency quantiles are taken from an exponential sojourn-time distribution
  with that mean.  Utilisation is *not* something you set: it is the arrival rate times the mean service
  time, and the mean service time depends on the batch.
- *Test-time compute.*  Sampling k answers raises the chance that a correct answer is *among* them; what
  a verifier can actually *select* is a different and lower quantity.
- *Noise.*  Throughput and latency carry multiplicative run-to-run noise; accuracies and acceptance rates
  carry sampling noise over a fixed number of items.  A benchmark averages over the decode steps it had
  time for, so how long you run it is what buys precision.

Not modelled: kernel-level scheduling, network, tensor/pipeline parallel topology, tokenizer, disaggregated
prefill, cache reuse across requests.

## 2. Guarantees

- **Fixed laws.**  Every mechanism is a fixed law whose constants do not depend on which request you make.
  There are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  An identical request with the same seed returns identical numbers (and is charged
  again).  Different seeds give independent draws.  `batch_max`, `kv_bytes_per_seq` and `util` are exact.
- **Duration buys precision.**  `dur` is measured against a 20-accelerator-second reference run: the
  relative noise on `tokens_per_s`, `ms_per_token` and the latency quantiles scales as `1/sqrt(dur/20)`,
  and the number of proposals behind each reported acceptance position is proportional to `dur`.  A run
  costs what its `dur` says, so precision is something you buy.
- **Refusals are free.**  A configuration the service refuses (for example, out of memory) is not charged.
- **Notebook rows are real service responses.**  The team's notes are the team's interpretation and may be wrong.
"""

CLI_HELP = r"""## 4. Using the service

    lab spec
    lab run svc=bench batch=64 seq=2048 bits=16 dur=20 seed=0
    lab run svc=bench batch=64 seq=2048 bits=16 draft=small spec_g=4 dur=20 seed=0
    lab run svc=quality batch=64 seq=2048 bits=4 seed=0
    lab run svc=load batch=64 seq=2048 bits=16 rate=12 dur=60 seed=0
    lab run svc=ttc k=16 batch=64 seq=2048 bits=16 seed=0
    lab batch plan.json
    lab status
    lab history

`bench` reports `tokens_per_s`, `ms_per_token`, `kv_bytes_per_seq`, `batch_max`, and -- when speculating --
`accept_by_position` (one estimated rate per proposal position) and their mean `accept_rate`.  `quality`
reports `acc` on a fixed held-out set.  `load`
reports `util`, `p50_ms`, `p99_ms`, or `status="unstable"` when the queue diverges.  `ttc` reports
`cov_at_k` (a correct answer is among the k samples) and `acc_selected` (the verifier's pick is correct).
Cost is accelerator-seconds: a long benchmark, a large-k `ttc` call and a `quality` call at a big batch
cost very different amounts.
"""
