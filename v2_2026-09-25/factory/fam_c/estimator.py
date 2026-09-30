"""One parameterised estimator so that the reference analyst and every impostor differ by exactly one flag.

The reference setting is the textbook-correct procedure for this data:
  encode every pre-run knob categorically, saturate all pairwise knob interactions, adjust for the
  pre-training hardware probe, adjust for NOTHING measured during the run, then answer a query only if the
  requested contrast is estimable from the observed design (and every requested level was observed).
"""
import numpy as np

KNOBS = ["lr_scale", "warmup", "micro_bs", "grad_accum", "zero_stage", "offload", "seq_len",
         "act_ckpt", "dropout", "opt_eps"]
PROBE = ["nccl_bw_gbps", "sm_clock_mhz"]
DURING = ["grad_norm_p95", "throughput_toks_s", "step_time_ms"]

REF = dict(interactions=True, probe="categorical", during=False, marginal=False,
           estimability=True, numeric_knobs=())


def _levels(rows, k):
    return sorted({r[k] for r in rows})


class Design:
    def __init__(self, rows, opt):
        self.opt = opt
        self.knobs = [k for k in KNOBS]
        self.lv = {k: _levels(rows, k) for k in self.knobs}
        self.blocks = []                                  # (kind, payload) describing each knob column
        for k in self.knobs:
            if k in opt["numeric_knobs"]:
                self.blocks.append(("num", k, None))
            else:
                for v in self.lv[k][1:]:
                    self.blocks.append(("dum", k, v))
        self.n_main = len(self.blocks)
        self.pairs = []
        if opt["interactions"]:
            for i in range(self.n_main):
                for j in range(i + 1, self.n_main):
                    if self.blocks[i][1] != self.blocks[j][1]:
                        self.pairs.append((i, j))

    def row(self, c):
        """Knob part of the feature row for a config, or None if a level was never observed."""
        m = []
        for kind, k, v in self.blocks:
            if kind == "num":
                m.append(float(c[k]))
            else:
                if c[k] not in self.lv[k]:
                    return None
                m.append(1.0 if c[k] == v else 0.0)
        return m + [m[i] * m[j] for i, j in self.pairs]


def fit(rows, opt, treated=None):
    d = Design(rows, opt)
    if opt["marginal"]:                                   # difference of means adjusting for nothing
        d2 = Design(rows, dict(opt, interactions=False, numeric_knobs=()))
        d2.blocks = [b for b in d2.blocks if b[1] == treated]
        d2.pairs = []
        d2.n_main = len(d2.blocks)
        d = d2
    extra = []
    if opt["probe"] == "categorical" and not opt["marginal"]:
        for k in PROBE:
            for v in _levels(rows, k)[1:]:
                extra.append((k, v))
    elif opt["probe"] == "numeric" and not opt["marginal"]:
        extra = [(k, None) for k in PROBE]
    dur = [] if opt["marginal"] else (DURING if opt["during"] is True else list(opt["during"] or []))
    X = []
    for r in rows:
        base = d.row(r)
        e = [(1.0 if r[k] == v else 0.0) if v is not None else float(r[k]) for k, v in extra]
        X.append([1.0] + base + e + [float(r[k]) for k in dur])
    X = np.asarray(X)
    y = np.asarray([r["val_loss"] for r in rows])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = int((s > s[0] * 1e-10).sum())
    beta = Vt[:rank].T @ ((U[:, :rank].T @ y) / s[:rank])
    null = Vt[rank:]
    return d, beta, null, X.shape[1], len(extra) + len(dur)


def answer(rows, queries, opt):
    out = {}
    cache = {}
    for q in queries:
        key = q["knob"] if opt["marginal"] else "_"
        if key not in cache:
            cache[key] = fit(rows, opt, treated=q["knob"])
        d, beta, null, p, n_extra = cache[key]
        ca = dict(q["baseline"]); ca[q["knob"]] = q["from"]
        cb = dict(q["baseline"]); cb[q["knob"]] = q["to"]
        ra, rb = d.row(ca), d.row(cb)
        if ra is None or rb is None:
            out[q["id"]] = {"verdict": "underdetermined"}
            continue
        c = np.asarray([0.0] + [x - y for x, y in zip(rb, ra)] + [0.0] * n_extra)
        if opt["estimability"] and null.size:
            if np.linalg.norm(null @ c) > 1e-8 * max(1.0, np.linalg.norm(c)):
                out[q["id"]] = {"verdict": "underdetermined"}
                continue
        out[q["id"]] = {"verdict": "identified", "delta": float(c @ beta)}
    return out


# Wrong procedures: each is the reference with exactly one methodological flaw.  The generator requires
# every one of them to produce at least one wrong verdict or one out-of-tolerance estimate.
CANDIDATES = {
    "throughput_as_hw_proxy": dict(REF, probe="none", during=["throughput_toks_s"]),
    "additive_only": dict(REF, interactions=False),
    "no_hardware_probe": dict(REF, probe="none"),
    "adjust_during_run_too": dict(REF, during=True),
    "marginal_means": dict(REF, marginal=True),
    "no_estimability_check": dict(REF, estimability=False),
    "numeric_knobs": dict(REF, numeric_knobs=("warmup", "micro_bs", "grad_accum", "opt_eps", "lr_scale")),
    "additive_no_probe": dict(REF, interactions=False, probe="none"),
    "additive_with_during": dict(REF, interactions=False, during=True),
    "numeric_no_estimability": dict(REF, estimability=False, numeric_knobs=("warmup", "opt_eps")),
}

# Different but equally valid procedures.  The generator requires these to PASS: a difficulty gate that
# rejected them would be rejecting competence rather than error.
ALTERNATIVES = {
    "probe_as_numeric": dict(REF, probe="numeric"),
}
