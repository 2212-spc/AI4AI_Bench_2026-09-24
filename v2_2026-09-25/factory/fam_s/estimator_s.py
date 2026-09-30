"""Family S reference estimator and the library of plausible-but-wrong procedures.

The reference makes no functional-form assumption at all: it fits a *saturated* model, one mean per
launched cell, by maximum likelihood for a normal sample observed only when it falls below the guard.
Preemption is ignored on purpose - it is independent of the outcome, so conditioning on "this run is in
the table" leaves an i.i.d. draw from the truncated law and only costs sample size.
"""
import json, math
import numpy as np

SQ2 = math.sqrt(2.0)


def _Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / SQ2))


def _phi(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _logPhi(a):
    """log of the standard normal cdf, accurate in the far left tail where Phi underflows to zero."""
    a = float(a)
    if a > -8.0:
        return math.log(max(0.5 * math.erfc(-a / SQ2), 1e-300))
    return -0.5 * a * a - math.log(-a) - 0.5 * math.log(2 * math.pi) + math.log1p(-1.0 / (a * a) + 3.0 / a ** 4)


def _ll_cell(y, mu, sig, tau):
    r = (y - mu) / sig
    return float(-len(y) * math.log(sig) - 0.5 * np.sum(r * r) - len(y) * _logPhi((tau - mu) / sig))


def _argmax_mu(y, sig, tau, lo, hi, iters=80):
    """Golden-section maximisation of the 1-D truncated-normal log-likelihood in mu."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _ll_cell(y, c, sig, tau), _ll_cell(y, d, sig, tau)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = _ll_cell(y, d, sig, tau)
        else:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = _ll_cell(y, c, sig, tau)
    return 0.5 * (a + b)


def fit_saturated(groups, tau, sig_grid=None):
    """groups: {cell_key: np.array of observed val_loss}.  Returns (mu per cell, sigma)."""
    ys = np.concatenate([v for v in groups.values()]) if groups else np.array([0.0])
    lo, hi = float(ys.min()) - 1.5, tau + 1.5
    if sig_grid is None:
        s0 = float(ys.std()) + 1e-6
        sig_grid = np.linspace(0.4 * s0, 3.0 * s0, 45)
    best = None
    for sig in sig_grid:
        mus, tot = {}, 0.0
        for k, y in groups.items():
            m = _argmax_mu(y, sig, tau, lo, hi)
            mus[k] = m
            tot += _ll_cell(y, m, sig, tau)
        if best is None or tot > best[0]:
            best = (tot, mus, float(sig))
    # one refinement pass around the winning sigma
    s = best[2]
    fine = np.linspace(s * 0.85, s * 1.15, 25)
    for sig in fine:
        mus, tot = {}, 0.0
        for k, y in groups.items():
            m = _argmax_mu(y, sig, tau, lo, hi)
            mus[k] = m
            tot += _ll_cell(y, m, sig, tau)
        if tot > best[0]:
            best = (tot, mus, float(sig))
    return best[1], best[2]


class Data:
    def __init__(self, manifest, results, knobs, tau):
        self.knobs, self.tau = knobs, tau
        self.launched, self.obs = {}, {}
        self.levels = {k: set() for k in knobs}
        for r in manifest:
            k = tuple(r[c] for c in knobs)
            self.launched[k] = self.launched.get(k, 0) + 1
            for c in knobs:
                self.levels[c].add(r[c])
        for r in results:
            k = tuple(r[c] for c in knobs)
            self.obs.setdefault(k, []).append(float(r["val_loss"]))
        self.obs = {k: np.array(v) for k, v in self.obs.items()}


def _key(cfg, knobs):
    return tuple(cfg[k] for k in knobs)


REF = dict(method="saturated_truncated_mle")


def fit(d, spec=REF):
    if spec["method"] == "complete_case":
        return {k: float(v.mean()) for k, v in d.obs.items()}, float(np.concatenate(list(d.obs.values())).std())
    if spec["method"] == "complete_case_median":
        return {k: float(np.median(v)) for k, v in d.obs.items()}, 0.09
    if spec["method"] == "impute_guard":
        mu = {}
        for k, n in d.launched.items():
            y = d.obs.get(k, np.array([]))
            fill = d.tau + spec.get("plus", 0.0)
            mu[k] = float((y.sum() + fill * (n - len(y))) / n)
        return mu, 0.09
    if spec["method"] == "reweight_preempt":
        mu = {}
        for k, y in d.obs.items():
            mu[k] = float(y.mean())            # inverse-probability weights cancel inside a cell
        return mu, 0.09
    if spec["method"] == "survival_no_preempt":
        s = fit_saturated(d.obs, d.tau)[1]
        mu = {}
        for k, n in d.launched.items():
            p_miss = 1.0 - len(d.obs.get(k, [])) / float(n)
            p = min(max(p_miss, 1e-4), 1 - 1e-4)
            z = _z_of(1.0 - p)
            mu[k] = d.tau - s * z
        return mu, s
    if spec["method"] == "survival_with_preempt":
        s = fit_saturated(d.obs, d.tau)[1]
        clean = [k for k in d.launched if k in d.obs and len(d.obs[k]) > 0]
        # preemption rate per batch level, read off cells the guard cannot touch
        pre = {}
        for k in clean:
            y = d.obs[k]
            if float(y.mean()) < d.tau - 3.2 * s:
                pre.setdefault(k[1], []).append(1.0 - len(y) / float(d.launched[k]))
        pre = {b: float(np.mean(v)) for b, v in pre.items()}
        mu = {}
        for k, n in d.launched.items():
            pm = 1.0 - len(d.obs.get(k, [])) / float(n)
            pp = pre.get(k[1], 0.0)
            pg = min(max((pm - pp) / max(1e-6, 1.0 - pp), 1e-4), 1 - 1e-4)
            mu[k] = d.tau - s * _z_of(1.0 - pg)
        return mu, s
    if spec["method"] == "two_moment":
        s_glob = fit_saturated(d.obs, d.tau)[1]
        mu = {}
        for k, y in d.obs.items():
            m, v = float(y.mean()), float(y.var())
            MM = np.linspace(m - 0.05, d.tau + 0.9, 1400)[:, None]
            SS = np.linspace(0.6 * s_glob, 1.8 * s_glob, 40)[None, :]
            A = (d.tau - MM) / SS
            lam = _phi(A) / np.maximum(_Phi(A), 1e-12)
            em = MM - SS * lam
            ev = SS * SS * (1.0 - lam * (lam - A))
            e = (em - m) ** 2 / 1e-4 + (ev - v) ** 2 / 1e-8
            mu[k] = float(MM[np.unravel_index(np.argmin(e), e.shape)[0], 0])
        return mu, s_glob
    if spec["method"] == "tau_from_max":
        tau = float(max(v.max() for v in d.obs.values())) + 1e-6
        mu, s = fit_saturated(d.obs, tau)
        return mu, s
    if spec["method"] == "saturated_truncated_mle":
        return fit_saturated(d.obs, d.tau)
    raise ValueError(spec["method"])


def _z_of(p):
    """Inverse standard normal cdf (Acklam's rational approximation, ~1e-9 absolute)."""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def answer(d, queries, report_cells, spec=REF):
    """Returns (answers dict, censoring dict)."""
    mu, sig = fit(d, spec)
    pol = spec.get("policy", "cells")
    out = {}
    for q in queries:
        base = q["baseline"]
        ca = dict(base); ca[q["knob"]] = q["from"]
        cb = dict(base); cb[q["knob"]] = q["to"]
        ka, kb = _key(ca, d.knobs), _key(cb, d.knobs)
        bad = None
        for cfg, k in ((ca, ka), (cb, kb)):
            if cfg[q["knob"]] not in d.levels[q["knob"]] and k not in d.launched:
                bad = "level_never_launched"
            elif k not in d.launched:
                bad = bad or "combination_never_launched"
            elif len(d.obs.get(k, [])) == 0:
                bad = bad or "no_surviving_run"
        if pol == "extrapolate" and bad == "no_surviving_run":
            bad = None
            for k in (ka, kb):
                if len(d.obs.get(k, [])) == 0:
                    mu[k] = _extrapolate(d, mu, k)
        if pol == "abstain_if_any_missing":
            for k in (ka, kb):
                if k in d.launched and len(d.obs.get(k, [])) < d.launched[k]:
                    bad = bad or "no_surviving_run"
        if pol == "abstain_if_censored":
            for k in (ka, kb):
                if k in d.obs and len(d.obs[k]) < 0.97 * d.launched[k]:
                    bad = bad or "no_surviving_run"
        if bad:
            out[q["id"]] = {"verdict": "underdetermined", "reason": bad}
        else:
            out[q["id"]] = {"verdict": "identified", "delta": round(mu[kb] - mu[ka], 6),
                            "reason": "identified"}
    cen = _censor(d, mu, sig, report_cells, spec.get("censor", "counts_with_preemption"))
    return out, cen


def preempt_rates(d, mu, sig):
    """Per-batch preemption rate.

    Preemption is a function of `batch` alone and is independent of the outcome, so for every launched
    cell  E[n_obs] = n * (1 - p_pre(batch)) * (1 - p_guard(cell)).  The fitted mean supplies p_guard, and
    pooling the ratio over every cell at a batch level gives a low-variance estimate of the nuisance rate.
    Cells the guard has effectively swallowed carry no information and are dropped.
    """
    bi = d.knobs.index("batch")
    num, den = {}, {}
    for k, n in d.launched.items():
        y = d.obs.get(k, [])
        if len(y) == 0 or k not in mu:
            continue
        surv = float(_Phi((d.tau - mu[k]) / sig))
        if surv < 0.25:
            continue
        num[k[bi]] = num.get(k[bi], 0.0) + len(y)
        den[k[bi]] = den.get(k[bi], 0.0) + n * surv
    return {b: float(min(0.95, max(0.0, 1.0 - num[b] / den[b]))) for b in num if den.get(b, 0) > 0}


def _censor(d, mu, sig, report_cells, how):
    bi = d.knobs.index("batch")
    pre = preempt_rates(d, mu, sig)
    cen = {}
    for rid, cfg in report_cells:
        k = _key(cfg, d.knobs)
        n = d.launched.get(k, 0)
        if n == 0:
            cen[rid] = None
            continue
        nobs = len(d.obs.get(k, []))
        if how == "zero":
            cen[rid] = 0.0
        elif how == "raw_missing":
            cen[rid] = round(1.0 - nobs / float(n), 5)
        elif how == "model_phi":
            cen[rid] = 1.0 if nobs == 0 else round(float(1.0 - _Phi((d.tau - mu[k]) / sig)), 5)
        else:                                    # counts_with_preemption  (reference)
            p = pre.get(k[bi], 0.0)
            cen[rid] = round(float(min(1.0, max(0.0, 1.0 - (nobs / float(n)) / max(1e-9, 1.0 - p)))), 5)
    return cen


def _extrapolate(d, mu, k):
    """Additive + pairwise extrapolation into a cell with no survivor (what a model-based fit does)."""
    cols, rows, ys = _basis(d, list(mu)), [], []
    for kk, m in mu.items():
        rows.append(_row(kk, cols)); ys.append(m)
    A = np.array(rows, dtype=float); y = np.array(ys, dtype=float)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.array(_row(k, cols), dtype=float) @ beta)


def _basis(d, keys):
    cols = [("1",)]
    for i, kn in enumerate(d.knobs):
        lv = sorted({kk[i] for kk in keys}, key=str)
        for v in lv[1:]:
            cols.append(("m", i, v))
    for i in range(len(d.knobs)):
        for j in range(i + 1, len(d.knobs)):
            li = sorted({kk[i] for kk in keys}, key=str)[1:]
            lj = sorted({kk[j] for kk in keys}, key=str)[1:]
            for a in li:
                for b in lj:
                    cols.append(("x", i, a, j, b))
    return cols


def _row(k, cols):
    out = []
    for c in cols:
        if c[0] == "1":
            out.append(1.0)
        elif c[0] == "m":
            out.append(1.0 if k[c[1]] == c[2] else 0.0)
        else:
            out.append(1.0 if (k[c[1]] == c[2] and k[c[3]] == c[4]) else 0.0)
    return out


CANDIDATES = {
    "complete_case": dict(method="complete_case"),
    "complete_case_median": dict(method="complete_case_median"),
    "impute_at_guard": dict(method="impute_guard"),
    "impute_above_guard": dict(method="impute_guard", plus=0.09),
    "reweight_preemption_only": dict(method="reweight_preempt"),
    "survival_rate_ignoring_preemption": dict(method="survival_no_preempt"),
    "survival_rate_with_preemption_model": dict(method="survival_with_preempt"),
    "correct_fit_but_extrapolates_dead_cell": dict(method="saturated_truncated_mle", policy="extrapolate"),
    "correct_fit_but_abstains_on_any_hole": dict(method="saturated_truncated_mle",
                                                 policy="abstain_if_any_missing"),
    "correct_fit_but_abstains_when_censored": dict(method="saturated_truncated_mle",
                                                   policy="abstain_if_censored"),
    "complete_case_with_extrapolation": dict(method="complete_case", policy="extrapolate"),
    "reports_raw_missing_fraction": dict(method="saturated_truncated_mle", censor="raw_missing"),
    "reports_no_censoring": dict(method="saturated_truncated_mle", censor="zero"),
}

ALTERNATIVES = {
    "guard_threshold_read_off_max": dict(method="tau_from_max"),
    "model_based_censoring_report": dict(method="saturated_truncated_mle", censor="model_phi"),
}

# Consistent in principle but ill-conditioned here: under 40-75%% truncation the two moment equations are
# nearly degenerate, so the estimator carries a systematic ~0.03 error on the deep cells and lands exactly
# on the published tolerance.  It is neither required to pass nor required to fail; the certificate records
# where it lands.  This is a disclosed limitation of the task, not a hidden one.
BORDERLINE = {
    "moment_matching_under_heavy_truncation": dict(method="two_moment"),
}
