"""Goal-2 prototype: a transplanted real mechanism (Multi-Power Law, Luo et al. ICLR 2025)
turned into a virtual LR-schedule world with resampled constants + two documented extensions,
plus a sibling world (Tissue et al. 2024 momentum law) used only as an ensemble gate.

numpy only. All learning rates are in units of the "doc" LR (doc LR == 1.0).

Primary world (kind='mpl'):
  theta_t   = jit * h(t) * (th_inf - (th_inf - th0) * exp(-Araw_{t-1} / Sc))      # stability edge, raised by warmup area
  diverge   if eta_t >= theta_t for any t                                          # extension 1 (edge)
  u_t       = eta_t * (1 - (eta_t / theta_t)^p)                                    # extension 1 (efficiency near edge)
  L(t)      = L0 + A * (s0 + sum_{tau<=t} u_tau)^(-alpha) + B * peak(t) - LD(t)    # extension 2: plateau noise B*peak
  LD(t)     = B * sum_{k>t_pk}^{t} (eta_{k-1} - eta_k) * G(eta_k^(-gamma) * S_k(t)),  G(x) = 1 - (C x + 1)^(-beta)
  S_k(t)    = sum_{tau=k}^{t} eta_tau                                             # raw LR area after k (as in MPL)
  h(t)      = 1                     (kinds F, C)
            = min(1, (t/tc)^(-psi)) (kind L: late-onset instability / progressive sharpening)

Sibling world (kind='mom'): same edge/efficiency/plateau; annealing term replaced by
  C' * sum_{i>t_pk}^{t} m_i * eta_i^eps,   m_i = lam * m_{i-1} + (eta_{i-1} - eta_i)
with (L0', A', alpha', B', C', lam, eps) fitted to the primary world on a monotone "validated domain".
"""
import numpy as np

EPS_ETA = 1e-6


def _theta(c, eta, jit=1.0):
    T = len(eta)
    araw_prev = np.concatenate([[0.0], np.cumsum(eta)[:-1]])
    th = c['th_inf'] - (c['th_inf'] - c['th0']) * np.exp(-araw_prev / c['Sc'])
    if c.get('tc') is not None:
        t = np.arange(1, T + 1, dtype=float)
        h = np.minimum(1.0, (t / c['tc']) ** (-c['psi']))
        th = th * h
    return jit * th


def diverge_step(c, eta, jit=1.0):
    """1-based step of first divergence, or None."""
    th = _theta(c, eta, jit)
    bad = np.nonzero(eta >= th)[0]
    return int(bad[0]) + 1 if len(bad) else None


def _eff_area(c, eta, jit=1.0):
    th = _theta(c, eta, jit)
    r = np.clip(eta / th, 0.0, 1.0)
    return eta * (1.0 - r ** c['p'])


def _anneal_mpl(c, eta, t_pk, upto=None):
    """LD(t) for t = upto (1-based length of prefix). O(t)."""
    e = eta if upto is None else eta[:upto]
    if len(e) <= t_pk + 1:
        return 0.0
    d = e[t_pk:-1] - e[t_pk + 1:]                  # drops at k = t_pk+1 .. t-1 (0-based index k)
    if c.get('absbug'):                            # PLANTED builder bug for the gate demo: |drop| instead of signed drop
        d = np.abs(d)
    rc = np.cumsum(e[::-1])[::-1]                  # rc[k] = sum_{tau>=k} e[tau]
    Sk = rc[t_pk + 1:]
    x = np.maximum(e[t_pk + 1:], EPS_ETA) ** (-c['gamma']) * Sk
    G = 1.0 - (c['C'] * x + 1.0) ** (-c['beta'])
    return c['B'] * float(np.dot(d, G))


def _mom_series(lam, d):
    """m_i = lam*m_{i-1} + d_i via FFT convolution (d is 1-D, m_0 = d_0)."""
    n = len(d)
    if n == 0:
        return d
    L = 1
    while L < 2 * n:
        L *= 2
    k = lam ** np.arange(n)
    m = np.fft.irfft(np.fft.rfft(d, L) * np.fft.rfft(k, L), L)[:n]
    return m


def _anneal_mom(c, eta, t_pk, upto=None):
    e = eta if upto is None else eta[:upto]
    if len(e) <= t_pk + 1:
        return 0.0
    d = e[t_pk:-1] - e[t_pk + 1:]
    m = _mom_series(c['lam'], d)
    w = np.maximum(e[t_pk + 1:], 0.0) ** c['eps'] if c['eps'] > 0 else 1.0
    return c['Cm'] * float(np.sum(m * w))


def loss_at(c, eta, upto=None, jit=1.0, kind='mpl'):
    e = eta if upto is None else eta[:upto]
    t_pk = int(np.argmax(e))
    u = _eff_area(c, e, jit)
    S = c['s0'] + float(np.sum(u))
    base = c['L0'] + c['A'] * S ** (-c['alpha']) + c['B'] * float(e[t_pk])
    if kind == 'mpl':
        return base - _anneal_mpl(c, e, t_pk)
    else:
        return base - _anneal_mom(c, e, t_pk)


def L_div(c):
    return c['L0'] + c['A'] * c['s0'] ** (-c['alpha'])


def final_loss(c, eta, jit=1.0, kind='mpl'):
    """Deterministic final loss; divergence -> L_div (loss at init)."""
    if diverge_step(c, eta, jit) is not None:
        return L_div(c)
    return loss_at(c, eta, None, jit, kind)


GRADE_JITS = np.exp(0.03 * np.array([-1.53, -0.89, -0.49, -0.16, 0.16, 0.49, 0.89, 1.53]))  # 8 fixed quantiles of N(0,1)


def graded_loss(c, eta, kind='mpl'):
    """Grading: mean over 8 fixed edge-jitter quantiles (run-to-run variation of the stability edge)."""
    return float(np.mean([final_loss(c, eta, j, kind) for j in GRADE_JITS]))


def curve(c, eta, every=100, jit=1.0, kind='mpl'):
    """Noise-free loss at eval points; truncated at divergence."""
    T = len(eta)
    ds = diverge_step(c, eta, jit)
    last = T if ds is None else ds - 1
    pts = [t for t in range(every, last + 1, every)]
    if last not in pts and last > 0:
        pts.append(last)
    return np.array(pts), np.array([loss_at(c, eta, t, jit, kind) for t in pts]), ds


class Lab:
    """What the agent (or a scripted policy) can touch: noisy runs under a step budget."""

    def __init__(self, c, budget, sigma_obs=0.004, seed=0, every=100):
        self.c, self.budget, self.sigma, self.every = c, budget, sigma_obs, every
        self.rng = np.random.default_rng(seed)
        self.used = 0
        self.log = []

    def run(self, eta):
        eta = np.asarray(eta, float)
        jit = float(np.exp(0.03 * self.rng.standard_normal()))
        ts, ls, ds = curve(self.c, eta, self.every, jit)
        cost = len(eta) if ds is None else ds
        if self.used + cost > self.budget:
            raise RuntimeError('budget exhausted')
        self.used += cost
        obs = ls + self.sigma * self.rng.standard_normal(len(ls))
        self.log.append(dict(T=len(eta), cost=cost, diverged=ds))
        return ts, obs, ds

    def left(self):
        return self.budget - self.used


# ---------------------------------------------------------------- schedules
def sched_const(T, peak, wu):
    t = np.arange(1, T + 1, dtype=float)
    return np.where(t <= wu, peak * t / max(wu, 1), peak)


def sched_cosine(T, peak, wu, final_frac=0.1):
    t = np.arange(1, T + 1, dtype=float)
    prog = np.clip((t - wu) / max(T - wu, 1), 0, 1)
    cos = final_frac + (1 - final_frac) * 0.5 * (1 + np.cos(np.pi * prog))
    return np.where(t <= wu, peak * t / max(wu, 1), peak * cos)


def sched_wsd(T, peak, wu, decay_frac, final=0.0):
    t = np.arange(1, T + 1, dtype=float)
    td = T * (1 - decay_frac)
    dec = np.clip((t - td) / max(T - td, 1), 0, 1)
    body = peak * (1 - (1 - final) * dec)
    return np.where(t <= wu, peak * t / max(wu, 1), body)


# ---------------------------------------------------------------- world sampling
def sample_world(kind, seed, T=10000):
    r = np.random.default_rng(seed)
    U = lambda a, b: float(r.uniform(a, b))
    LU = lambda a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
    alpha = U(0.35, 0.55)
    c = dict(L0=U(2.6, 3.0), alpha=alpha, A=U(0.30, 0.40) * 8000.0 ** alpha, s0=10.0,
             B=U(0.15, 0.30), C=LU(0.005, 0.05), beta=U(0.4, 0.7), gamma=U(0.4, 0.7),
             th_inf=U(1.8, 2.4), th0=U(0.9, 1.3), Sc=LU(20, 100), p=U(3.0, 6.0), tc=None, psi=0.0)
    if kind == 'C':   # counterfactual constants: doc LR sits in the wasteful band just under the edge; slow anneal saturation
        c.update(th_inf=U(1.15, 1.35), th0=U(1.06, 1.12), Sc=LU(20, 100), p=U(2.5, 3.5),
                 B=U(0.35, 0.5), C=LU(0.002, 0.008), beta=U(0.5, 0.8))
    if kind == 'L':   # late-onset instability: edge shrinks after tc (progressive sharpening)
        tc = U(0.35, 0.5) * T
        shrink = U(0.40, 0.55)
        c.update(tc=tc, psi=float(np.log(1 / shrink) / np.log(T / tc)))
    c['kind'] = kind
    c['seed'] = seed
    return c
