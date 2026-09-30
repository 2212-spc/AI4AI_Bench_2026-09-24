"""Hidden world for delayed_labels (credit default with censoring + calendar-time shocks).

Accounts open every month (cohorts 0..T-1); snapshot at month T. Each account has features x (drifting
across cohorts). Monthly default hazard
    h(k, c, x) = h0(k) * exp(r(x)) * m(c)          k = tenure month (1..), c = calendar month
h0 = tenure shape (hump), r = risk score, m = calendar multiplier (economy).  Accounts also close
(attrition, independent of default given x) and are administratively censored at the snapshot.
Target: for NEW applicants (x from the latest cohort distribution), P(default within 12 months | account
stays open) under the CURRENT conditions m_now = mean m over the last 3 months (stated in task.md)."""
import numpy as np
import pandas as pd

D = 6
BETA = np.array([0.55, -0.40, 0.30, 0.0, 0.20, -0.25])


def risk(X):
    return X @ BETA + 0.35 * np.tanh(X[:, 0] * X[:, 1]) + 0.25 * (X[:, 2] > 1.0)


def h0(k, peak, base):
    k = np.asarray(k, float)
    return base * (0.5 + 1.2 * np.exp(-0.5 * ((k - peak) / 3.0) ** 2) + 0.3 * np.exp(-k / 24))


def calendar(kind, T):
    c = np.arange(T + 12)
    if kind == "flat":  m = np.ones_like(c, float)
    elif kind == "shock_up":   m = np.where(c >= T - 7, 2.2, 1.0)
    elif kind == "rise":       m = np.exp(0.9 * np.clip((c - (T - 18)) / 18, 0, None))
    elif kind == "recovery":   m = np.where(c >= T - 8, 0.5, 1.4)
    elif kind == "cycle":      m = 1 + 0.5 * np.sin(2 * np.pi * c / 20)
    else: raise ValueError(kind)
    return m


def simulate(seed, T, n_per_month, kind, peak, base, churn, drift):
    rng = np.random.default_rng(seed)
    m = calendar(kind, T)
    rows = []
    for s in range(T):
        n = rng.poisson(n_per_month)
        X = rng.normal(0, 1, (n, D)); X[:, 0] += drift * (s - T / 2) / T; X[:, 4] -= 0.5 * drift * (s - T / 2) / T
        r = risk(X)
        dm = np.full(n, np.nan); cm = np.full(n, np.nan)
        alive = np.ones(n, bool)
        q = churn * np.exp(-0.3 * X[:, 1])                   # attrition depends on x only
        for k in range(1, T - s + 1):                          # tenure months observed before snapshot
            c = s + k - 1
            if c >= T: break
            idx = np.nonzero(alive)[0]
            if not len(idx): break
            u = rng.random(len(idx))
            hz = np.minimum(h0(k, peak, base) * np.exp(r[idx]) * m[c], 0.9)
            d = u < hz
            dm[idx[d]] = c
            cl = (~d) & (rng.random(len(idx)) < q[idx])
            cm[idx[cl]] = c
            alive[idx[d | cl]] = False
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(D)])
        df.insert(0, "open_month", s)
        df["default_month"] = dm; df["close_month"] = cm
        rows.append(df)
    hist = pd.concat(rows, ignore_index=True)
    hist.insert(0, "account_id", np.arange(len(hist)))
    # deployment applicants: latest cohort distribution
    Xn = rng.normal(0, 1, (4000, D)); Xn[:, 0] += drift * 0.5; Xn[:, 4] -= 0.25 * drift
    m_now = m[T - 3:T].mean()
    haz = np.minimum(h0(np.arange(1, 13), peak, base)[None, :] * np.exp(risk(Xn))[:, None] * m_now, 0.9)
    p = 1 - np.prod(1 - haz, axis=1)
    new = pd.DataFrame(Xn, columns=[f"x{i+1}" for i in range(D)])
    return hist, new, p


# name, seed, T, accounts/month, calendar kind, hazard peak month, base hazard, churn/month, drift, hidden
SETTINGS = [
    ("dev_flat", 41, 48, 800, "flat", 6, 0.006, 0.02, 1.0, False),
    ("dev_shock", 42, 48, 800, "shock_up", 5, 0.006, 0.025, 1.0, False),
    ("hid_rise", 401, 60, 700, "rise", 8, 0.005, 0.02, 1.5, True),
    ("hid_recover", 402, 48, 900, "recovery", 4, 0.007, 0.03, 0.5, True),
    ("hid_cycle", 403, 54, 800, "cycle", 7, 0.006, 0.015, 1.0, True),
    ("hid_flat_churn", 404, 42, 1000, "flat", 10, 0.004, 0.05, 2.0, True),
]


def build_setting(row):
    name, seed, T, n, kind, peak, base, churn, drift, hidden = row
    hist, new, p = simulate(seed, T, n, kind, peak, base, churn, drift)
    return dict(name=name, history=hist, snapshot_month=T, new=new, truth=p, hidden=hidden)
