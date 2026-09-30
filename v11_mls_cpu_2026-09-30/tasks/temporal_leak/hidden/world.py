"""Store-demand warehouse with realistic data-arrival semantics.

Tables (the warehouse dump; every row carries the day it was written, `recorded_at`):
  sales_log : store, day, units, recorded_at   -- units of day d are first written at the end of day d
              (preliminary, incomplete count), then re-written at ages 1..A until final.  completeness[a]
              = expected fraction of the final count visible at age a (can be >1 when returns are netted later).
  promos    : store, day, discount, planned_at -- most promos are planned days ahead; 'flash' promos are
              decided the same morning (planned_at = day).
  weather   : store, day, kind in {forecast, actual}, temp, recorded_at -- forecast for day d issued at d-1,
              actual written at end of d.
  stockouts : store, day, recorded_at (= day)  -- the store ran out that day (sales are capped).
Forecast task: at the end of day t-1, predict final units of day t for every store.
"""
import numpy as np, pandas as pd

# name, seed, n_stores, n_hist, completeness by revision (last = final), flash share, fcst sd, level shock sd,
# (share of stores that report late, their first-report delay in days), hidden
SETTINGS = [
    ("dev_a",        11, 20, 420, [0.80, 0.95, 1.0],             0.15, 1.5, 0.06, (0.25, 1), False),
    ("dev_b",        12, 16, 380, [0.72, 0.90, 0.97, 1.0],       0.25, 2.0, 0.08, (0.00, 0), False),
    ("hid_slow",    111, 24, 400, [0.55, 0.75, 0.88, 0.95, 1.0], 0.20, 1.5, 0.07, (0.30, 1), True),
    ("hid_returns", 112, 18, 450, [1.15, 1.07, 1.0],             0.10, 1.0, 0.06, (0.20, 1), True),
    ("hid_late",    113, 20, 400, [0.85, 0.96, 1.0],             0.30, 2.0, 0.08, (0.50, 2), True),
    ("hid_shocky",  114, 30, 360, [0.65, 0.90, 1.0],             0.20, 1.5, 0.14, (0.25, 1), True),
]
N_LIVE = 60


def simulate(seed, n_stores, n_days, comp, flash, fsd, lsd, late):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(np.log(120), 0.5, n_stores)
    wk = np.exp(rng.normal(0, 0.15, (n_stores, 7))); wk[:, 5:] *= 1.3
    t = np.arange(n_days)
    lvl = np.zeros((n_stores, n_days))
    for d in range(1, n_days): lvl[:, d] = 0.96 * lvl[:, d - 1] + rng.normal(0, lsd, n_stores)
    temp = 15 + 10 * np.sin(2 * np.pi * (t - 100) / 365) + rng.normal(0, 3, (n_stores, n_days))
    fcst = temp + rng.normal(0, fsd, temp.shape)
    disc = np.zeros((n_stores, n_days)); planned = np.full((n_stores, n_days), -1)
    for s in range(n_stores):
        d = int(rng.integers(0, 8))
        while d < n_days:
            disc[s, d] = rng.choice([0.1, 0.2, 0.3])
            planned[s, d] = d if rng.random() < flash else d - int(rng.integers(2, 22))
            d += int(rng.integers(5, 14))
    mu = (base[:, None] * wk[:, t % 7] * (1 + 0.25 * np.sin(2 * np.pi * t / 365)) * np.exp(lvl)
          * (1 + 2.0 * disc) * np.exp(0.03 * (temp - 15)))
    units = rng.poisson(mu).astype(float)
    so = rng.random(units.shape) < 0.03
    units[so] = np.floor(units[so] * rng.uniform(0.3, 0.8, so.sum()))
    S, D = np.meshgrid(np.arange(n_stores), t, indexing="ij")
    delay = np.where(rng.random(n_stores) < late[0], late[1], 0)[:, None] + 0 * D
    logs = []
    for a, c in enumerate(comp):
        u = units if a == len(comp) - 1 else np.round(units * c * rng.normal(1, 0.02, units.shape))
        logs.append(pd.DataFrame(dict(store=S.ravel(), day=D.ravel(), units=u.ravel(), recorded_at=(D + delay + a).ravel())))
    sales_log = pd.concat(logs, ignore_index=True)
    m = disc > 0
    promos = pd.DataFrame(dict(store=S[m], day=D[m], discount=disc[m], planned_at=planned[m]))
    weather = pd.concat([pd.DataFrame(dict(store=S.ravel(), day=D.ravel(), kind="forecast", temp=fcst.ravel().round(1), recorded_at=(D - 1).ravel())),
                         pd.DataFrame(dict(store=S.ravel(), day=D.ravel(), kind="actual", temp=temp.ravel().round(1), recorded_at=D.ravel()))],
                        ignore_index=True)
    stockouts = pd.DataFrame(dict(store=S[so], day=D[so], recorded_at=D[so]))
    tables = dict(sales_log=sales_log, promos=promos, weather=weather, stockouts=stockouts)
    # information-fair one-step-ahead oracle: true state up to d-1, forecast temperature, planned promos only
    lvl1 = np.concatenate([np.zeros((n_stores, 1)), 0.96 * lvl[:, :-1]], 1)
    known = (planned >= 0) & (planned < np.arange(n_days)[None, :])
    flash_rate = ((planned >= 0) & ~known).sum() / max(1, (~known).sum())
    pro = np.where(known, 1 + 2.0 * disc, 1 + 2.0 * 0.2 * flash_rate)
    mu1 = (base[:, None] * wk[:, t % 7] * (1 + 0.25 * np.sin(2 * np.pi * t / 365)) * np.exp(lvl1 + lsd ** 2 / 2)
           * pro * np.exp(0.03 * (fcst - 15)) * 0.985)
    final = pd.DataFrame(dict(store=S.ravel(), day=D.ravel(), units=units.ravel(), oracle=mu1.ravel()))
    return tables, final


def as_of(tables, t_end):
    """the warehouse exactly as it looked at the end of day t_end (only rows written by then)"""
    return dict(sales_log=tables["sales_log"][tables["sales_log"].recorded_at <= t_end].reset_index(drop=True),
                promos=tables["promos"][tables["promos"].planned_at <= t_end].reset_index(drop=True),
                weather=tables["weather"][tables["weather"].recorded_at <= t_end].reset_index(drop=True),
                stockouts=tables["stockouts"][tables["stockouts"].recorded_at <= t_end].reset_index(drop=True))


def build_setting(row):
    name, seed, ns, nh, comp, flash, fsd, lsd, late, hidden = row
    tables, final = simulate(seed, ns, nh + N_LIVE, comp, flash, fsd, lsd, late)
    dump = as_of(tables, nh - 1)                       # the training-time warehouse dump
    oracle = final[final.day >= nh].sort_values(["day", "store"]).oracle.to_numpy()
    train_rows = final[(final.day >= 28) & (final.day <= nh - 1 - len(comp) - late[1])][["store", "day", "units"]].reset_index(drop=True)  # final labels only
    live_days = list(range(nh, nh + N_LIVE))
    return dict(name=name, hidden=hidden, dump=dump, train_rows=train_rows, tables=tables, final=final,
                live_days=live_days, oracle=oracle)
