"""AV compiler (goal-3 prototype): mechanism cards x assumption-violation templates -> certified items.

Zero model calls.  The roles of the item-production loop are played by code:
  proposer     templates.PRIOR (one-shot) and templates.REPAIR (loop): reads the gate diagnostic and edits the design
  gates        evaluate(): G1 fairness ... G9 red team (see GATES below)
  proposer'    run_cell_search(): the gates themselves used as a cheap oracle (random draws + hill-climbing per world)
  red team     the `red` routes of each template (added after certification, to count false certifications)
  analyst      attributability + driver analysis (which world constant sets the nearest shortcut's bias) -> analyze.py
  curator      signature dedup, cross-card collapse, realism flags -> analyze.py
  auditor      render.py: items rendered as the agent would see them (found the shown-number leak; audit_leak.py)

usage: python3 compiler.py --modes oneshot,loop,search,search_t1 [--evals 40] [--worlds 48] [--seed 0]
       writes results/*.json; then python3 analyze.py (MATRIX.md) and python3 render.py (items/examples.md)
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cards import CARDS                       # noqa: E402
from routes_v5 import certify, census         # noqa: E402
from templates import TEMPLATES, PRIOR, BUILD, REPAIR, mutate   # noqa: E402

R_CAL, R_HOLD, N_BOOT = 400, 400, 200
N_CAP = 400
B_FLOOR = 3.0
KAPPA_MAX = 3.0
TIGHT_REL, TIGHT_ABS = 0.05, 0.05
REL_FLOOR, ABS_FLOOR = 0.001, 0.001       # grading floor on T: 0.1% of a level (4 sig. figs), 0.001 of a fraction
BIAS_BUY = 1.5                            # G3: the reference's bias may inflate T by at most 50% (v5 G17)
GATES = ["G0_build", "G1_budget", "G2_fair", "G3_oracle_bias", "G4_tight", "G5_B", "G6_kappa", "G7_free",
         "G8_census", "G9_redteam"]
GATE_DOC = {
    "G0_build": "design is inside the card's axis and the estimand is identified",
    "G1_budget": "reference uses <= N_CAP runs",
    "G2_fair": "reference estimator lands within T on >= 95% of HELD-OUT lab seeds (T calibrated on other seeds)",
    "G3_oracle_bias": "T <= 1.5 x T computed from the de-biased reference errors (the reference's bias must not buy tolerance, v5 G17)",
    "G4_tight": "T <= 5% of |truth| (levels) or <= 0.05 absolute (fractions)",
    "G5_B": "B >= 3 at the 97.5% bootstrap-upper T (B_lo), over base routes, reference/alt/valid routes excluded",
    "G6_kappa": "kappa <= 3 (reference interpolates / averages / selects; no extrapolation amplification)",
    "G7_free": "every free-data honest estimate is >= 2 T from the truth (the item cannot be done without buying), "
               "and every salient number the task shows in the answer's units is >= 1 T from the truth",
    "G8_census": "G18 (>=2 numerical + >=1 structural family) for off-grid targets; else >=2 families incl. the AV family",
    "G9_redteam": "B_lo >= 3 still holds after the red team's extra routes are added",
}


def evaluate(card, tpl, th, d, seed):
    """Build and gate one item.  Returns a result dict; never raises on a bad design."""
    item, why = BUILD[tpl](card, th, d, seed)
    res = dict(card=card.key, tpl=tpl, design=d, seed=seed, gates={}, first_fail=None, nearest=None,
               noise_limited=False)
    if item is None:
        res["gates"]["G0_build"] = False
        res["first_fail"] = "G0_build"
        res["why"] = why
        return res, None
    res["gates"]["G0_build"] = True
    truth = item["truth"]
    rng = np.random.default_rng(10_000 + seed)
    est = item["ref"](rng, R_CAL)
    err = est - truth
    ok = np.isfinite(err)
    ae = np.where(ok, np.abs(err), np.inf)           # an unidentified estimate (all-censored median) is a miss
    p90 = float(np.quantile(ae, 0.9))
    floor_abs = REL_FLOOR * abs(truth) if item["scale"] == "rel" else ABS_FLOOR
    T = max(floor_abs, 2.25 * p90)
    bs = np.array([np.quantile(rng.choice(ae, R_CAL), 0.9) for _ in range(N_BOOT)])
    T_hi = max(floor_abs, 2.25 * float(np.quantile(bs, 0.975)))
    hold = item["ref"](np.random.default_rng(20_000 + seed), R_HOLD)
    fair = float(np.mean(np.abs(hold - truth) <= T))
    bias = float(np.mean(err[ok])) if ok.any() else float("inf")
    T_nobias = max(floor_abs, 2.25 * float(np.quantile(np.where(ok, np.abs(err - bias), np.inf), 0.9)))
    res.update(truth=truth, T=T, T_hi=T_hi, fair=fair, oracle_bias_share=abs(bias) / max(2.25 * p90, 1e-300),
               bias_buy=T / T_nobias if math.isfinite(T_nobias) else float("inf"), frac_unidentified=float(1 - ok.mean()),
               kappa=item["kappa"], n=item["n"], eff=float(np.std(est) / item["sd_ideal"]),
               T_at_floor=bool(T <= floor_abs * 1.0000001), spec=item["spec"])
    res["noise_limited"] = bool(res["oracle_bias_share"] < 0.5 and not res["T_at_floor"])
    # ---- alternative readings: add the clause when a reading lands > T away; exclude alt values from B
    clauses = [c for (v, c) in item["alts"].values() if abs(v - truth) > T]
    res["clauses_added"] = clauses
    base = dict(item["routes"])
    cert = certify(truth, T_hi, base, floor=B_FLOOR, reference=item["reference"])
    cert_mid = certify(truth, T, base, floor=B_FLOOR, reference=item["reference"])
    res.update(B_lo=cert["B"], B=cert_mid["B"], nearest=cert["nearest"],
               table=[dict(route=r["route"], over_T=round(r["over_T"], 2)) for r in cert_mid["table"]])
    allr = dict(base, **item["red"])
    cert_red = certify(truth, T_hi, allr, floor=B_FLOOR, reference=item["reference"])
    res.update(B_lo_red=cert_red["B"], nearest_red=cert_red["nearest"])
    cen = census({k: v for k, v in base.items() if k not in item["reference"]})
    res["census"] = cen
    if item["census_rule"] == "G18":
        cen_ok = len(cen["numerical"]) >= 2 and len(cen["structural"]) >= 1
    else:
        fams = set(cen["numerical"]) | set(cen["structural"])
        cen_ok = len(fams) >= 2 and any(f.startswith(item["av_fams"]) for f in fams)
    free_over_T = {k: abs(v - truth) / T for k, v in item["free"].items()}
    res["free_over_T"] = {k: round(v, 2) for k, v in free_over_T.items()}
    shown_over_T = {k: abs(v - truth) / T for k, v in item.get("shown", {}).items()}
    res["shown_over_T"] = {k: round(v, 2) for k, v in shown_over_T.items()}
    res["valid_over_T"] = {k: round(abs(v - truth) / T, 3) for k, v in item["valid"].items()}
    # attributability: can an analyst tell which rival an off-by-shortcut answer came from?
    vals = sorted([v for k, v in allr.items() if k not in item["reference"]] + [truth])
    sep = np.diff(vals) / T if len(vals) > 1 else np.array([np.inf])
    res["min_route_sep_over_T"] = float(np.min(sep))
    g = res["gates"]
    g["G1_budget"] = item["n"] <= N_CAP
    g["G2_fair"] = fair >= 0.95
    g["G3_oracle_bias"] = res["bias_buy"] <= BIAS_BUY
    g["G4_tight"] = T <= (TIGHT_REL * abs(truth) if item["scale"] == "rel" else TIGHT_ABS)
    g["G5_B"] = cert["B"] >= B_FLOOR
    g["G6_kappa"] = item["kappa"] <= KAPPA_MAX
    g["G7_free"] = all(v >= 2.0 for v in free_over_T.values()) and all(v >= 1.0 for v in shown_over_T.values())
    g["G8_census"] = bool(cen_ok)
    g["G9_redteam"] = cert_red["B"] >= B_FLOOR
    for k in GATES:
        if not g.get(k, True):
            res["first_fail"] = k
            break
    if res["first_fail"] == "G9_redteam":
        res["nearest"] = cert_red["nearest"]
    res["certified"] = res["first_fail"] is None
    res["certified_without_redteam"] = all(g[k] for k in GATES if k != "G9_redteam")
    return res, item


def driver(card, tpl, th, d, seed, res, tier=2):
    """Which world constant sets the binding shortcut's bias?  Perturb each key by 10% of its prior range
    (same design, same random draws) and measure |d(route - truth)| / T for the item's nearest route
    (over base + red-team rivals for tier 2, base rivals only for tier 1)."""
    name = res["nearest_red"] if tier == 2 else res["nearest"]
    item0, _ = BUILD[tpl](card, th, d, seed)
    allr0 = dict(item0["routes"], **item0["red"])
    b0 = allr0[name] - item0["truth"]
    sens = {}
    for k, (lo, hi) in card.prior.items():
        t2 = dict(th)
        t2[k] = th[k] + 0.1 * (hi - lo)
        it, _ = BUILD[tpl](card, t2, d, seed)
        if it is None:
            continue
        a = dict(it["routes"], **it["red"])
        if name in a:
            sens[k] = abs((a[name] - it["truth"]) - b0) / res["T"]
    if not sens:
        return None, "none", {}
    k = max(sens, key=sens.get)
    cls = "stat" if k in card.stat_keys else "mechanism"
    return k, cls, {kk: round(v, 3) for kk, v in sorted(sens.items(), key=lambda x: -x[1])[:4]}


MAX_CHAIN = 5


def run_cell(card, tpl, mode, evals, seed0):
    """mode 'oneshot': `evals` independent proposals (world + design drawn from the prior).
    mode 'loop': chains of up to MAX_CHAIN evaluations; after each failure the proposer edits the design from the
    first failing gate's diagnostic (templates.REPAIR), or resamples the world, until `evals` evaluations are spent.
    Both modes use the same RNG stream, so the first proposal of every chain is drawn the same way."""
    rng = np.random.default_rng(zlib.crc32(("%s|%s|%d" % (card.key, tpl, seed0)).encode()))
    log, trace = [], []
    spent, chain_id = 0, 0
    while spent < evals:
        th = card.sample(rng)
        d = PRIOR[tpl](card, rng)
        wseed = int(rng.integers(1 << 30))
        steps = []
        for step in range(MAX_CHAIN if mode == "loop" else 1):
            if spent >= evals:
                break
            res, item = evaluate(card, tpl, th, d, wseed)
            spent += 1
            res["chain"], res["step"] = chain_id, step
            steps.append(dict(step=step, first_fail=res["first_fail"], nearest=res.get("nearest"),
                              B_lo=None if res.get("B_lo") is None else round(res["B_lo"], 2), design=dict(d)))
            if res.get("certified"):
                k, cls, sens = driver(card, tpl, th, d, wseed, res)
                res.update(driver_key=k, driver_class=cls, driver_sens=sens, world=th)
                log.append(res)
                break
            log.append(res)
            if res["first_fail"] == "G0_build" and "N/A" in res.get("why", ""):
                trace.append(dict(chain=chain_id, steps=steps))
                return log, trace, spent, True
            if mode != "loop":
                break
            if res["first_fail"] == "G0_build":
                nd, why = PRIOR[tpl](card, rng), "design invalid -> proposer redraws the design"
            else:
                nd, why = REPAIR[tpl](card, th, d, res, rng)
            steps[-1]["action"] = why
            if nd is None:
                break
            if isinstance(nd, str) and nd == "RESAMPLE":
                th = card.sample(rng)
                wseed = int(rng.integers(1 << 30))
            else:
                d = nd
        trace.append(dict(chain=chain_id, steps=steps))
        chain_id += 1
    return log, trace, spent, False


def realism_flags(r):
    d, f = r["design"], []
    if d.get("r_nb", 1.0) >= 4:
        f.append("notebook noise x%.1f" % d["r_nb"])
    if r.get("n", 0) >= 300:
        f.append("n=%d runs" % r["n"])
    return f


def _cert(res, tier):
    return res.get("certified") if tier == 2 else res.get("certified_without_redteam")


def score(res, tier=2):
    """Search objective, lexicographic: certified-and-realistic > certified-but-flagged > gates passed in order,
    with the certificate margin as the tie-breaker (capped at B = 8 so the search does not chase huge B).
    tier=1 certifies against the template's base rivals only (G9 red team ignored)."""
    if _cert(res, tier):
        base = 15.0 if realism_flags(res) else 20.0
        return base + min(math.log2(max(res["B_lo_red"] if tier == 2 else res["B_lo"], 1.0)), 3.0) / 3.0
    ff = res["first_fail"]
    if tier == 1 and ff == "G9_redteam":        # cannot happen when not certified_without_redteam; guard anyway
        ff = "G5_B"
    i = GATES.index(ff)
    part = 0.0
    if res["first_fail"] in ("G5_B", "G9_redteam"):
        part = min(res["B_lo_red"] if res["first_fail"] == "G9_redteam" else res["B_lo"], B_FLOOR) / B_FLOOR
    elif res["first_fail"] == "G7_free" and (res.get("free_over_T") or res.get("shown_over_T")):
        part = min([v / 2.0 for v in res.get("free_over_T", {}).values()] + [v for v in res.get("shown_over_T", {}).values()] + [1.0])
    elif res["first_fail"] == "G3_oracle_bias":
        part = min(BIAS_BUY / max(res["bias_buy"], 1e-9), 1.0)
    return i + 0.99 * part


def run_cell_search(card, tpl, worlds, probes, seed0, n_random=12, tier=2):
    """Driver-aware proposer, zero model calls: for each sampled world, search the template's design box
    (random prior draws, then hill-climbing with templates.mutate) using the gates themselves as a cheap oracle.
    Returns one record per WORLD: the best item found, so 'fertility' = fraction of worlds with a certifiable design."""
    # two streams: worlds (identical for both tiers, so tier-1 vs tier-2 fertility is a PAIRED comparison) and
    # proposals (tier-specific).  A single stream would desynchronise the worlds after the first early stop.
    rng_w = np.random.default_rng(zlib.crc32(("world|%s|%s|%d" % (card.key, tpl, seed0)).encode()))
    rng = np.random.default_rng(zlib.crc32(("search|%s|%s|%d|%d" % (card.key, tpl, seed0, tier)).encode()))
    out = []
    for w in range(worlds):
        th = card.sample(rng_w)
        wseed = int(rng_w.integers(1 << 30))
        best, best_s, used, first_cert = None, -1.0, 0, None
        for p in range(probes):
            if p < n_random or best is None:
                d = PRIOR[tpl](card, rng)
            else:
                d = mutate(card, tpl, best["design"], rng, k_moves=int(rng.choice([1, 1, 2])))
            res, _ = evaluate(card, tpl, th, d, wseed)
            used += 1
            if res["first_fail"] == "G0_build" and "N/A" in res.get("why", ""):
                return [dict(res, world=th, probes=used)], True
            s = score(res, tier)
            if first_cert is None and _cert(res, tier):
                first_cert = used
            if s >= best_s:
                best, best_s = res, s
            if _cert(best, tier) and not realism_flags(best) and (best["B_lo_red"] if tier == 2 else best["B_lo"]) >= 6:
                break
        best = dict(best, world=th, probes=used, first_cert_probe=first_cert, world_idx=w, tier=tier,
                    tier_cert=bool(_cert(best, tier)))
        if _cert(best, tier):
            k, cls, sens = driver(card, tpl, th, best["design"], wseed, best, tier)
            best.update(driver_key=k, driver_class=cls, driver_sens=sens, realism_flags=realism_flags(best))
        out.append(best)
    return out, False


def _js(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _slim(r):
    r = {k: v for k, v in r.items() if k != "table"} | {"table": r.get("table", [])[:6]}
    if r.get("certified") and "realism_flags" not in r:
        r["realism_flags"] = realism_flags(r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=40, help="oneshot / loop: evaluations per cell")
    ap.add_argument("--worlds", type=int, default=48, help="search: worlds per cell")
    ap.add_argument("--probes", type=int, default=48, help="search: gate evaluations per world")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cards", default="")
    ap.add_argument("--tpls", default="")
    ap.add_argument("--modes", default="oneshot,loop,search")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    cards = [c for c in CARDS if not a.cards or c.key in a.cards.split(",")]
    tpls = [t for t in TEMPLATES if not a.tpls or t in a.tpls.split(",")]
    for card in cards:
        for tpl in tpls:
            for mode in a.modes.split(","):
                t0 = time.time()
                fn = os.path.join(HERE, "results", "%s_%s_%s.json" % (card.key, tpl, mode))
                if mode.startswith("search"):
                    out, na = run_cell_search(card, tpl, a.worlds, a.probes, a.seed, tier=1 if mode == "search_t1" else 2)
                    trace, spent = [], sum(r["probes"] for r in out)
                else:
                    out, trace, spent, na = run_cell(card, tpl, mode, a.evals, a.seed)
                json.dump(dict(card=card.key, tpl=tpl, mode=mode, spent=spent, na=na, results=[_slim(r) for r in out],
                               trace=trace), open(fn, "w"), indent=1, default=_js)
                key = "certified_without_redteam" if mode == "search_t1" else "certified"
                nc = sum(1 for r in out if r.get(key))
                ncl = sum(1 for r in out if r.get(key) and not realism_flags(r))
                print("%-13s %s %-7s spent=%4d certified=%3d clean=%3d  %.1fs%s" % (
                    card.key, tpl, mode, spent, nc, ncl, time.time() - t0, "  N/A" if na else ""), flush=True)


if __name__ == "__main__":
    main()
