"""E01 release gap: an eval team is about to publish "M_new beats M_ref by several points".  Four separate
things are wrong with that sentence, and each one is a measurement the agent has to make.

(O4 measurement artifact, O16 denominator swap that changes the sign, O10 aggregation, O5 non-identifiability.)

The team's memo reports `acc` exactly as the service returns it: correct / *extractable* items.  The release
harness counts an unparseable answer as wrong, and M_new's answers are harder to parse than M_ref's (E2,
`ext_model_sd`), so the convention is not a rounding detail - it moves each model by a different amount
(O16).  A quarter of the bank occurs in the pretraining corpus and M_new memorised some of it (E3); the
team's evidence against contamination is the *cheap* detector, whose score is a monotone function of the
occurrence count plus item-level noise (E8), so thresholding it neither finds the duplicated items nor
bounds their effect - only the expensive exact scan does.  The memo's error bar is the binomial one; the
service also has a between-call shift shared by every item in a call (E6), which the binomial formula
cannot see and which no number of items removes, only repetitions (Miller 2024).  Finally the release
report will weight slice 0 by a factor the product team has not fixed (known unknown, zero footprint), and
the two models' slice profiles differ (E7 `slice_off_sd`), so the headline comparison is genuinely
undetermined.

Note on E7: this task does *not* use aggregation as a difficulty lever - the card's own grounding records
that micro-vs-macro is a weak one.  The weighting enters as an *ignorance* mechanism: an undecided release
weight turns one question into a set and one comparison into `undetermined`.  The strong lever here is the
denominator, exactly as the card says.

Useful designs: `corpus mode=scan` over the whole bank (the only sound contamination measurement);
`detail=1` rows, which give per-item outcomes and therefore a clean-minus-all difference measured *inside*
one call, where the shared between-call shift cancels; one high-`reps` row, whose `reps_acc` is a direct
sample of the quantity the binomial formula gets wrong; per-slice rows for the reweighting.  `by_slice` and
`detail` are first-repetition only, so per-item resolution costs one request per replicate while a mean
does not.
"""
import json, math
import numpy as np
from .. import queries as Q
from ..common import draw_card, run_rows
from ..labs import evallab as EV

ID = "e01-release-gap"
TITLE = "Release gap: the denominator, the contamination, the error bar and the weighting of one eval report"
CARDS = ["E1", "E2", "E3", "E6", "E7", "E8"]
OBSTACLES = ["O4", "O16", "O10", "O5"]
CLAIM = ("hold an evaluation report to the convention it will be released under: separate the reported "
         "denominator from the quantity of interest, measure contamination instead of trusting a detector "
         "score, price the between-call variance a binomial error bar omits, and report a "
         "weighting-dependent comparison as undetermined")
DIFFICULTY = {"depth": 3, "nuisance": ["E2", "E7", "E8"], "anti_prior": ["q4"]}
EXEMPT_LOAD_BEARING = {
    "E1": "the bank's difficulty/discrimination distribution is the substrate every key is computed on, and "
          "the bank is generated from those constants: neutralising a_mu/b_sd regenerates a *different* "
          "benchmark, so a drop rival would compare two banks rather than test a shortcut.  Load-bearing by "
          "construction, not testable by neutralisation (declared exemption).",
    "E8": "the detector's calibration enters no key - it is the instrument the shortcut misuses, so "
          "neutralising it changes nothing and drop:E8 cannot kill.  Its load-bearingness is certified by "
          "G12 instead, through naive_ignore:E8 (declared exemption).",
}
CERT_N = 250                     # see build.identifiability_certificate: one request here can be 150k calls

N_ITEMS, N_SLICES = 1500, 3
FMT, FMT2 = "mc_letter", "mc_cloze"
NEW, REF = "M_new", "M_ref"
MODELS = (NEW, REF)
W0 = (0.15, 0.85)                # known unknown: the release report's weight on slice 0
W_GRID = 41
MODEL_COST = 0.05
SCAN_COST, FAST_COST = 1.2, 0.03
DET_THRESH = 0.5                 # the threshold the team's memo uses on the cheap detector score
BAND = 0.025                     # q5 must be this far from a tie at both ends of the w0 range (~3.5 sigma)
R_ALL, R_SL, R_DET = 100, 100, 12  # oracle reps: pooled row, each slice row, detail=1 replicates per model


# ------------------------------------------------------------------------------------------ world
def _set(p, path, v):
    node = p
    parts = path.split(".")
    for s in parts[:-1]:
        node = node[s]
    node[parts[-1]] = v
    return p


def _centred(rng, sd, n):
    """A spread with mean zero, so a card's spread parameter moves the *profile* and not the overall level
    (otherwise slice_off_sd would double as an ability knob and E7 would be confounded with E1)."""
    x = rng.normal(0.0, sd, n)
    return [float(v) for v in (x - x.mean())]


_C = np.array([2.0, -1.0, -1.0]) / math.sqrt(2.0)      # slice 0 against the rest; mean 0, sd 1
_D = np.array([0.0, 1.0, -1.0]) / math.sqrt(2.0 / 3)   # orthogonal contrast;      mean 0, sd 1


def _profile(sd, sign, phi):
    """A three-slice profile with the drawn spread `sd`, dominated by a slice-0-against-the-rest contrast.

    Three independent normals would satisfy the card just as well, but the quantity the release weighting
    moves is exactly this contrast, and an unstructured draw leaves it near zero most of the time: the
    instance is then well-formed and trivial.  The spread is the card's parameter and is preserved
    (_C and _D are orthonormal in the mean-zero subspace), only its direction is chosen.
    """
    return [float(x) for x in sd * (math.cos(phi) * sign * _C + math.sin(phi) * _D)]


def _slice_clean(pf, m, ok, g, clean):
    """Release-convention accuracy on the uncontaminated items of each slice, for a given ability."""
    idx = np.arange(int(pf["n_items"]))
    pc = np.zeros(len(idx))
    for x, wt in zip(EV._GH_X, EV._GH_W):
        pc += wt * EV.p_correct(pf, m, FMT, idx, dtheta=pf["sig_call"] * x)
    return [_acc(ok, pc, clean & (g == s)) for s in range(N_SLICES)]


def draw(rng):
    why = "no draw attempted"
    for _ in range(300):
        p = {"n_items": N_ITEMS, "n_slices": N_SLICES, "bank_seed": int(rng.integers(1, 10 ** 6)),
             "det_fast_cost": FAST_COST, "det_scan_cost": SCAN_COST}
        p.update(draw_card(rng, "E1", {"b_mu": (-0.35, 0.35)}))
        e2 = draw_card(rng, "E2", {"ext_base": (0.07, 0.13), "ext_slice_sd": (0.02, 0.06),
                                   "ext_model_sd": (0.10, 0.15), "fmt_off_sd": (0.10, 0.35)})
        e3 = draw_card(rng, "E3", {"dup_frac": (0.22, 0.34), "dup_lam": (4.0, 9.0), "kappa": (0.65, 1.15)})
        e6 = draw_card(rng, "E6", {"sig_call": (0.16, 0.22)})
        e7 = draw_card(rng, "E7", {"slice_b_sd": (0.35, 0.70), "slice_off_sd": (0.0, 0.75)})
        # E8 is drawn at its lossy end: the cheap n-gram detector fires on a small minority of the
        # duplicated items, which is the published failure mode (paraphrased or reformatted contamination
        # passes a surface detector).  A detector that caught most of it would make the full scan optional,
        # and q2 would stop separating "ran the scan" from "trusted the screen".
        p.update(draw_card(rng, "E8", {"det_a": (0.6, 1.1), "det_b": (-3.2, -2.5), "det_s": (0.8, 1.3)}))
        p["dup_frac"], p["dup_lam"] = e3["dup_frac"], e3["dup_lam"]
        p["sig_call"] = e6["sig_call"]
        # slice_b is common to both models, so it widens q3's set without touching q5's comparison;
        # the per-model slice_off below is the opposite - differential, so it drives q5 and cancels out of
        # nothing.  The two are tuned separately because they carry different items.
        p["slice_b"] = _profile(e7["slice_b_sd"], 1.0 if rng.random() < 0.5 else -1.0, rng.uniform(0.0, 0.55))
        # E2: the extraction-failure rate is a property of (format, slice, model), not of the format alone.
        # `by_slice` is an offset the backend adds to `base`, so it is drawn centred.
        p["fmt_ext"] = {}
        for f, scale in ((FMT, 1.0), (FMT2, float(rng.uniform(0.3, 0.8)))):
            base = e2["ext_base"] * scale
            off = _centred(rng, e2["ext_slice_sd"] * scale, N_SLICES)
            p["fmt_ext"][f] = {"base": base, "by_slice": [max(d, 0.02 - base) for d in off]}
        # M_new is weaker on slice 0 and stronger on the rest, so the pooled comparison the memo makes
        # favours it while the release weighting, which leans on slice 0, need not.
        th = p["b_mu"] + float(rng.uniform(-0.25, 0.25))
        phi_n, phi_r = float(rng.uniform(0.0, 0.55)), float(rng.uniform(0.0, 0.55))
        p["models"] = {
            NEW: dict(theta=th, kappa=e3["kappa"], ext=e2["ext_model_sd"], cost=MODEL_COST, length=300.0,
                      fmt={FMT: 0.0, FMT2: float(rng.normal(0.0, e2["fmt_off_sd"]))},
                      slice_off=_profile(0.3, -1.0, phi_n)),
            REF: dict(theta=th - 0.2, kappa=0.0, ext=0.0, cost=MODEL_COST, length=300.0,
                      fmt={FMT: 0.0, FMT2: float(rng.normal(0.0, e2["fmt_off_sd"]))},
                      slice_off=_profile(0.3, +1.0, phi_r))}
        # (M_ref was frozen before the bank was published: kappa = 0, and its answers parse cleanly.)
        pf = EV.full(p)
        _a, _b, g, dup = EV.bank(pf)
        clean = dup == 0
        ok_new = EV.extract_ok(pf, NEW, FMT, np.arange(N_ITEMS))
        ok_ref = EV.extract_ok(pf, REF, FMT, np.arange(N_ITEMS))

        def endpoints():
            """The release-convention gap at the two ends of the w0 range.  _mix is affine in w0, so these
            two numbers determine the whole line: its swing, and where (if anywhere) it crosses zero."""
            dn = _slice_clean(pf, NEW, ok_new, g, clean)
            dr = _slice_clean(pf, REF, ok_ref, g, clean)
            return (_mix(dn, W0[0]) - _mix(dr, W0[0]), _mix(dn, W0[1]) - _mix(dr, W0[1]))

        def set_scale(s):
            pf["models"][NEW]["slice_off"] = _profile(s, -1.0, phi_n)
            pf["models"][REF]["slice_off"] = _profile(s, +1.0, phi_r)

        # 1) SOLVE for the size of the differential slice profile.  q5 is undetermined only if the gap
        # straddles zero across the w0 range, and each chain step kills q5 only if that step's shift moves
        # the whole line to one side.  A shift is a level; a straddle is a swing; so the swing has to be
        # *smaller* than the shifts, not merely non-zero.  Drawing slice_off_sd and hoping gave 0 admissible
        # worlds in 10 draws, because a free draw of this spread overshoots the window by 3x.
        swing_t = float(rng.uniform(0.060, 0.072))
        lo, hi = 0.0, 1.5
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            set_scale(mid)
            d0, d1 = endpoints()
            if d0 - d1 > swing_t:
                hi = mid
            else:
                lo = mid
        scale = 0.5 * (lo + hi)
        set_scale(scale)
        d0, d1 = endpoints()
        if not 0.02 <= scale <= 0.75 or abs((d0 - d1) - swing_t) > 0.2 * swing_t:
            why = "slice profile %.3f gives swing %.4f, wanted %.4f" % (scale, d0 - d1, swing_t)
            continue
        p["models"][NEW]["slice_off"] = list(pf["models"][NEW]["slice_off"])
        p["models"][REF]["slice_off"] = list(pf["models"][REF]["slice_off"])

        # 2) SOLVE for the reference model's ability, so the line crosses zero at an interior weight: the
        # crossing is the point of the item, and it has to sit near the middle of the range or one end of
        # the straddle is too shallow for the oracle to call.
        wstar = float(rng.uniform(0.44, 0.60))
        lo, hi = 0.0, 0.9
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            pf["models"][REF]["theta"] = th - mid
            d0, d1 = endpoints()
            if _lerp(d0, d1, wstar) > 0:
                hi = mid
            else:
                lo = mid
        delta = 0.5 * (lo + hi)
        if not 0.03 <= delta <= 0.85:
            why = "ability gap %.3f outside [0.03, 0.85]" % delta
            continue
        p["models"][REF]["theta"] = th - delta
        p["w0_rel"] = float(rng.uniform(*W0))                  # scenario unknown: nothing in the lab reads it
        ok, why = _admissible(EV.full(p))
        if ok:
            return p
    raise RuntimeError("no admissible draw (last: %s)" % why)


def spec(p):
    return {"lab": "evallab",
            "knobs": {"svc": {"type": "choice", "values": ["score", "corpus"], "default": "score"},
                      "model": {"type": "choice", "values": list(MODELS), "default": NEW},
                      "split": {"type": "choice", "values": ["all", "s0", "s1", "s2"], "default": "all"},
                      "fmt": {"type": "choice", "values": [FMT, FMT2], "default": FMT},
                      "n": {"type": "float", "min": 10, "max": N_ITEMS, "int": True},
                      "reps": {"type": "float", "min": 1, "max": 400, "int": True, "default": 1},
                      "detail": {"type": "choice", "values": [0, 1], "default": 0},
                      "mode": {"type": "choice", "values": ["fast", "scan"], "default": "fast"}},
            "fixed": {},
            # The credit budget is the binding constraint and the intended one (the exact scan alone costs
            # 1800, and a 100-repetition pooled row costs 7500 of the 8000 per-request cap).  The request
            # count is deliberately loose enough that no reasonable plan dies on it alone.
            "caps": {"run_cost": 8000.0, "total_cost": 30000.0, "max_runs": 36}}


def LAB_EXTRA(w):
    return ["",
            "Models: `%s` is the release candidate, `%s` the reference the memo compares it against.  Both "
            "cost %.2g credits per item per repetition." % (NEW, REF, MODEL_COST),
            "",
            "Formats: `%s` presents the options and reads back a letter; `%s` scores the options by "
            "likelihood.  An item whose answer cannot be extracted is reported as presented (`n_items`) but "
            "not scored (`n_scored`); the reported `acc` is correct / `n_scored`." % (FMT, FMT2),
            "",
            "**The release harness that will publish these numbers counts an item whose answer cannot be "
            "extracted as wrong, over every item presented.**  The questions below say which convention they "
            "mean; the service's own `acc` is not the release number.",
            "",
            "`svc=corpus` addresses the pretraining corpus of the models, not the item bank: `mode=fast` "
            "returns a cheap n-gram `overlap` score per item (%.2g credits per item), `mode=scan` the exact "
            "`occurrences` count (%.3g credits per item).  Both are deterministic, and both join to a "
            "`score` row's `detail` table by item index." % (FAST_COST, SCAN_COST)]


def known_unknowns(p):
    return [{"name": "release weight on slice 0", "param": "w0_rel", "range": list(W0),
             "text": "The release report aggregates the three slices with weight w0 on slice 0 and the "
                     "remaining 1 - w0 split 2:1 between slice 1 and slice 2.  The product team will fix w0 "
                     "somewhere in [%.2f, %.2f] and has not done so.  Nothing in this service depends on "
                     "that choice.  Questions about the release report must cover the whole range." % W0}]


# ------------------------------------------------------------------------------------------ quantities
def _weights(w0):
    return [w0, (1.0 - w0) * 2.0 / 3.0, (1.0 - w0) / 3.0]


def _mix(per_slice, w0):
    return float(sum(w * a for w, a in zip(_weights(w0), per_slice)))


def _grid():
    return list(np.linspace(W0[0], W0[1], W_GRID))


def _lerp(v_lo, v_hi, w0):
    """Value at weight w0 of a quantity known at the two ends of the w0 range.  _weights is affine in w0
    and so is every mixture of fixed per-slice numbers, so two endpoints determine the line exactly."""
    return v_lo + (v_hi - v_lo) * (w0 - W0[0]) / (W0[1] - W0[0])


def _tables(p, m, fmt=FMT):
    """Per-item extraction success and per-item probability correct, over the whole bank, once per model.
    `subset` only permutes, so a mask over the natural order is the same set as any split."""
    idx = np.arange(int(p["n_items"]))
    ok = EV.extract_ok(p, m, fmt, idx)
    pc = np.zeros(len(idx))
    for x, wt in zip(EV._GH_X, EV._GH_W):                  # average over the between-call shift
        pc += wt * EV.p_correct(p, m, fmt, idx, dtheta=p["sig_call"] * x)
    return ok, pc


def _acc(ok, pc, mask, denom="release"):
    """denom='release': an unextractable answer counts wrong, over every item presented.
       denom='scored' : the service's own denominator (extractable items only)."""
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    sel = mask & ok
    if not sel.any():
        return 0.0 if denom == "release" else float("nan")
    return float(pc[sel].sum()) / (n if denom == "release" else int(sel.sum()))


def _quantities(p):
    """Every accuracy any item or rival needs, noise-free from the world."""
    a, b, g, dup = EV.bank(p)
    ovl = EV.det_fast(p, np.arange(int(p["n_items"])))
    masks = {"all": np.ones(int(p["n_items"]), dtype=bool),
             "clean": dup == 0,                            # what the exact scan finds
             "det": ovl < DET_THRESH}                      # what the cheap detector calls clean
    q = {}
    for m in MODELS:
        ok, pc = _tables(p, m)
        d = {}
        for tag, msk in masks.items():
            d[tag] = _acc(ok, pc, msk)
            d[tag + "_s"] = _acc(ok, pc, msk, "scored")
            d["sl_" + tag] = [_acc(ok, pc, msk & (g == s)) for s in range(N_SLICES)]
            d["sl_" + tag + "_s"] = [_acc(ok, pc, msk & (g == s), "scored") for s in range(N_SLICES)]
        q[m] = d
        if m == NEW:
            ns = int(ok.sum())
            acc_s = d["all_s"]
            q["n_scored"] = ns
            q["sd_binom"] = math.sqrt(max(acc_s * (1 - acc_s), 1e-12) / ns)   # the memo's error bar
    q["sd"] = float(EV.score_sd(p, NEW, "all", FMT, N_ITEMS, 1))
    return q


def _bundle(q, denom="release", clean="clean"):
    """The four fields the answers are built from, selected by convention.  A shortcut rival is exactly a
    different selection here, so every rival is the truth with one step replaced - never a noisy refit."""
    sx = "_s" if denom == "scored" else ""
    out = {"sd": q["sd"]}
    for m in MODELS:
        out[m] = {"all": q[m]["all" + sx], "clean": q[m][clean + sx],
                  "sl_all": q[m]["sl_all" + sx], "sl_clean": q[m]["sl_" + clean + sx]}
    return out


def _flags(est):
    return [_mix(est[NEW]["sl_clean"], w) - _mix(est[REF]["sl_clean"], w) > 0 for w in _grid()]


def _verdict(flags):
    return "supported" if all(flags) else ("refuted" if not any(flags) else "undetermined")


def _answers(est):
    mix_all = [_mix(est[NEW]["sl_all"], w) for w in _grid()]
    prem = est[NEW]["all"] - est[NEW]["clean"]
    return {"q1": {"lo": est[NEW]["clean"], "hi": est[NEW]["clean"]},
            "q2": {"lo": prem, "hi": prem},
            "q3": {"lo": min(mix_all), "hi": max(mix_all)},
            "q4": {"lo": est["sd"], "hi": est["sd"]},
            "q5": {"verdict": _verdict(_flags(est))}}


# ------------------------------------------------------------------------------------------ notebook
NB_REQS = [{"svc": "score", "model": NEW, "split": "all", "fmt": FMT, "n": N_ITEMS, "seed": 0},
           {"svc": "score", "model": REF, "split": "all", "fmt": FMT, "n": N_ITEMS, "seed": 0},
           {"svc": "score", "model": NEW, "split": "all", "fmt": FMT2, "n": N_ITEMS, "seed": 0},
           {"svc": "score", "model": REF, "split": "all", "fmt": FMT2, "n": N_ITEMS, "seed": 0},
           {"svc": "corpus", "mode": "scan", "n": 40, "seed": 0},
           {"svc": "corpus", "mode": "fast", "n": N_ITEMS, "seed": 0}]


def notebook(p, sess, rng):
    rows = []
    for r in NB_REQS:
        cfg, seed, ex = sess.validate(dict(r))
        rows.append(sess.execute(cfg, seed, ex))
    sc, scan40, fast = rows[:4], rows[4], rows[5]
    gap = sc[0]["acc"] - sc[1]["acc"]
    gap2 = sc[2]["acc"] - sc[3]["acc"]
    ovl = [it["overlap"] for it in fast["items"]]
    n_flag = sum(1 for v in ovl if v >= DET_THRESH)
    dup40 = sum(1 for it in scan40["items"] if it["occurrences"] > 0)
    sl = sc[0]["by_slice"]
    a = sc[0]["acc"]
    se = math.sqrt(max(a * (1 - a), 1e-12) / sc[0]["n_scored"])
    notes = """# Release review memo (eval team)

Candidate `%s` versus reference `%s`, whole bank (%d items), format `%s`, one pass each (runs 1-2 in
`notebook/runs.jsonl`):

- `%s` %.4f, `%s` %.4f -> **%+.4f**.  Binomial standard error on %d scored items is %.4f, so the gap is
  about %.1f sigma.  We are treating it as real.
- Format check (runs 3-4, `%s`): %.4f vs %.4f (%+.4f).  Same direction, so the result is not a format artifact.
- Denominator: the service could not extract an answer for %d of the %d items we presented to `%s`
  (%d of %d for `%s`).  Those are excluded from `acc`, which is the number we quote.
- Contamination: we ran the cheap n-gram detector over the whole bank (run 6, table in
  `notebook/detector_fast.csv`).  Only %d items of %d (%.1f%%) score above %.1f, and a %d-item exact scan
  (run 5) found %d of those %d items in the corpus, so we do not think contamination is driving the gap.
- Slices: `%s` scores %s on slices 0/1/2 (%d/%d/%d items presented).  The release report will weight slice 0
  by w0 and split the rest 2:1 between slices 1 and 2; product has not fixed w0 yet (see the known unknown
  in the manual).  We assume the pooled number is representative.

Open action: the release harness scores an unextractable answer as wrong, over every item presented.
Someone should check what that does to the table above before we publish.
""" % (NEW, REF, N_ITEMS, FMT,
       NEW, sc[0]["acc"], REF, sc[1]["acc"], gap, sc[0]["n_scored"], se, abs(gap) / max(se, 1e-9),
       FMT2, sc[2]["acc"], sc[3]["acc"], gap2,
       sc[0]["n_items"] - sc[0]["n_scored"], sc[0]["n_items"], NEW,
       sc[1]["n_items"] - sc[1]["n_scored"], sc[1]["n_items"], REF,
       n_flag, len(ovl), 100.0 * n_flag / float(len(ovl)), DET_THRESH, scan40["n"], dup40, scan40["n"],
       NEW, "/".join("%.4f" % (s["n_correct"] / max(s["n_scored"], 1)) for s in sl),
       sl[0]["n_items"], sl[1]["n_items"], sl[2]["n_items"])
    ctx = {"rows": rows, "gap": gap, "gap2": gap2, "se": se, "n_flag": n_flag, "dup40": dup40,
           "slice_n": [s["n_items"] for s in sl], "n_scored": [r["n_scored"] for r in sc],
           "acc": [r["acc"] for r in sc], "overlap": ovl,
           "slice_acc": [s["n_correct"] / max(s["n_scored"], 1) for s in sl]}
    return rows, notes, ctx


def files(p, ctx, rng):
    """What the agent reads of the notebook.  runs.csv keeps scalar columns only, so the rows that carry a
    table (per-slice counts, the detector scan) have to be shipped in full or the memo cites evidence the
    agent cannot see."""
    rows = ctx["rows"]
    fast = [r for r in rows if r.get("svc") == "corpus" and r.get("mode") == "fast"][0]
    csv = ["item,overlap"] + ["%d,%.6f" % (it["item"], it["overlap"]) for it in fast["items"]]
    keep = [r for r in rows if r is not fast]
    return {"notebook/runs.jsonl": "\n".join(json.dumps(r, sort_keys=True) for r in keep) + "\n",
            "notebook/detector_fast.csv": "\n".join(csv) + "\n"}


# ------------------------------------------------------------------------------------------ items
def items(p, ctx, tol=None):
    tol = tol or {}
    q = _quantities(p)
    key = _answers(_bundle(q))
    rel = ("the release convention (an item whose answer cannot be extracted counts wrong, over every item "
           "presented)")
    return [
        Q.point("q1", "Under %s, `%s`'s accuracy (a fraction in [0, 1]) on the items of the bank that do "
                      "**not** occur in the pretraining corpus, format `%s`, whole bank, noise-free "
                      "expected value." % (rel, NEW, FMT),
                "fraction", key["q1"]["lo"], tol.get("q1", 0.02), floor=0.006,
                cards=["E2", "E3"], obstacles=["O4"], chain=["denominator", "clean"]),
        Q.point("q2", "The contamination premium of `%s` under %s: its accuracy over the whole bank minus "
                      "its accuracy over the items that do not occur in the corpus (a difference of "
                      "fractions, format `%s`, noise-free expected values)." % (NEW, rel, FMT),
                "fraction", key["q2"]["lo"], tol.get("q2", 0.02), floor=0.004,
                cards=["E3", "E8"], obstacles=["O4"], chain=["clean", "detector"]),
        Q.interval("q3", "The accuracy the **release report** will publish for `%s` on the whole bank under "
                         "%s: the three slices aggregated with weight w0 on slice 0 and the remaining 1 - w0 "
                         "split 2:1 between slices 1 and 2, format `%s`.  Give the set of values this takes "
                         "over the documented range of w0 (noise-free expected values)." % (NEW, rel, FMT),
                   "fraction", key["q3"]["lo"], key["q3"]["hi"], tol.get("q3", 0.02), floor=0.006,
                   cards=["E2", "E7"], obstacles=["O10", "O5"], chain=["denominator", "slices"]),
        Q.point("q4", "The standard deviation, across repeated calls, of the `acc` that one "
                      "`svc=score model=%s split=all fmt=%s n=%d reps=1` request reports."
                      % (NEW, FMT, N_ITEMS),
                "fraction", key["q4"]["lo"], tol.get("q4", 0.01), floor=0.002,
                cards=["E6"], obstacles=["O4"],
                prior_key={"lo": q["sd_binom"], "hi": q["sd_binom"]}),
        Q.verdict("q5", "On the release report's own aggregate (weights as in q3), restricted to the items "
                        "that do not occur in the pretraining corpus and under %s, `%s` scores higher than "
                        "`%s`." % (rel, NEW, REF),
                  _flags(_bundle(q)), cards=["E2", "E3", "E7"], obstacles=["O16", "O10", "O5"],
                  chain=["denominator", "clean", "slices"]),
    ]


def answers_from(ph, ctx, collapse=False):
    out = json.loads(json.dumps(ph.get("answers", ph)))
    if collapse:
        for a in out.values():
            if isinstance(a, dict) and "lo" in a and a["hi"] > a["lo"]:
                a["lo"] = a["hi"] = 0.5 * (a["lo"] + a["hi"])
    return out


# ------------------------------------------------------------------------------------------ oracle
def oracle_design(slice_n):
    """One exact corpus scan; one high-repetition pooled row (its `reps_acc` is a direct sample of the
    quantity q4 asks about, and its mean is the pooled level); per-slice repetitions for the reweighting;
    and `detail=1` rows, whose value is that clean-minus-all is measured *inside* one call, where the
    shared between-call shift cancels.  32 requests, ~26.1k of 30k credits, peak request 7.5k of 8k.
    The detail rows are the cheapest thing on the menu (75 credits each) and the mean over them is what
    sets q1's and q2's precision, so they are bought in bulk: at 8 rows the 8-degree-of-freedom mean gave
    a t-like tail that failed G1's verification on one seed in eight."""
    reqs = [{"svc": "corpus", "mode": "scan", "n": N_ITEMS, "seed": 0},
            {"svc": "score", "model": NEW, "split": "all", "fmt": FMT, "n": N_ITEMS, "reps": R_ALL, "seed": 11}]
    for m in MODELS:
        for s in range(N_SLICES):
            reqs.append({"svc": "score", "model": m, "split": "s%d" % s, "fmt": FMT,
                         "n": int(slice_n[s]), "reps": R_SL, "seed": 20 + s})
        for k in range(R_DET):
            reqs.append({"svc": "score", "model": m, "split": "all", "fmt": FMT, "n": N_ITEMS,
                         "reps": 1, "detail": 1, "seed": 40 + k})
    return reqs


def _est_from_rows(rows):
    """Turn the oracle's rows into the same four fields `_answers` consumes."""
    scan = [r for r in rows if r.get("svc") == "corpus"][0]
    dirty = {it["item"] for it in scan["items"] if it["occurrences"] > 0}
    est = {}
    for m in MODELS:
        det = [r for r in rows if r.get("svc") == "score" and r.get("model") == m and r.get("detail")]
        pool = {"all": [], "clean": []}
        per = {"all": [[] for _ in range(N_SLICES)], "clean": [[] for _ in range(N_SLICES)]}
        for r in det:                      # release convention: unextractable counts wrong, so the
            for tag in ("all", "clean"):   # denominator is every item presented
                use = [x for x in r["detail"] if tag == "all" or x["item"] not in dirty]
                pool[tag].append(sum(1 for x in use if x["correct"]) / float(len(use)))
                for s in range(N_SLICES):
                    sub = [x for x in use if x["slice"] == s]
                    per[tag][s].append(sum(1 for x in sub if x["correct"]) / float(len(sub)))
        sl_all, sl_clean, w_g = [], [], []
        for s in range(N_SLICES):
            row = [r for r in rows if r.get("svc") == "score" and r.get("model") == m
                   and r.get("split") == "s%d" % s and not r.get("detail")][0]
            lvl = row["acc"] * row["n_scored"] / float(row["n_items"])     # scored -> release convention
            sl_all.append(lvl)
            w_g.append(row["n_items"] / float(N_ITEMS))
            sl_clean.append(lvl + float(np.mean(per["clean"][s]) - np.mean(per["all"][s])))
        e = {"sl_all": sl_all, "sl_clean": sl_clean}
        same = [r for r in rows if r.get("svc") == "score" and r.get("model") == m and r.get("fmt") == FMT
                and r.get("split") == "all" and int(r["n_items"]) == N_ITEMS]
        if m == NEW:
            # every repetition of a row at q4's own design is one draw of the quantity q4 asks about,
            # including the single-repetition detail rows, which were bought for something else
            est["sd"] = float(np.std([x for r in same for x in r["reps_acc"]], ddof=1))
        big = [r for r in same if not r.get("detail") and int(r.get("reps", 1)) > 1]
        # The slices partition the bank, so their size-weighted sum estimates the same pooled level as the
        # pooled row - but from three rows carrying three independent between-call shifts, which is the
        # dominant noise term here.  Its variance is sum(w^2) in units of one row's, so combine the two by
        # inverse variance instead of discarding either; that halves the spread of q1 and q2 for no cost.
        derived = float(sum(w * a for w, a in zip(w_g, sl_all)))
        vs = float(sum(w * w for w in w_g))
        if big:
            pooled = big[0]["acc"] * big[0]["n_scored"] / float(big[0]["n_items"])
            e["all"] = (derived / vs + pooled) / (1.0 / vs + 1.0)
        else:
            e["all"] = derived
        e["clean"] = e["all"] + float(np.mean(pool["clean"]) - np.mean(pool["all"]))
        est[m] = e
    return est


def oracle(sess, rows_nb, ctx, rng, drop=None):
    if drop:
        # G9 asks whether a mechanism is load-bearing for the *answers*.  Neutralise it in the world and
        # recompute the keys exactly: a drop rival is then noise-free, so a kill is the mechanism itself
        # and never a bad draw.
        p2 = json.loads(json.dumps(EV.full(sess.p)))
        for path, v in drop.items():
            _set(p2, path, v)
        ans = _answers(_bundle(_quantities(EV.full(p2))))
        return ans, {"answers": ans, "drop": sorted(drop)}
    rows = run_rows(sess, oracle_design(ctx["slice_n"]))
    ans = _answers(_est_from_rows(rows))
    return ans, {"answers": ans, "n_rows": len(rows)}


def cert_requests(spec, rng):
    """The structured requests the notebook and the oracle actually make, so G4 quantifies over them too."""
    return [dict(r, seed=r.get("seed", 0)) for r in NB_REQS] + oracle_design([500, 500, 500])


# ------------------------------------------------------------------------------------------ rivals
def rivals(p, rows_nb, ctx, rng):
    q = _quantities(EV.full(p))
    truth = _answers(_bundle(q))
    out = {}

    # everything right except the error bar, which is the published binomial one the memo quotes
    a = json.loads(json.dumps(truth)); a["q4"] = {"lo": q["sd_binom"], "hi": q["sd_binom"]}
    out["B_prior"] = a

    # the service's own denominator instead of the release convention (the memo's reading)
    out["skip:denominator"] = _answers(_bundle(q, denom="scored"))

    # contamination ignored: the whole bank is treated as clean, so there is no premium to report
    a = _answers(_bundle(q, clean="all")); a["q2"] = {"lo": 0.0, "hi": 0.0}
    out["skip:clean"] = a

    # the cheap detector's verdict used as the clean set, instead of the exact scan
    out["skip:detector"] = _answers(_bundle(q, clean="det"))

    # no reweighting: the pooled number is reported as if it were the release aggregate
    a = json.loads(json.dumps(truth))
    a["q3"] = {"lo": q[NEW]["all"], "hi": q[NEW]["all"]}
    a["q5"] = {"verdict": "supported" if q[NEW]["clean"] > q[REF]["clean"] else "refuted"}
    out["skip:slices"] = a

    # G12 names a nuisance card, G11 names a step; they are the same shortcut seen from two sides, so the
    # same answers are registered under both names rather than inventing a second rival for each.
    for card, step in (("E2", "denominator"), ("E7", "slices"), ("E8", "detector")):
        out["naive_ignore:" + card] = out["skip:" + step]
    return out


def rival_designs(p, rows_nb, rng):
    return {}


DROP = {"E2": {"fmt_ext": {}, "models.M_new.ext": 0.0, "models.M_ref.ext": 0.0,
               "models.M_new.fmt": {}, "models.M_ref.fmt": {}},
        "E3": {"models.M_new.kappa": 0.0, "models.M_ref.kappa": 0.0},
        "E6": {"sig_call": 0.0},
        "E7": {"slice_b": [0.0] * N_SLICES,
               "models.M_new.slice_off": [0.0] * N_SLICES, "models.M_ref.slice_off": [0.0] * N_SLICES}}
INFO_RIVALS = ()


# ------------------------------------------------------------------------------------------ posedness
def _admissible(pf):
    """Thresholds are the tolerance arithmetic of this task, not taste: calibrate sets T = 2.25*p90(|oracle
    error|), so a rival has to move an answer by about 7.4 oracle sigmas, and G3 wants a set at least 4T
    wide.  The oracle's measured spread over 24 replays is sd 0.005-0.009 on q1 and 0.004-0.008 on q2 (the
    between-call shift on the pooled row dominates, and it does not shrink at this budget), which puts
    2T at 0.057-0.068 on the worst seed.  Every floor below is that number with headroom; the earlier,
    smaller floors passed on quiet seeds and left rivals unkilled on noisy ones."""
    q = _quantities(pf)
    gap_memo = q[NEW]["all_s"] - q[REF]["all_s"]                     # what the memo reports
    d = [_mix(q[NEW]["sl_clean"], w) - _mix(q[REF]["sl_clean"], w) for w in _grid()]
    d_sc = [_mix(q[NEW]["sl_clean_s"], w) - _mix(q[REF]["sl_clean_s"], w) for w in _grid()]
    d_al = [_mix(q[NEW]["sl_all"], w) - _mix(q[REF]["sl_all"], w) for w in _grid()]
    mix_all = [_mix(q[NEW]["sl_all"], w) for w in _grid()]
    width = max(mix_all) - min(mix_all)
    prem = q[NEW]["all"] - q[NEW]["clean"]
    prem_det = q[NEW]["all"] - q[NEW]["det"]
    denom_eff = abs(q[NEW]["clean"] - q[NEW]["clean_s"])
    definite = lambda xs: min(xs) >= 0.01 or max(xs) <= -0.01
    checks = [
        (0.30 <= q[NEW]["all_s"] <= 0.80, "M_new pooled accuracy %.3f outside [0.30, 0.80]" % q[NEW]["all_s"]),
        (gap_memo >= 0.03, "the memo needs a headline gap: %+.4f" % gap_memo),
        (max(d) >= BAND and min(d) <= -BAND,
         "q5 must be undetermined with margin: release gap spans [%+.4f, %+.4f]" % (min(d), max(d))),
        # G11 wants every step of q5's chain to be load-bearing *for q5*, which here is the substantive
        # claim of the task: take either shortcut and the undetermined comparison becomes a confident one.
        # Geometrically each shortcut moves the line by a level, so its level has to exceed the deeper of
        # the two margins; that caps the swing from above, which is why draw() solves for it.
        (definite(d_sc), "the scored denominator must decide q5: [%+.4f, %+.4f]" % (min(d_sc), max(d_sc))),
        (definite(d_al), "ignoring contamination must decide q5: [%+.4f, %+.4f]" % (min(d_al), max(d_al))),
        (width >= 0.085, "q3 set width %.4f below 4T + headroom" % width),
        (prem >= 0.085, "contamination premium %.4f too small to kill skip:clean" % prem),
        (abs(prem - prem_det) >= 0.075,
         "the cheap detector is too good: premium %.4f vs %.4f" % (prem, prem_det)),
        (q["sd"] - q["sd_binom"] >= 0.034,
         "q4 anti-prior gap %.4f (sd %.4f vs binomial %.4f)" % (q["sd"] - q["sd_binom"], q["sd"], q["sd_binom"])),
        (denom_eff >= 0.075, "the denominator convention must matter: %.4f" % denom_eff),
    ]
    for ok, why in checks:
        if not ok:
            return False, why
    return True, ("memo gap %+.4f; release gap [%+.4f, %+.4f] (scored denominator would say [%+.4f, %+.4f], "
                  "contamination ignored [%+.4f, %+.4f]); q3 width %.4f; premium %.4f (the cheap detector "
                  "would say %.4f); sd %.4f vs binomial %.4f; denominator %.4f"
                  % (gap_memo, min(d), max(d), min(d_sc), max(d_sc), min(d_al), max(d_al),
                     width, prem, prem_det, q["sd"], q["sd_binom"], denom_eff))


def wellposed(w):
    return _admissible(w["pf"])


def public_values(ctx):
    vals = [N_ITEMS, DET_THRESH, ctx["gap"], ctx["gap2"], ctx["se"], ctx["n_flag"], ctx["dup40"]]
    vals += list(ctx["acc"]) + list(ctx["slice_acc"]) + list(ctx["slice_n"]) + list(ctx["n_scored"])
    vals += list(W0)
    # The shipped detector table is 1500 high-entropy numbers, so at 3-5 significant digits some of them
    # collide with some hidden constant in roughly every instance.  They are disclosed by construction (the
    # team ran that scan and the memo cites it), and a detector score that happens to share four digits with
    # a difficulty parameter reveals nothing, so they are declared public rather than left to chance.  Only
    # this column is declared: every other number the notebook prints stays under the scanner.
    vals += list(ctx.get("overlap") or [])
    return vals
