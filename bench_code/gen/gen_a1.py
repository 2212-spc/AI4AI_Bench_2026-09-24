"""Certificate-driven generator for family A1 (masked convention-conflict regression).
Mechanism cards: GA-normalisation (x k) + momentum dampening (x (1-mu)).
Usage: python gen_a1.py <instance_seed> [<instance_seed> ...]
"""
import json, os, shutil, sys, time, hashlib
import numpy as np
sys.path.insert(0, "/tmp/bench/lab"); sys.path.insert(0, "/tmp/bench/gen")
import engine as E
import a1_render as R
import a1_verify_core as V

OUT = "/tmp/bench/instances/a1"
N_SEEDS = 8


def stats(v):
    v = np.asarray(v)
    fin = np.isfinite(v).all()
    return (float(v.mean()) if fin else float("inf")), (float(v.std(ddof=1)) if fin else float("inf")), fin


def gen(inst_seed, log):
    r = np.random.default_rng(inst_seed)
    world = dict(d_in=int(r.choice([24, 32, 40])), t_hidden=int(r.choice([48, 64])), noise=float(r.choice([0.1, 0.15])))
    teacher = E.make_teacher(int(r.integers(1 << 30)), **world)
    k = int(r.choice([2, 4, 8])); mom = float(r.choice([0.8, 0.85, 0.9])); mb = int(r.choice([16, 32]))
    wd = float(r.choice([0.0, 1e-4, 5e-4]))
    base = dict(hidden=int(r.choice([96, 128])), lr=None, momentum=mom, weight_decay=wd, warmup_steps=25,
                steps=500, min_lr_ratio=0.05, grad_accum_steps=k, micro_batch_size=mb, clip_norm=None)
    dseed = int(r.integers(1 << 30))
    xtr, ytr = E.sample(teacher, 20000, dseed); xva, yva = E.sample(teacher, 4000, dseed + 1)
    data = (xtr, ytr, xva, yva)
    seeds = list(range(N_SEEDS))
    rec = {"inst_seed": inst_seed, "world": world, "k": k, "mom": mom, "mb": mb, "wd": wd}
    # 1) "v1 tuned" lr: best lr for the correct implementation
    grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    best = None
    for lr in grid:
        cfg = dict(base, lr=lr)
        m, s, fin = stats(E.eval_runs(cfg, data, seeds))
        if fin and (best is None or m < best[1]):
            best = (lr, m, s)
    lr = best[0]; cfg = dict(base, lr=lr)
    rec["lr_tuned"] = lr
    # 2) masking certificate on canonical
    res = {}
    for name, fl in [("fixed", {}), ("current", dict(bug_ga=True, bug_damp=True)),
                     ("fix_ga_only", dict(bug_damp=True)), ("fix_damp_only", dict(bug_ga=True))]:
        res[name] = stats(E.eval_runs(cfg, data, seeds, **fl))
    rec["canonical"] = {k2: [round(v[0], 6), round(v[1], 6)] for k2, v in res.items()}
    fm, fs, _ = res["fixed"]; cm, cs, cfin = res["current"]
    se = np.sqrt(fs ** 2 / N_SEEDS + cs ** 2 / N_SEEDS) if cfin else np.inf
    reasons = []
    if not cfin: reasons.append("current diverges (bug not silent)")
    elif cm - fm < max(4 * se, 0.05 * fm): reasons.append("regression too small: %.5f vs %.5f" % (cm, fm))
    for single in ["fix_ga_only", "fix_damp_only"]:
        sm, ss, sfin = res[single]
        if sfin and cfin and sm < cm + 2 * se:
            reasons.append("%s does not look worse (%.5f vs current %.5f) -> not masked" % (single, sm, cm))
    if reasons:
        rec["rejected"] = reasons; log.append(rec); return None
    # 3) hidden configs: k*(1-mu) must differ from canonical by >=2x; reference must converge
    prod0 = k * (1 - mom)
    cands = []
    for k2 in [1, 2, 4, 8]:
        for m2 in [0.5, 0.7, 0.8, 0.9, 0.95]:
            p2 = k2 * (1 - m2)
            if max(p2 / prod0, prod0 / p2) >= 2.0:
                cands.append((k2, m2))
    r.shuffle(cands)
    checks = [dict(name="canonical", cfg=cfg, n_train=20000, n_test=4000, train_data_seed=dseed,
                   test_data_seed=dseed + 7, seeds=list(range(100, 108)))]
    for (k2, m2) in cands:
        if len(checks) >= 3: break
        for lr2 in [lr, lr * 0.5, lr * 2]:
            c2 = dict(cfg, grad_accum_steps=k2, momentum=m2, lr=lr2, weight_decay=float(r.choice([0.0, 1e-4, 5e-4])),
                      micro_batch_size=int(r.choice([16, 32])))
            ds2 = int(r.integers(1 << 30))
            x2, y2 = E.sample(teacher, 20000, ds2)
            v = E.eval_runs(c2, (x2, y2) + E.sample(teacher, 4000, ds2 + 7), list(range(200, 208)))
            m, s, fin = stats(v)
            if fin and m < 1.5 * fm:
                checks.append(dict(name="heldout_%d" % len(checks), cfg=c2, n_train=20000, n_test=4000,
                                   train_data_seed=ds2, test_data_seed=ds2 + 7, seeds=list(range(200, 208))))
                break
    if len(checks) < 3:
        rec["rejected"] = ["could not find 2 convergent held-out configs"]; log.append(rec); return None
    # reference stats for every check (engine, bug-free)
    for chk in checks:
        x2, y2 = E.sample(teacher, chk["n_train"], chk["train_data_seed"])
        x3, y3 = E.sample(teacher, chk["n_test"], chk["test_data_seed"])
        v = E.eval_runs(chk["cfg"], (x2, y2, x3, y3), chk["seeds"])
        m, s, _ = stats(v)
        chk["ref_mean"] = m; chk["ref_std"] = s
        chk["tol"] = max(3 * np.sqrt(2.0 / len(chk["seeds"])) * s, 0.01 * m)
    spec = {"checks": checks}
    ga_form = str(r.choice(["len_idx", "internal_mean"])); damp_form = str(r.choice(["default_arg", "ema"]))
    iid = "a1-%s" % hashlib.sha1(str(inst_seed).encode()).hexdigest()[:6]
    d = os.path.join(OUT, iid); shutil.rmtree(d, ignore_errors=True)
    pub = os.path.join(d, "public"); os.makedirs(pub + "/data")
    np.savez(pub + "/data/train.npz", x=xtr, y=ytr); np.savez(pub + "/data/val.npz", x=xva, y=yva)
    rs = {"ga_form": ga_form, "damp_form": damp_form, "canonical": cfg,
          "readme_nums": {"ref_mean": fm, "ref_std": fs, "cur_mean": cm}}
    instr = R.render(pub, rs, bug_ga=True, bug_damp=True)
    instr_hint = R.render(d + "/tmp_hint", rs, bug_ga=True, bug_damp=True, hint=True); shutil.rmtree(d + "/tmp_hint")
    open(d + "/instruction.md", "w").write(instr); open(d + "/instruction_hint.md", "w").write(instr_hint)
    tdict = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in teacher.items()}
    hid = os.path.join(d, "hidden"); os.makedirs(hid)
    json.dump({"spec": spec, "teacher": tdict}, open(hid + "/verify_spec.json", "w"))
    # 4) battery on RENDERED code through the real verifier
    C_all = 1.0 / (k * (1 - mom))
    variants = {
        "oracle": dict(bug_ga=False, bug_damp=False),
        "nop": dict(bug_ga=True, bug_damp=True),
        "fix_ga_only": dict(bug_ga=False, bug_damp=True),
        "fix_damp_only": dict(bug_ga=True, bug_damp=False),
        "const_lr_hack": dict(bug_ga=True, bug_damp=True, patch=("opt.lr = lr_at(step, cfg)", "opt.lr = %r * lr_at(step, cfg)" % C_all)),
        "fix_ga+const_lr_hack": dict(bug_ga=False, bug_damp=True, patch=("opt.lr = lr_at(step, cfg)", "opt.lr = %r * lr_at(step, cfg)" % (1 / (1 - mom)))),
        "fix_damp+const_lr_hack": dict(bug_ga=True, bug_damp=False, patch=("opt.lr = lr_at(step, cfg)", "opt.lr = %r * lr_at(step, cfg)" % (1.0 / k))),
        "oracle+decoy_grad_x2": dict(bug_ga=False, bug_damp=False, mpatch=("g_out = (r / denom)", "g_out = (2.0 * r / denom)")),
        "oracle+decoy_wd_skip_bias": dict(bug_ga=False, bug_damp=False, opatch=("if self.weight_decay:", "if self.weight_decay and not k.startswith('b'):")),
        "oracle+decoy_w2_he": dict(bug_ga=False, bug_damp=False, mpatch=("np.sqrt(1.0 / hidden)", "np.sqrt(2.0 / hidden)")),
    }
    expect_pass = {"oracle": True, "nop": False, "fix_ga_only": False, "fix_damp_only": False, "const_lr_hack": False,
                   "fix_ga+const_lr_hack": False, "fix_damp+const_lr_hack": False}
    battery = {}
    for name, vv in variants.items():
        vd = os.path.join(d, "variants", name)
        R.render(vd, rs, bug_ga=vv["bug_ga"], bug_damp=vv["bug_damp"])
        for key, fn in [("patch", "trainer.py"), ("mpatch", "model.py"), ("opatch", "optim.py")]:
            if key in vv:
                pth = os.path.join(vd, "minilab", fn); s = open(pth).read()
                assert vv[key][0] in s, (name, vv[key][0]); open(pth, "w").write(s.replace(vv[key][0], vv[key][1]))
        passed, rep = V.evaluate(vd, spec, V.teacher_from(tdict), verbose=False)
        battery[name] = {"pass": passed, "detail": rep}
    rec["battery"] = {n: b["pass"] for n, b in battery.items()}
    bad = [n for n, e in expect_pass.items() if battery[n]["pass"] != e]
    # render consistency: oracle rendered code == engine numbers on canonical check (exact seeds)
    orc = battery["oracle"]["detail"][0]
    if abs(orc.get("mean", 1e9) - checks[0]["ref_mean"]) > 1e-5 * checks[0]["ref_mean"]:
        bad.append("render-consistency: oracle %.8f vs engine %.8f" % (orc.get("mean", -1), checks[0]["ref_mean"]))
    nopc = battery["nop"]["detail"][0]
    # stability: oracle passes on 2 extra fresh seed sets
    for rep_i in range(2):
        spec2 = json.loads(json.dumps(spec))
        for chk in spec2["checks"]:
            chk["seeds"] = [s + 1000 * (rep_i + 1) for s in chk["seeds"]]
        ok2, _ = V.evaluate(os.path.join(d, "variants", "oracle"), spec2, V.teacher_from(tdict), verbose=False)
        if not ok2: bad.append("oracle unstable on fresh seeds (rep %d)" % rep_i)
    rec["decoys"] = {n: battery[n]["pass"] for n in battery if "decoy" in n}
    if bad:
        rec["rejected"] = ["battery: " + str(bad)]; log.append(rec); shutil.rmtree(d); return None
    json.dump({"battery": battery, "record": rec, "surface": {"ga_form": ga_form, "damp_form": damp_form}},
              open(hid + "/certificate.json", "w"), indent=1, default=float)
    rec["accepted"] = iid; log.append(rec)
    return iid


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    log = []
    for s in sys.argv[1:]:
        t = time.time()
        iid = gen(int(s), log)
        print(s, "->", iid, "%.1fs" % (time.time() - t), json.dumps({k: log[-1].get(k) for k in ["rejected", "canonical", "lr_tuned", "k", "mom", "battery", "decoys"]}, default=str)[:900])
    with open(OUT + "/genlog.jsonl", "a") as f:
        for rec in log: f.write(json.dumps(rec, default=str) + "\n")
