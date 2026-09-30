"""Analyst + curator pass over results/*.json -> MATRIX.md (and items/examples.md via --examples).

Zero model calls.  Everything here is recomputed from the saved results; E4 rebuilds items from (world, design, seed).
usage: python3 analyze.py            (after: python3 compiler.py --modes oneshot,loop,search,search_t1)
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cards import CARDS                                      # noqa: E402
from compiler import GATES, realism_flags                    # noqa: E402
from templates import TEMPLATES, BUILD                       # noqa: E402

CARD = {c.key: c for c in CARDS}


def load(mode):
    out = {}
    for fn in glob.glob(os.path.join(HERE, "results", "*_%s.json" % mode)):
        J = json.load(open(fn))
        if J["mode"] == mode:
            out[(J["card"], J["tpl"])] = J
    return out


ONE, LOOP, S2, S1 = load("oneshot"), load("loop"), load("search"), load("search_t1")
KEYS = [(c.key, t) for c in CARDS for t in TEMPLATES]
NW = max(len(J["results"]) for J in S2.values() if not J["na"])      # worlds per search cell
NO = max(len(J["results"]) for J in ONE.values() if not J["na"])     # evaluations per oneshot/loop cell
L = []
P = L.append


def fam(name):
    return (name or "-").split(":")[0] if not (name or "").startswith("naive_ignore") else ":".join(name.split(":")[:2])


def pct(a, b):
    return "%d%%" % round(100.0 * a / b) if b else "-"


# ---------------------------------------------------------------------------------------------------- A. yield
P("# MATRIX — AV 编译器产出矩阵（自动生成，勿手改；`python3 analyze.py`）\n")
P(("所有数字来自 `results/*.json`，零模型调用。oneshot / loop：每格 %d 次门评估；search：每格 %d 个世界 × ≤48 次探针，" % (NO, NW)) +
  "每个世界只记最优设计（fertility = 有可认证设计的世界占比；search 是启发式，所以这是下界）。"
  "T2 = 过全部门含红队 G9；T1 = 只对模板基础对手认证（不计 G9）。T1 与 T2 用同一批世界（配对）。"
  "clean = 无现实性标记（r_nb<4 且 n<300）。判定：T2 可育世界 ≥50% = 可育，≥12.5% = 世界受限，否则近乎不育。\n")
P("## A. 产出与可育性矩阵\n")
P("| 卡 × 模板 | oneshot 认证/%d (clean) | loop 认证/%d (clean) | search T2 可育/%d (clean) | 探针/可育世界 | search T1 可育/%d (clean) | T1→T2 被红队杀 | 判定 |" % (NO, NO, NW, NW))
P("|---|---|---|---|---|---|---|---|")
verdict = {}
tot = collections.Counter()
for k in KEYS:
    if k not in S2:
        continue
    if S2[k]["na"]:
        P("| %s × %s | N/A | N/A | N/A | - | N/A | - | **N/A**：卡无物理下限（floor_kind=None），M6 前提不成立 |" % k)
        verdict[k] = "N/A"
        continue
    o = [r for r in ONE[k]["results"] if r.get("certified")]
    lp = [r for r in LOOP[k]["results"] if r.get("certified")]
    s2 = [r for r in S2[k]["results"] if r.get("tier_cert")]
    s1 = [r for r in S1[k]["results"] if r.get("tier_cert")]
    killed = sum(1 for r in s1 if not r.get("certified"))
    c2 = sum(1 for r in s2 if not realism_flags(r))
    v = "可育" if len(s2) >= 0.5 * NW else ("世界受限" if len(s2) >= 0.125 * NW else "近乎不育（T2）")
    if len(s2) < 0.125 * NW and len(s1) >= 0.3 * NW:
        v += "；T1 可育 → 分层候选"
    verdict[k] = v
    for nm, val in (("one", len(o)), ("loop", len(lp)), ("s2", len(s2)), ("s1", len(s1)), ("c2", c2)):
        tot[nm] += val
    tot["cells"] += 1
    P("| %s × %s | %d (%d) | %d (%d) | %d (%d) | %s | %d (%d) | %d | %s |" % (
        k[0], k[1], len(o), sum(1 for r in o if not realism_flags(r)), len(lp), sum(1 for r in lp if not realism_flags(r)),
        len(s2), c2, "%.0f" % (S2[k]["spent"] / len(s2)) if s2 else "∞（%d 探针 0 题）" % S2[k]["spent"], len(s1), sum(1 for r in s1 if not realism_flags(r)),
        killed, v))
P("\n合计（%d 个非 N/A 格）：oneshot %d/%d 认证，loop %d/%d，search T2 %d/%d 世界可育（clean %d），T1 %d/%d。\n" % (
    tot["cells"], tot["one"], NO * tot["cells"], tot["loop"], NO * tot["cells"], tot["s2"], NW * tot["cells"], tot["c2"],
    tot["s1"], NW * tot["cells"]))
# paired-world check: T2 certification implies T1 certification on the same world, so any T2-fertile world the T1 search
# missed measures the search's miss rate (fertility is a lower bound).
miss = n2f = 0
for k in S2:
    if S2[k]["na"]:
        continue
    for a, b in zip(S2[k]["results"], S1[k]["results"]):
        assert a["world"] == b["world"], "tiers must be run on paired worlds"
        if a.get("tier_cert"):
            n2f += 1
            miss += not b.get("tier_cert")
P("配对检验：T2 可育的 %d 个世界里，T1 搜索漏掉 %d 个（T2 ⊂ T1，所以这是搜索的漏检率 %s；可育性是下界）。\n" % (
    n2f, miss, pct(miss, n2f)))

# per-100-evaluations efficiency
P("**每 100 次门评估的认证数**（同一计算量下三种 proposer 的效率；search 每世界只计 1 个）：")
e1 = sum(len([r for r in ONE[k]["results"] if r.get("certified")]) for k in ONE if not ONE[k]["na"])
n1 = sum(ONE[k]["spent"] for k in ONE if not ONE[k]["na"])
e2 = sum(len([r for r in LOOP[k]["results"] if r.get("certified")]) for k in LOOP if not LOOP[k]["na"])
n2 = sum(LOOP[k]["spent"] for k in LOOP if not LOOP[k]["na"])
e3 = sum(len([r for r in S2[k]["results"] if r.get("tier_cert")]) for k in S2 if not S2[k]["na"])
n3 = sum(S2[k]["spent"] for k in S2 if not S2[k]["na"])
fc = [r["first_cert_probe"] for J in S2.values() if not J["na"] for r in J["results"]
      if r.get("tier_cert") and r.get("first_cert_probe")]
P("oneshot %.1f（%d/%d），loop（规则修复）%.1f（%d/%d），search（门作 oracle 的爬山）%.1f（%d/%d；但 search 在找到认证后"
  "还会继续找 clean 且 B≥6 的设计，按“首次认证所需探针”算，可育世界的中位数是 %d 次、P75 %d 次）。\n" % (
      100.0 * e1 / n1, e1, n1, 100.0 * e2 / n2, e2, n2, 100.0 * e3 / n3, e3, n3,
      int(np.median(fc)), int(np.quantile(fc, .75))))


def coverage(D, key):
    per = collections.Counter()
    for k, J in D.items():
        if not J["na"]:
            per[k] = sum(1 for r in J["results"] if r.get(key))
    n = sum(per.values())
    top = per.most_common(1)[0] if n else (None, 0)
    byt = collections.Counter()
    for k, v in per.items():
        byt[k[1]] += v
    return sum(1 for v in per.values() if v), n, top, byt


P("**产出分布**（同样是零模型调用，门评估单核约 5–6 ms 一次（中位数），所以瓶颈不是评估次数，而是“每个世界能不能出题”和题目在格子间是否分散）：\n")
P("| proposer | 有 ≥1 题的格 /18 | 题数 | 最集中的格（占比） | 按模板 M1/M2/M3/M6 |")
P("|---|---|---|---|---|")
for nm, D, key in (("oneshot", ONE, "certified"), ("loop", LOOP, "certified"), ("search T2", S2, "tier_cert"),
                   ("search T1", S1, "tier_cert")):
    ncell, n, top, byt = coverage(D, key)
    P("| %s | %d | %d | %s × %s（%s） | %s |" % (nm, ncell, n, top[0][0], top[0][1], pct(top[1], n),
                                              "/".join(str(byt[t]) for t in TEMPLATES)))
P("\noneshot/loop 的题有一半以上来自 M6 这类“随手就能认证”的格；search 把每个世界的问题答到了底，"
  "所以它的价值不在单位评估效率，而在：(1) 覆盖面（难格也能出题）；(2) 给出“这个世界能不能出题”的可证伪答案——"
  "剩下的不育由世界常数决定（见 C、G），不是 proposer 不够聪明。当世界本身很贵（目标 2 的移植世界需要人工/模型整理），"
  "按世界计的产出才是该优化的量。\n")

# ---------------------------------------------------------------------------------------------------- B. certified items
P("## B. 已认证题（search T2）：裕度、绑定对手、驱动常数\n")
P("绑定对手 = 加入红队后离真值最近的捷径（B 由它决定）。驱动常数 = 把每个世界常数沿先验范围挪 10%，"
  "该捷径偏差变化（以 T 计）最大的那个；stat = 噪声/总体离散类常数（与卡的机制无关），mechanism = 卡的机制常数。\n")
P("| 模板 | 认证数 | B_lo(含红队) 中位 [IQR] | 绑定对手族 | 驱动类 stat/mech | 最常见驱动常数 | 路由间隔≥2T 占比 |")
P("|---|---|---|---|---|---|---|")
by_tpl = collections.defaultdict(list)
for k, J in S2.items():
    if not J["na"]:
        by_tpl[k[1]] += [r for r in J["results"] if r.get("tier_cert")]
for t in TEMPLATES:
    R = by_tpl[t]
    if not R:
        continue
    B = np.array([r["B_lo_red"] for r in R])
    nf = collections.Counter(fam(r["nearest_red"]) for r in R)
    dc = collections.Counter(r.get("driver_class") for r in R)
    dk = collections.Counter("%s(%s)" % (r.get("driver_key"), r["card"][:2]) for r in R)
    sep = sum(1 for r in R if r["min_route_sep_over_T"] >= 2.0)
    P("| %s | %d | %.1f [%.1f, %.1f] | %s | %d / %d | %s | %s |" % (
        t, len(R), np.median(B), np.quantile(B, .25), np.quantile(B, .75),
        ", ".join("%s %d" % kv for kv in nf.most_common(3)), dc["stat"], dc["mechanism"],
        ", ".join("%s %d" % kv for kv in dk.most_common(3)), pct(sep, len(R))))
P("")
P("读法：M2/M3 若驱动类以 stat 为主，说明这些题的难度来自统计结构（删失比例、赢家诅咒的噪声/离散比），与机制卡无关——"
  "换一张卡只是换皮（见 E 的跨卡坍缩）。M1/M6 若以 mechanism 为主，说明难度绑定在卡的物理上，换卡才是真正的新题。\n")

# ---------------------------------------------------------------------------------------------------- C. why infertile
P("## C. 不育世界为什么不育（search T2 中未认证世界的最优设计卡在哪道门）\n")
P("| 卡 × 模板 | 不育世界数 | 首个失败门（最优设计） | 该门处最近对手 |")
P("|---|---|---|---|")
for k in KEYS:
    if k not in S2 or S2[k]["na"]:
        continue
    R = [r for r in S2[k]["results"] if not r.get("tier_cert")]
    if not R:
        continue
    ff = collections.Counter(r["first_fail"] for r in R)
    nr = collections.Counter(fam(r.get("nearest")) for r in R if r["first_fail"] in ("G5_B", "G9_redteam"))
    P("| %s × %s | %d | %s | %s |" % (k[0], k[1], len(R), ", ".join("%s %d" % kv for kv in ff.most_common(3)),
                                     ", ".join("%s %d" % kv for kv in nr.most_common(3)) or "-"))
P("")

# ---------------------------------------------------------------------------------------------------- D. red team
P("## D. 红队杀伤率（E2）：只对基础对手认证的题，有多少会被“校准过的迁移/假设型”对手杀掉\n")
P("| 模板 | T1 认证世界 | 其中 T2 仍认证 | 被红队杀 | 杀伤率 | 杀手路由 |")
P("|---|---|---|---|---|---|")
kill_items = []
for t in TEMPLATES:
    R = [dict(r, _k=k) for k, J in S1.items() if k[1] == t and not J["na"] for r in J["results"] if r.get("tier_cert")]
    if not R:
        continue
    kd = [r for r in R if not r.get("certified")]
    kill_items += kd
    P("| %s | %d | %d | %d | %s | %s |" % (t, len(R), len(R) - len(kd), len(kd), pct(len(kd), len(R)),
                                          ", ".join("%s %d" % kv for kv in collections.Counter(r["nearest_red"] for r in kd).most_common(3))))
ol = [r for J in list(ONE.values()) + list(LOOP.values()) if not J["na"] for r in J["results"] if r.get("certified_without_redteam")]
P("\n注意：“其中 T2 仍认证”是 T1 搜索找到的最优设计在加入红队后仍认证的数目；A 表的 T2 列是另一次以 T2 为目标的搜索，所以两者不同（例如 M1：T1 最优设计几乎全被杀，但以 T2 为目标仍能在少数世界找到设计）。杀伤率衡量的是“只对基础对手优化出来的题”有多不可靠。\n")
P("oneshot+loop 中不计红队即可认证的 %d 题里，%d 题被红队杀（%s）。这是“只对作者想到的对手认证”的假阳性率下界。\n" % (
    len(ol), sum(1 for r in ol if not r.get("certified")), pct(sum(1 for r in ol if not r.get("certified")), len(ol))))

# ---------------------------------------------------------------------------------------------------- E. dedup
P("## E. 策展：签名去重与跨卡坍缩\n")
P("签名 = (卡, 模板, 绑定对手族, 驱动类或机制常数, ⌊log2 B_lo⌋)。策展器每个签名最多保留 2 题。"
  "跨卡签名去掉“卡”；若同一跨卡签名出现在 ≥2 张卡上且驱动类为 stat，这些题是同一考点换皮。\n")


def sig(r, with_card=True):
    drv = r.get("driver_class") if r.get("driver_class") != "mechanism" else "mech:" + str(r.get("driver_key"))
    s = (r["tpl"], fam(r["nearest_red"]), drv, int(math.floor(math.log2(max(r["B_lo_red"], 1e-9)))))
    return ((r["card"],) + s) if with_card else s


allc = [r for J in S2.values() if not J["na"] for r in J["results"] if r.get("tier_cert")]
cs = collections.Counter(sig(r) for r in allc)
kept = sum(min(v, 2) for v in cs.values())
P("T2 认证 %d 题 → %d 个签名 → 策展后保留 %d 题（%s）。\n" % (len(allc), len(cs), kept, pct(kept, len(allc))))
P("| 模板 | 认证 | 签名数 | 策展保留 | 跨卡签名数 | 跨 ≥2 卡的 stat 签名（覆盖题数） |")
P("|---|---|---|---|---|---|")
for t in TEMPLATES:
    R = [r for r in allc if r["tpl"] == t]
    if not R:
        continue
    c1 = collections.Counter(sig(r) for r in R)
    c0 = collections.defaultdict(set)
    n0 = collections.Counter()
    for r in R:
        c0[sig(r, False)].add(r["card"])
        n0[sig(r, False)] += 1
    multi = [s for s, cardset in c0.items() if len(cardset) >= 2 and s[2] == "stat"]
    P("| %s | %d | %d | %d | %d | %d（%d 题） |" % (t, len(R), len(c1), sum(min(v, 2) for v in c1.values()), len(c0),
                                             len(multi), sum(n0[s] for s in multi)))
P("")

# ---------------------------------------------------------------------------------------------------- F. realism
P("## F. 现实性：多少可育性依赖不现实的设计\n")
fl = collections.Counter()
for r in allc:
    for f in realism_flags(r):
        fl[f.split(" x")[0].split("=")[0]] += 1
nfl = sum(1 for r in allc if realism_flags(r))
P("T2 认证 %d 题中 %d 题（%s）带现实性标记：%s。搜索框刻意越过现实线（r_nb≤6，n≤400），"
  "所以这个比例就是“要靠夸张的笔记本噪声或预算才能出题”的份额。\n" % (len(allc), nfl, pct(nfl, len(allc)),
                                                  ", ".join("%s %d" % kv for kv in fl.most_common())))

# ---------------------------------------------------------------------------------------------------- G. AUC
P("## G. 哪些世界常数预测可育性（AUC，%d 世界/格，|AUC−0.5|≥0.25 才列出）\n" % NW)
P("AUC = P(可育世界的该常数 > 不育世界的该常数)。>0.5 表示该常数越大越可育。n 很小，只作为假设线索。\n")
P("| 卡 × 模板 | 可育/%d | 强预测常数 |" % NW)
P("|---|---|---|")


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return s / (len(pos) * len(neg))


for k in KEYS:
    if k not in S2 or S2[k]["na"]:
        continue
    R = S2[k]["results"]
    y = [bool(r.get("tier_cert")) for r in R]
    if min(sum(y), len(y) - sum(y)) < 4:
        P("| %s × %s | %d | （少数类 <4 个世界，不计算） |" % (k[0], k[1], sum(y)))
        continue
    items = []
    for key in CARD[k[0]].prior:
        a = auc([r["world"][key] for r, f in zip(R, y) if f], [r["world"][key] for r, f in zip(R, y) if not f])
        if a is not None and abs(a - 0.5) >= 0.25:
            items.append((abs(a - .5), "%s %.2f%s" % (key, a, "*" if key in CARD[k[0]].stat_keys else "")))
    P("| %s × %s | %d | %s |" % (k[0], k[1], sum(y), ", ".join(s for _, s in sorted(items, reverse=True)) or "-"))
P("\n（* = stat 类常数）\n")

# ---------------------------------------------------------------------------------------------------- H. E4 attributability
P("## H. 可归因性演示（E4）：隐藏路由会产生“无法解释的错答”，红队库补上后才可归因\n")
P("取 T1 认证、T2 被杀的题，模拟一个“用红队路由作答”的 agent（例如用笔记本曲线 + 一个生产点做常数偏移迁移）。"
  "只拿基础对手库做归因时，这个答案要么被判对（假认证：该捷径落在 T 内），要么被判错但不靠近任何已知对手（不可归因）；"
  "把红队路由加入库后才能归因到具体的错误假设。\n")
cnt = collections.Counter()
ex = []
viol = collections.defaultdict(list)          # (card, judged_correct?) -> spread of per-unit shifts / T (M2 unit_slope)


def shift_spread(card, r):
    """IQR over units of (value at target config - value at notebook config), in units of T: how badly the
    'every unit shifts by the same amount' assumption behind naive_ignore:unit_slope is violated in this world."""
    d, th = r["design"], r["world"]
    rg = np.random.default_rng(r["seed"])
    q, q2 = rg.standard_normal(d["K"]), rg.standard_normal(d["K"])
    sh = np.array([float(card.unit_y(d["xs"], th, q[j], q2[j])) - float(card.unit_y(d["x_nb"], th, q[j], q2[j]))
                   for j in range(d["K"])])
    return float(np.subtract(*np.quantile(sh, [.75, .25]))) / r["T"]


for r in kill_items:
    card = CARD[r["card"]]
    item, _ = BUILD[r["tpl"]](card, r["world"], r["design"], r["seed"])
    if item is None:
        continue
    T, truth = r["T"], item["truth"]
    ans = item["red"][r["nearest_red"]]
    e = abs(ans - truth) / T
    base_hit = [k for k, v in item["routes"].items() if k not in item["reference"] and abs(v - ans) <= T]
    if e <= 1.0:
        cat = "judged_correct"
    elif base_hit:
        cat = "attr_base"
    else:
        cat = "attr_red_only"
    cnt[(r["tpl"], cat)] += 1
    if r["tpl"] == "M2" and r["nearest_red"].startswith("naive_ignore:unit_slope"):
        viol[(r["card"], cat == "judged_correct")].append(shift_spread(card, r))
    if len(ex) < 4 and cat != "attr_base":
        ex.append((r["card"], r["tpl"], r["nearest_red"], e, cat))
P("| 模板 | 被杀题 | 判对（红队路由落在 T 内） | 判错且基础库不可归因、红队库可归因 | 判错且基础库可归因 |")
P("|---|---|---|---|---|")
for t in TEMPLATES:
    n = sum(v for (tt, _), v in cnt.items() if tt == t)
    if n:
        P("| %s | %d | %d | %d | %d |" % (t, n, cnt[(t, "judged_correct")], cnt[(t, "attr_red_only")], cnt[(t, "attr_base")]))
vrow = []
for c in sorted({k[0] for k in viol}):
    a, b = viol.get((c, True), []), viol.get((c, False), [])
    if a and b:
        vrow.append("%s %.2f T（判对 %d 题）对 %.2f T（判错 %d 题）" % (c[:2], np.median(a), len(a), np.median(b), len(b)))
P("\n“判对”一列：这些被杀题里，红队路由在该世界恰好落在 T 内，用它作答会被判对。部分原因是它的假设在该世界近似成立。"
  "以 M2 的 `naive_ignore:unit_slope` 为例，它假设各单元从笔记本配置到目标配置的平移相同；违背量取各单元平移的四分位距"
  "（以 T 计），判对与判错世界的中位数对比：%s。判对世界的违背量是否系统性更小、分离是否干净，以这行数字为准。"
  "这类题不能区分“做了未检验的假设”和“做对了”，所以红队路由也需要 M1 那样的有效性检验（README §5）。"
  "其余被杀题才是真的“基础库不可归因的错答”。\n" % ("；".join(vrow) or "（没有同时含两类的卡）"))
lower = [c for c in sorted({k[0] for k in viol}) if viol.get((c, True)) and viol.get((c, False))
         and np.median(viol[(c, True)]) < np.median(viol[(c, False)])]
both = [c for c in sorted({k[0] for k in viol}) if viol.get((c, True)) and viol.get((c, False))]
P("汇总：%d/%d 张卡上判对世界的违背量中位数更低（%s）；反例：%s。四分位距只是违背的粗代理——"
  "该路由的误差取决于平移分布相对中位数的不对称，而不只是离散度，所以不能把它当成可在出题时检查的有效性条件。\n" % (
      len(lower), len(both), ", ".join(c[:2] for c in lower), ", ".join(c[:2] for c in both if c not in lower) or "无"))
for c, t, nm, e, cat in ex:
    P("- %s × %s：红队路由 `%s` 的答案离真值 %.2f T → %s" % (c, t, nm, e, cat))
P("")

# ---------------------------------------------------------------------------------------------------- I. loop rules
P("## I. loop 修复规则的实测效果（规则作者错误的证据）\n")
P("对每条修复动作：应用次数、下一步是否推进到更靠后的门、链条最终是否认证。推进率低的规则就是写错了的规则。\n")
P("| 模板 | 修复动作 | 次数 | 下一步推进 | 链最终认证 |")
P("|---|---|---|---|---|")
st = collections.defaultdict(lambda: [0, 0, 0])


def gi(ff):
    return len(GATES) if ff is None else GATES.index(ff)


for k, J in LOOP.items():
    if J["na"]:
        continue
    for ch in J["trace"]:
        steps = ch["steps"]
        final = steps[-1]["first_fail"] is None
        for i, s in enumerate(steps[:-1]):
            if "action" not in s:
                continue
            a = (k[1], s["action"].split(" -> ")[-1][:60])
            st[a][0] += 1
            st[a][1] += gi(steps[i + 1]["first_fail"]) > gi(s["first_fail"])
            st[a][2] += final
for (t, a), (n, pr, fin) in sorted(st.items(), key=lambda x: (x[0][0], -x[1][0])):
    if n >= 3:
        P("| %s | %s | %d | %s | %s |" % (t, a, n, pct(pr, n), pct(fin, n)))
P("")

open(os.path.join(HERE, "MATRIX.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
