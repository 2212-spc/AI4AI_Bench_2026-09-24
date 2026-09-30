"""Render a few certified items as the agent would see them, with the grader's hidden side -> items/examples.md.

Picks, per template, the clean tier-2 item with the largest red-team margin (preferring mechanism-driven items),
plus one M1 item that is certified at tier 1 but killed by the red team (the tier-gap example).
usage: python3 render.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cards import CARDS, Y, S                     # noqa: E402
from compiler import realism_flags                # noqa: E402
from templates import BUILD, TEMPLATES            # noqa: E402

CARD = {c.key: c for c in CARDS}
GLOSS = {
    "B_prior:ctx0": "直接用笔记本（旧上下文）在目标点的值 —— 忽略上下文变化",
    "B_prior:nominal_floor": "背出版常数当下限（Chinchilla E=1.69）—— 不测本世界的下限",
    "naive_ignore:ctx:offset@near": "笔记本曲线 + 最近一个生产点做常数平移迁移 —— 假设上下文效应是常数偏移",
    "naive_ignore:ctx:ratio@near": "同上，但假设上下文效应是常数比例",
    "naive_ignore:ctx:pool": "把两种上下文的数据混在一起拟合二次曲线 —— 假设上下文可忽略",
    "R3:ctx0_slope@near": "最近生产点 + 笔记本曲线的斜率做线性外推 —— 假设上下文不改变斜率",
    "R1:readoff": "读最近网格点的值",
    "R1:notebook_config": "直接用笔记本配置下的全体中位数 —— 忽略配置变化",
    "R1:incumbent_value": "报告在位配方（已发表）的值",
    "R2:endpoint": "读区间端点",
    "R8:saturated": "假设最大配置已经到达下限（M6）/ 假设响应已饱和、读最远网格点（M1）",
    "B_guess:threshold": "把删失阈值 c 当作中位数 —— 题面给出的数字被直接抄用",
    "R4:secant_floor": "用两点割线外推下限",
    "R7:average": "取邻点平均",
    "R7:menu_mean": "报告全菜单均值（过度收缩）",
    "R7:top2_mean": "报告前两名的均值（半吊子收缩）",
    "drop:censored": "丢掉删失单元后取中位数 —— 把“只知道下界”当成“没有信息”",
    "B_guess:mean_uncensored": "未删失单元的均值",
    "naive_ignore:unit_slope": "笔记本配置的中位数 + 未删失单元的中位平移 —— 假设各单元平移相同",
    "drop:censored_mean_nb_shift": "均值版的平移迁移，且丢掉删失单元",
    "B_guess:notebook_value": "直接报告笔记本里冠军配方的数字 —— 赢家诅咒",
    "B_guess:notebook_plus_1se": "冠军数字 + 1 个标准误的“修正” —— 知道有偏但修正量拍脑袋",
    "naive_ignore:floor": "假设下限为 0",
}


def task_text(card, tpl, it, r):
    sp, d = it["spec"], r["design"]
    better = "lower" if card.better == "min" else "higher"
    if tpl == "M1":
        return ("**Card**: %s (%s).\n\n**Notebook (free)**: %s. Context: %s.\n\n**Question**: what is the expected value in "
                "the *production* context at x = %.3f?\n\n**Lab**: you may buy up to %d production runs at any of x ∈ %s "
                "(each bought run has the lab's standard seed noise; the notebook pilots' noise multiplier is stated above). "
                "Answer with one number." % (
                    card.title, card.axis, sp["notebook"], card.ctx_story, sp["target"], sp["n"], sp["grid"]))
    if tpl == "M2":
        return ("**Card**: %s (%s).\n\n**Setting**: %d units (eval domains / problem categories) at the target config "
                "x = %.3f. %s; the threshold is c = %s.\n\n**Notebook (free)**: %s.\n\n**Question**: the median over all %d "
                "units of the expected value at x = %.3f.\n\n**Lab**: up to %d runs (%d per unit). Answer with one number." % (
                    card.title, card.axis, sp["K"], sp["target"], card.censor_story, sp["c"], sp["notebook"], sp["K"],
                    sp["target"], sp["n"], sp["runs_per_unit"]))
    if tpl == "M3":
        return ("**Card**: %s (%s), all recipes at x = %.3f (%s is better).\n\n**Notebook (free)**: %d recipes, %s; "
                "recipe 0 is the incumbent. The notebook picked recipe %d.\n\n**Question**: the true expected value of the "
                "recipe the notebook picked.\n\n**Lab**: up to %d runs of any recipe. Answer with one number." % (
                    card.title, card.axis, sp["x0"], better, sp["G"], sp["notebook"], sp["selected"], sp["n"]))
    if tpl == "M6":
        return ("**Card**: %s (%s).\n\n**Question**: going from x = %.3f to x = %.3f, what fraction of the *reducible* value "
                "(above the floor) is removed?  Floor: %s.\n\n**Notebook (free)**: %s.\n\n**Lab**: up to %d runs at any of "
                "x ∈ %s (floor instrument included). Answer with a fraction." % (
                    card.title, card.axis, sp["xa"], sp["xb"], card.floor_story, sp["notebook"], sp["n"], sp["grid"]))


def block(r, label):
    card = CARD[r["card"]]
    it, _ = BUILD[r["tpl"]](card, r["world"], r["design"], r["seed"])
    T, truth = r["T"], it["truth"]
    L = ["## %s — %s × %s\n" % (label, r["card"], r["tpl"]), "### 题面（agent 看到的）\n", task_text(card, r["tpl"], it, r), ""]
    if r.get("clauses_added"):
        L.append("附加条款（G8：该读法落在 T 外，所以题面要写明）：%s\n" % "；".join(r["clauses_added"]))
    L += ["### 判分侧（agent 看不到）\n",
          "真值 %.5g；容差 T = %.3g（参考方法 90 分位误差 × 2.25）；B_lo（含红队）= %.2f，B_lo（仅基础对手）= %.2f（认证要求 ≥3）；"
          "κ = %.2f；n = %d；现实性标记：%s；驱动常数：%s（%s）。\n" % (
              truth, T, r["B_lo_red"], r["B_lo"], r["kappa"], r["n"], "；".join(realism_flags(r)) or "无",
              r.get("driver_key"), r.get("driver_class")),
          "世界常数：" + ", ".join("%s=%.4g" % kv for kv in r["world"].items()) + "\n",
          "| 路由 | 层 | 值 | 离真值 / T | 代表的失败 |", "|---|---|---|---|---|"]
    rows = [(k, "基础", v) for k, v in it["routes"].items() if k not in it["reference"]] + \
           [(k, "红队", v) for k, v in it["red"].items()]
    for k, tier, v in sorted(rows, key=lambda x: abs(x[2] - truth)):
        L.append("| `%s` | %s | %.5g | %.2f | %s |" % (k, tier, v, abs(v - truth) / T, GLOSS.get(k, "数值捷径")))
    for k, v in it["valid"].items():
        L.append("| `%s` | 合法替代（不计分） | %.5g | %.2f | 与参考同阶的其他正确方法 |" % (k, v, abs(v - truth) / T))
    for k, v in it["free"].items():
        L.append("| `%s` | 免费数据（G7） | %.5g | %.2f | 只用免费数据的诚实估计，必须 ≥2T |" % (k, v, abs(v - truth) / T))
    for k, v in it.get("shown", {}).items():
        L.append("| `%s` | 题面数字（G7） | %.5g | %.2f | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |" % (k, v, abs(v - truth) / T))
    L.append("")
    return "\n".join(L)


def main():
    S2, S1 = {}, {}
    for fn in glob.glob(os.path.join(HERE, "results", "*_search*.json")):
        J = json.load(open(fn))
        if J["na"]:
            continue
        (S2 if J["mode"] == "search" else S1)[(J["card"], J["tpl"])] = J
    out = ["# 已认证题示例（自动生成：`python3 render.py`）\n",
           "每个模板取一题：search T2 中 clean、优先机制驱动、红队裕度最大的那题；另加一道 M1 的“T1 认证但被红队杀”的题，"
           "展示分层证书。路由表按离真值的距离排序；B 由离真值最近的计分路由决定。\n",
           "**注意**：题面是草图。本原型只用门里的模拟参考运行来认证题目，还没有实现给 agent 调用的 lab API"
           "（买运行、读笔记本）；把这些题接到 v5 的 lab 服务上是下一步（README §7）。第一版渲染就查出了一类泄漏："
           "M3 在笔记本恰好选中在位配方时，题面公布的在位值就是答案；M2 的删失阈值 c 可能紧挨中位数。"
           "修复后新增了 `shown` 登记和 G7 的 ≥1T 检查（README §5）。\n"]
    for t in TEMPLATES:
        R = [r for (c, tt), J in S2.items() if tt == t for r in J["results"] if r.get("tier_cert")]
        if not R:
            continue
        R.sort(key=lambda r: (not realism_flags(r), r.get("driver_class") == "mechanism", min(r["B_lo_red"], 12)), reverse=True)
        out.append(block(R[0], "例 %s" % t))
    R = [r for (c, tt), J in S1.items() if tt == "M1" for r in J["results"]
         if r.get("tier_cert") and not r.get("certified") and not realism_flags(r)]
    R.sort(key=lambda r: (r["B_lo_red"] < 2.0, r["B_lo"]), reverse=True)    # a clear kill, then the widest base margin
    if R:
        out.append(block(R[0], "例 M1-T1（分层证书：对基础对手认证、被红队杀）"))
    os.makedirs(os.path.join(HERE, "items"), exist_ok=True)
    open(os.path.join(HERE, "items", "examples.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
