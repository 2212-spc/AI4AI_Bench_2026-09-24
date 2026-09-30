"""Heterogeneity report: how different are the blueprints, measured on what they test (mechanism cards,
obstacle templates, item kinds), not on surface text.

  python3 -m scalelab.hetero [--tasks DIR]  > docs/heterogeneity.md

Instances of one blueprint share cards/obstacles/kinds by construction (they differ only in world draw),
so the unit of heterogeneity is the blueprint, not the task directory."""
import argparse, glob, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scalelab import build as B, cards as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BPS = ["t01_lr_horizon", "t02_stability_edge", "t03_filter_repeat", "t07_checkpoint_ledger", "t05_contaminated_benchmark"]
SHORT = {"t01_lr_horizon": "T01", "t02_stability_edge": "T02", "t03_filter_repeat": "T03",
         "t07_checkpoint_ledger": "T07", "t05_contaminated_benchmark": "T05(未发布)"}


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def kinds_of(bp, tasks_dir):
    mod = B.load(bp)
    keys = sorted(glob.glob(os.path.join(tasks_dir, mod.ID + "-ws*", "tests", "key.json")))
    if keys:
        return sorted({it["kind"] for it in json.load(open(keys[0]))}), len(keys)
    inst = B.build_instance(bp, 1)                     # unshipped blueprint: read kinds off one build
    return sorted({it["kind"] for it in inst["cal"]["items"]}), 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=os.path.join(ROOT, "tasks"))
    a = ap.parse_args()
    info = {}
    for bp in BPS:
        mod = B.load(bp); k, n = kinds_of(bp, a.tasks)
        info[bp] = {"cards": list(mod.CARDS), "obs": list(mod.OBSTACLES), "kinds": k, "n_tasks": n}
    all_cards = [set(v["cards"]) for v in info.values()]
    common = set.intersection(*all_cards)
    print("# 题族异质性（按考点而非表面文本）\n")
    print("由 `python3 -m scalelab.hetero` 生成。同一蓝图的不同实例共享卡片/障碍/题型，只在世界抽样上不同，"
          "所以异质性的单位是**蓝图**，不是任务目录。\n")
    print("| 蓝图 | 已发布实例 | 机制卡 | 障碍模板 | 题型 | 独有卡片 |\n|---|---|---|---|---|---|")
    for bp, v in info.items():
        others = set().union(*[set(w["cards"]) for b2, w in info.items() if b2 != bp])
        own = sorted(set(v["cards"]) - others)
        print("| %s | %d | %s | %s | %s | %s |" % (SHORT[bp], v["n_tasks"], ", ".join(v["cards"]), ", ".join(v["obs"]),
              ", ".join(v["kinds"]), ", ".join(own) or "-"))
    print("\n所有蓝图共有的底座卡片：%s（损失面 + 种子噪声，是\"实验室\"本身，不是考点）。\n" % ", ".join(sorted(common)))
    print("| 对 | 卡片 Jaccard | 障碍 Jaccard | 题型 Jaccard | 卡片∪障碍 Jaccard |\n|---|---|---|---|---|")
    for x, y in itertools.combinations(BPS, 2):
        u, v = info[x], info[y]
        print("| %s–%s | %.2f | %.2f | %.2f | %.2f |" % (SHORT[x], SHORT[y], jac(u["cards"], v["cards"]), jac(u["obs"], v["obs"]),
              jac(u["kinds"], v["kinds"]), jac(u["cards"] + u["obs"], v["cards"] + v["obs"])))
    covered = sorted(set().union(*[set(v["obs"]) for bp, v in info.items() if v["n_tasks"]]), key=lambda s: int(s[1:]))
    print("\n已发布任务覆盖的障碍模板：%s（%d/%d）。未覆盖：%s。\n" % (", ".join(covered), len(covered), len(C.OBSTACLES),
          ", ".join(o for o in C.OBSTACLES if o not in covered)))
    var = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.tasks, "t0?b-*")))
    if var:
        print("紧预算变体（同一蓝图、同一世界、只收紧预算，不计为新考点）：%s。\n" % ", ".join(var))
    print("去掉底座卡片 %s 后，各蓝图的考点卡片集合：" % "/".join(sorted(common)))
    for bp, v in info.items():
        print("- %s: %s" % (SHORT[bp], ", ".join(c for c in v["cards"] if c not in common) or "-"))


if __name__ == "__main__":
    main()
