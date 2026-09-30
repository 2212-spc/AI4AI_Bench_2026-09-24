# MATRIX — AV 编译器产出矩阵（自动生成，勿手改；`python3 analyze.py`）

所有数字来自 `results/*.json`，零模型调用。oneshot / loop：每格 40 次门评估；search：每格 48 个世界 × ≤48 次探针，每个世界只记最优设计（fertility = 有可认证设计的世界占比；search 是启发式，所以这是下界）。T2 = 过全部门含红队 G9；T1 = 只对模板基础对手认证（不计 G9）。T1 与 T2 用同一批世界（配对）。clean = 无现实性标记（r_nb<4 且 n<300）。判定：T2 可育世界 ≥50% = 可育，≥12.5% = 世界受限，否则近乎不育。

## A. 产出与可育性矩阵

| 卡 × 模板 | oneshot 认证/40 (clean) | loop 认证/40 (clean) | search T2 可育/48 (clean) | 探针/可育世界 | search T1 可育/48 (clean) | T1→T2 被红队杀 | 判定 |
|---|---|---|---|---|---|---|---|
| K1_compute × M1 | 0 (0) | 0 (0) | 0 (0) | ∞（2304 探针 0 题） | 16 (9) | 16 | 近乎不育（T2）；T1 可育 → 分层候选 |
| K1_compute × M2 | 1 (1) | 3 (3) | 36 (33) | 49 | 48 (48) | 42 | 可育 |
| K1_compute × M3 | 2 (2) | 1 (1) | 26 (22) | 83 | 32 (29) | 21 | 可育 |
| K1_compute × M6 | 14 (14) | 13 (7) | 42 (38) | 20 | 44 (41) | 3 | 可育 |
| K2_lrbasin × M1 | 0 (0) | 2 (1) | 8 (5) | 288 | 25 (18) | 23 | 世界受限 |
| K2_lrbasin × M2 | 0 (0) | 1 (1) | 15 (10) | 152 | 43 (38) | 42 | 世界受限 |
| K2_lrbasin × M3 | 2 (2) | 0 (0) | 26 (21) | 83 | 28 (27) | 15 | 可育 |
| K2_lrbasin × M6 | N/A | N/A | N/A | - | N/A | - | **N/A**：卡无物理下限（floor_kind=None），M6 前提不成立 |
| K3_repeat × M1 | 0 (0) | 0 (0) | 0 (0) | ∞（2304 探针 0 题） | 24 (16) | 24 | 近乎不育（T2）；T1 可育 → 分层候选 |
| K3_repeat × M2 | 0 (0) | 0 (0) | 18 (13) | 124 | 47 (43) | 45 | 世界受限 |
| K3_repeat × M3 | 2 (2) | 3 (1) | 22 (20) | 101 | 29 (27) | 19 | 世界受限 |
| K3_repeat × M6 | 6 (6) | 11 (3) | 47 (43) | 14 | 47 (41) | 0 | 可育 |
| K4_passk × M1 | 0 (0) | 0 (0) | 1 (1) | 2304 | 24 (13) | 24 | 近乎不育（T2）；T1 可育 → 分层候选 |
| K4_passk × M2 | 0 (0) | 0 (0) | 7 (3) | 329 | 27 (21) | 25 | 世界受限 |
| K4_passk × M3 | 0 (0) | 1 (0) | 21 (15) | 108 | 30 (24) | 14 | 世界受限 |
| K4_passk × M6 | N/A | N/A | N/A | - | N/A | - | **N/A**：卡无物理下限（floor_kind=None），M6 前提不成立 |
| K5_critbatch × M1 | 0 (0) | 0 (0) | 0 (0) | ∞（2304 探针 0 题） | 8 (8) | 8 | 近乎不育（T2） |
| K5_critbatch × M2 | 1 (1) | 1 (1) | 23 (17) | 95 | 37 (29) | 27 | 世界受限 |
| K5_critbatch × M3 | 0 (0) | 1 (0) | 15 (10) | 151 | 19 (13) | 9 | 世界受限 |
| K5_critbatch × M6 | 0 (0) | 0 (0) | 9 (6) | 256 | 19 (14) | 19 | 世界受限 |

合计（18 个非 N/A 格）：oneshot 28/720 认证，loop 37/720，search T2 316/864 世界可育（clean 257），T1 547/864。

配对检验：T2 可育的 316 个世界里，T1 搜索漏掉 17 个（T2 ⊂ T1，所以这是搜索的漏检率 5%；可育性是下界）。

**每 100 次门评估的认证数**（同一计算量下三种 proposer 的效率；search 每世界只计 1 个）：
oneshot 3.9（28/720），loop（规则修复）5.1（37/720），search（门作 oracle 的爬山）0.9（316/37120；但 search 在找到认证后还会继续找 clean 且 B≥6 的设计，按“首次认证所需探针”算，可育世界的中位数是 18 次、P75 29 次）。

**产出分布**（同样是零模型调用，门评估单核约 5–6 ms 一次（中位数），所以瓶颈不是评估次数，而是“每个世界能不能出题”和题目在格子间是否分散）：

| proposer | 有 ≥1 题的格 /18 | 题数 | 最集中的格（占比） | 按模板 M1/M2/M3/M6 |
|---|---|---|---|---|
| oneshot | 7 | 28 | K1_compute × M6（50%） | 0/2/6/20 |
| loop | 10 | 37 | K1_compute × M6（35%） | 2/5/6/24 |
| search T2 | 15 | 316 | K3_repeat × M6（15%） | 9/99/110/98 |
| search T1 | 18 | 547 | K1_compute × M2（9%） | 97/202/138/110 |

oneshot/loop 的题有一半以上来自 M6 这类“随手就能认证”的格；search 把每个世界的问题答到了底，所以它的价值不在单位评估效率，而在：(1) 覆盖面（难格也能出题）；(2) 给出“这个世界能不能出题”的可证伪答案——剩下的不育由世界常数决定（见 C、G），不是 proposer 不够聪明。当世界本身很贵（目标 2 的移植世界需要人工/模型整理），按世界计的产出才是该优化的量。

## B. 已认证题（search T2）：裕度、绑定对手、驱动常数

绑定对手 = 加入红队后离真值最近的捷径（B 由它决定）。驱动常数 = 把每个世界常数沿先验范围挪 10%，该捷径偏差变化（以 T 计）最大的那个；stat = 噪声/总体离散类常数（与卡的机制无关），mechanism = 卡的机制常数。

| 模板 | 认证数 | B_lo(含红队) 中位 [IQR] | 绑定对手族 | 驱动类 stat/mech | 最常见驱动常数 | 路由间隔≥2T 占比 |
|---|---|---|---|---|---|---|
| M1 | 9 | 3.2 [3.1, 3.7] | naive_ignore:ctx 7, R3 2 | 0 / 9 | a(K2) 5, beta(K2) 3, s1(K4) 1 | 0% |
| M2 | 99 | 4.7 [3.6, 6.1] | drop 41, naive_ignore:unit_slope 37, B_guess 21 | 85 / 14 | tau2(K1) 20, tau(K1) 16, tau2(K3) 11 | 22% |
| M3 | 110 | 4.6 [3.6, 5.9] | B_guess 47, R7 33, R1 30 | 97 / 13 | tau_r(K2) 15, s0(K3) 15, s0(K1) 14 | 14% |
| M6 | 98 | 8.5 [6.5, 15.9] | B_prior 72, R4 24, R8 2 | 0 / 98 | E(K3) 38, E(K1) 25, al(K1) 17 | 99% |

读法：M2/M3 若驱动类以 stat 为主，说明这些题的难度来自统计结构（删失比例、赢家诅咒的噪声/离散比），与机制卡无关——换一张卡只是换皮（见 E 的跨卡坍缩）。M1/M6 若以 mechanism 为主，说明难度绑定在卡的物理上，换卡才是真正的新题。

## C. 不育世界为什么不育（search T2 中未认证世界的最优设计卡在哪道门）

| 卡 × 模板 | 不育世界数 | 首个失败门（最优设计） | 该门处最近对手 |
|---|---|---|---|
| K1_compute × M1 | 48 | G7_free 20, G5_B 16, G9_redteam 12 | R1 12, naive_ignore:ctx 9, R3 3 |
| K1_compute × M2 | 12 | G9_redteam 12 | naive_ignore:unit_slope 10, drop 2 |
| K1_compute × M3 | 22 | G5_B 9, G9_redteam 8, G7_free 5 | B_guess 8, R1 5, R7 4 |
| K1_compute × M6 | 6 | G7_free 5, G9_redteam 1 | R4 1 |
| K2_lrbasin × M1 | 40 | G9_redteam 14, G5_B 13, G7_free 7 | R1 12, naive_ignore:ctx 8, R3 6 |
| K2_lrbasin × M2 | 33 | G9_redteam 28, G5_B 5 | naive_ignore:unit_slope 18, drop 13, B_guess 2 |
| K2_lrbasin × M3 | 22 | G5_B 10, G7_free 7, G9_redteam 5 | R1 8, B_guess 4, R7 3 |
| K3_repeat × M1 | 48 | G9_redteam 27, G5_B 12, G7_free 9 | naive_ignore:ctx 16, R3 11, R1 8 |
| K3_repeat × M2 | 30 | G9_redteam 28, G5_B 2 | naive_ignore:unit_slope 20, drop 8, B_guess 2 |
| K3_repeat × M3 | 26 | G5_B 12, G9_redteam 9, G7_free 5 | R7 7, R1 7, B_guess 7 |
| K3_repeat × M6 | 1 | G7_free 1 | - |
| K4_passk × M1 | 47 | G9_redteam 26, G5_B 14, G7_free 7 | naive_ignore:ctx 25, B_prior 6, R1 5 |
| K4_passk × M2 | 41 | G5_B 20, G9_redteam 19, G4_tight 2 | naive_ignore:unit_slope 16, B_guess 14, drop 8 |
| K4_passk × M3 | 27 | G7_free 12, G9_redteam 8, G5_B 7 | R1 6, B_guess 6, R7 3 |
| K5_critbatch × M1 | 48 | G5_B 41, G7_free 4, G9_redteam 3 | R1 37, R7 4, naive_ignore:ctx 3 |
| K5_critbatch × M2 | 25 | G5_B 16, G9_redteam 9 | drop 12, B_guess 9, naive_ignore:unit_slope 4 |
| K5_critbatch × M3 | 33 | G5_B 17, G7_free 12, G9_redteam 4 | R1 14, B_guess 4, R7 3 |
| K5_critbatch × M6 | 39 | G9_redteam 16, G6_kappa 13, G7_free 6 | R4 16, R8 4 |

## D. 红队杀伤率（E2）：只对基础对手认证的题，有多少会被“校准过的迁移/假设型”对手杀掉

| 模板 | T1 认证世界 | 其中 T2 仍认证 | 被红队杀 | 杀伤率 | 杀手路由 |
|---|---|---|---|---|---|
| M1 | 97 | 2 | 95 | 98% | naive_ignore:ctx:ratio@near 40, R3:ctx0_slope@near 31, naive_ignore:ctx:offset@near 24 |
| M2 | 202 | 21 | 181 | 90% | naive_ignore:unit_slope 126, drop:censored_mean_nb_shift 55 |
| M3 | 138 | 60 | 78 | 57% | B_guess:notebook_plus_1se 55, R7:top2_mean 23 |
| M6 | 110 | 88 | 22 | 20% | R4:secant_floor 22 |

注意：“其中 T2 仍认证”是 T1 搜索找到的最优设计在加入红队后仍认证的数目；A 表的 T2 列是另一次以 T2 为目标的搜索，所以两者不同（例如 M1：T1 最优设计几乎全被杀，但以 T2 为目标仍能在少数世界找到设计）。杀伤率衡量的是“只对基础对手优化出来的题”有多不可靠。

oneshot+loop 中不计红队即可认证的 173 题里，108 题被红队杀（62%）。这是“只对作者想到的对手认证”的假阳性率下界。

## E. 策展：签名去重与跨卡坍缩

签名 = (卡, 模板, 绑定对手族, 驱动类或机制常数, ⌊log2 B_lo⌋)。策展器每个签名最多保留 2 题。跨卡签名去掉“卡”；若同一跨卡签名出现在 ≥2 张卡上且驱动类为 stat，这些题是同一考点换皮。

T2 认证 316 题 → 98 个签名 → 策展后保留 161 题（51%）。

| 模板 | 认证 | 签名数 | 策展保留 | 跨卡签名数 | 跨 ≥2 卡的 stat 签名（覆盖题数） |
|---|---|---|---|---|---|
| M1 | 9 | 6 | 8 | 6 | 0（0 题） |
| M2 | 99 | 31 | 52 | 14 | 6（82 题） |
| M3 | 110 | 39 | 65 | 16 | 8（97 题） |
| M6 | 98 | 22 | 36 | 19 | 0（0 题） |

## F. 现实性：多少可育性依赖不现实的设计

T2 认证 316 题中 59 题（19%）带现实性标记：n 48, notebook noise 25。搜索框刻意越过现实线（r_nb≤6，n≤400），所以这个比例就是“要靠夸张的笔记本噪声或预算才能出题”的份额。

## G. 哪些世界常数预测可育性（AUC，48 世界/格，|AUC−0.5|≥0.25 才列出）

AUC = P(可育世界的该常数 > 不育世界的该常数)。>0.5 表示该常数越大越可育。n 很小，只作为假设线索。

| 卡 × 模板 | 可育/48 | 强预测常数 |
|---|---|---|
| K1_compute × M1 | 0 | （少数类 <4 个世界，不计算） |
| K1_compute × M2 | 36 | tau2 0.94*, al 0.15 |
| K1_compute × M3 | 26 | - |
| K1_compute × M6 | 42 | - |
| K2_lrbasin × M1 | 8 | beta 0.87 |
| K2_lrbasin × M2 | 15 | s0 0.20* |
| K2_lrbasin × M3 | 26 | tau_r 0.83* |
| K3_repeat × M1 | 0 | （少数类 <4 个世界，不计算） |
| K3_repeat × M2 | 18 | - |
| K3_repeat × M3 | 22 | tau_r 0.78* |
| K3_repeat × M6 | 47 | （少数类 <4 个世界，不计算） |
| K4_passk × M1 | 1 | （少数类 <4 个世界，不计算） |
| K4_passk × M2 | 7 | tau2 0.91* |
| K4_passk × M3 | 21 | tau_r 0.77* |
| K5_critbatch × M1 | 0 | （少数类 <4 个世界，不计算） |
| K5_critbatch × M2 | 23 | - |
| K5_critbatch × M3 | 15 | tau_r 0.87* |
| K5_critbatch × M6 | 9 | bc 0.83 |

（* = stat 类常数）

## H. 可归因性演示（E4）：隐藏路由会产生“无法解释的错答”，红队库补上后才可归因

取 T1 认证、T2 被杀的题，模拟一个“用红队路由作答”的 agent（例如用笔记本曲线 + 一个生产点做常数偏移迁移）。只拿基础对手库做归因时，这个答案要么被判对（假认证：该捷径落在 T 内），要么被判错但不靠近任何已知对手（不可归因）；把红队路由加入库后才能归因到具体的错误假设。

| 模板 | 被杀题 | 判对（红队路由落在 T 内） | 判错且基础库不可归因、红队库可归因 | 判错且基础库可归因 |
|---|---|---|---|---|
| M1 | 95 | 45 | 49 | 1 |
| M2 | 181 | 116 | 65 | 0 |
| M3 | 78 | 29 | 49 | 0 |
| M6 | 22 | 8 | 14 | 0 |

“判对”一列：这些被杀题里，红队路由在该世界恰好落在 T 内，用它作答会被判对。部分原因是它的假设在该世界近似成立。以 M2 的 `naive_ignore:unit_slope` 为例，它假设各单元从笔记本配置到目标配置的平移相同；违背量取各单元平移的四分位距（以 T 计），判对与判错世界的中位数对比：K1 1.53 T（判对 21 题）对 2.86 T（判错 11 题）；K2 1.44 T（判对 25 题）对 2.90 T（判错 7 题）；K3 1.21 T（判对 23 题）对 2.45 T（判错 9 题）；K4 2.73 T（判对 11 题）对 4.46 T（判错 5 题）；K5 8.25 T（判对 6 题）对 6.41 T（判错 8 题）。判对世界的违背量是否系统性更小、分离是否干净，以这行数字为准。这类题不能区分“做了未检验的假设”和“做对了”，所以红队路由也需要 M1 那样的有效性检验（README §5）。其余被杀题才是真的“基础库不可归因的错答”。

汇总：4/5 张卡上判对世界的违背量中位数更低（K1, K2, K3, K4）；反例：K5。四分位距只是违背的粗代理——该路由的误差取决于平移分布相对中位数的不对称，而不只是离散度，所以不能把它当成可在出题时检查的有效性条件。

- K3_repeat × M1：红队路由 `naive_ignore:ctx:offset@near` 的答案离真值 1.43 T → attr_red_only
- K3_repeat × M1：红队路由 `R3:ctx0_slope@near` 的答案离真值 1.39 T → attr_red_only
- K3_repeat × M1：红队路由 `R3:ctx0_slope@near` 的答案离真值 0.56 T → judged_correct
- K3_repeat × M1：红队路由 `naive_ignore:ctx:offset@near` 的答案离真值 0.95 T → judged_correct

## I. loop 修复规则的实测效果（规则作者错误的证据）

对每条修复动作：应用次数、下一步是否推进到更靠后的门、链条最终是否认证。推进率低的规则就是写错了的规则。

| 模板 | 修复动作 | 次数 | 下一步推进 | 链最终认证 |
|---|---|---|---|---|
| M1 | narrow the bracket (h x0.6) | 64 | 56% | 3% |
| M1 | buy 2x runs | 45 | 7% | 0% |
| M1 | widen bracket (h x1.5) | 18 | 6% | 0% |
| M1 | proposer redraws the design | 13 | 85% | 0% |
| M1 | demand more precision (2x runs, T shrinks) | 9 | 0% | 11% |
| M1 | make the notebook's production pilots noisier (x2) | 7 | 57% | 29% |
| M1 | move target to the largest-context-gap region | 3 | 0% | 0% |
| M2 | censor one more unit (m+1) | 61 | 8% | 3% |
| M2 | make the notebook's config staler | 49 | 8% | 8% |
| M2 | 2x runs | 32 | 22% | 0% |
| M2 | 2 more units (K+2) | 7 | 14% | 0% |
| M3 | 2x runs | 87 | 8% | 10% |
| M3 | noisier notebook eval (r_nb x1.5) | 69 | 14% | 13% |
| M6 | demand more precision (2x runs, T shrinks) | 39 | 28% | 82% |
| M6 | proposer redraws the design | 15 | 60% | 0% |
| M6 | noisier notebook (x2) | 13 | 54% | 62% |
| M6 | lower the largest sold config | 12 | 8% | 0% |
| M6 | start the comparison further from the floor (xa lower) | 4 | 0% | 0% |

