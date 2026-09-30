# v8 排除清单：不计入结果矩阵的运行与题族

排除规则（与 v7 相同，v7 曾误用过一次，已回滚）：

- **只排除两类运行**：
  - 截断：agent 进程没有正常结束。
  - provider 侧拦截：分类器拒答，模型输出为 `<synthetic>` 的 refusal。
- **api_errors / 网关超时不是排除理由。** GPT-6 的运行有 0–6 次 api_errors，gpt-5.5 的 rr10 有 9 次，都照常计分。
- **被拦截的实例不改写题面去绕过拦截。** 做法是换同一模板的另一个池实例重跑，被拦截的运行原样保留作证据。

---

## 1. 被 provider 安全分类器拦截的运行（3 个，均为 claude-fable-5-1）

| 运行 | 实例 | 拦截时点 | lab 调用 | 分类 |
|---|---|---|---|---|
| `rr11__fable__r1` | r_retune s11（tokens8x） | 第 10 个 tick，运行中途 | 29 | `frontier_llm` |
| `rr16__fable__r1` | r_retune s16 | 第 1 个 tick | 0 | `frontier_llm` |
| `rc0__fable__r1` | r_crit s0（batch2x_tokens8x） | 第 1 个 tick，第 1 条消息 | 0 | `frontier_llm` |

三次拦截的 stream 里是同一段 refusal 说明：

> "This request was blocked as our automated systems flagged that it may violate Anthropic's Terms of Service restrictions on developing competing model products."

`api_refusal_category = "frontier_llm"`, `stop_reason = "refusal"`

**处理方式**：

- rr11、rr16 不计入 Fable 的 r_retune 分母。r_retune 的 Fable 结果按 4 个有效运行报告：rr10、rr14、rr20、rr29。
- rc0 换成同模板的 rc10（tokens16x）。之后又补跑了 rc18、rc20、rc22。

**对 benchmark 的含义（重要）**：

- 在 r_retune / r_crit 两个族上，Fable 被拦截的比例为 3/13（r_retune 6 次中 2 次，r_crit 7 次中 1 次）。这两个族的题面是"为下一次预训练重调配方"。
- 这类 AI4AI 题（预训练配方、扩展律）本身就落在 provider 的"开发竞争模型"类分类器的敏感区。
- 用 Fable 做 AI4AI 评测时，需要预留重跑额度，并单独报告拦截率。
- 不要为了降低拦截率而改写题面：那样测到的就不再是同一道题。

---

## 2. 被裁掉的题族

| 题族 | 裁掉原因 | 证据 | 保留形式 |
|---|---|---|---|
| **a_stack** | **搜索闸门否证**。它通过了 v7 全部九道闸门、8 个闭式实例条件和两条原则消融，但一个零洞察的暴力策略 **18/18 全过**，只用了 65–88% 的预算。暴力策略的做法：在目标负载上把所有可行子集各测 1 次、取 argmax，再从枚举数据里直接读出 LOO。<br>根因：可行集只有 22–36 个，4×n ≤ 144 < 160 的预算；朴素测量本身是无偏的。 | `l15/tasks/a_stack.py` 文件头 | 文件保留，作为 `search_gate` 的标定用例。后继题族是 a_span（目标负载结构性不可测） |
| v7 的 k1 / k2 / k3 / k4 / k6 / k8 | 不是 v8 题族。它们在 `l15/tasks/` 里只是 v7 的原样继承，没有过 v8 族闸门（p_ablation / item_activity / mutation / search / pool），所以不计入 v8 的任何结果 | v7 目录 | 原样保留 |

---

## 3. 饱和题族（前沿全过，保留为对照，列入退休候选）

| 题族 | Fable | GPT-6 | 为什么保留 |
|---|---|---|---|
| b_control | 2/2 | 2/2 | 正确性证书完整。它是"默认实验卫生（控制变量、2×2 析因）会打穿混杂类陷阱"这一结论的证据 |
| b_control_min | 2/2 | 2/2 | 与 b_control 在同一批实例上构成最小披露对照，是"披露杠杆被否证"的证据 |
| a_span / a_span_min | 4/4 | 2/4 | GPT-6 的两次失败都是远失败（R1 62% vs 需要 90%；R3 差距 0.13 vs 容差 0.015），但 Fable 全过，所以不能算作"困住前沿" |

---

## 4. 迭代过程中作废、没有进入实跑的配置

以下配置由 S2 条件或 S5 审计自动暴露，改掉后才发证书，不涉及任何模型运行：

| 族 | 作废配置 | 原因 |
|---|---|---|
| r_crit | CENTRE_DEPTH=2 | batch 变体的预算条件 C5 100% 失败（深度 2 的 run 每次花费 16） |
| r_crit | AHI=0.45（收窄 α） | 池上背诵覆盖率反而升到 0.46–0.47，已回滚到 0.70 |
| r_crit | 固定 3 点定中心（不走步） | oracle 可行性条件 C3 在 batch 变体上失败约 50% |
| r_crit | 初版变体集（沿用 r_retune 的 4x 级变体，含 batch4x_tokens4x） | 杠杆臂 ≤2 octave。初版池上的最佳固定猜测覆盖 44–88%（原始先验上只有 23–43%），一个"介于 sqrt 与 linear 之间"的对冲猜测就能过一大半。定稿只保留 lr 移动 ≥3 octave 的五个变体 |
| r_crit | 题面沿用 r_retune 的"224 credits / 0.0008" | 由 I8（题面数字由代码替换并 assert）发现并修正 |

---

## 5. 不排除、但需要标注的运行条件

以下情况**不构成排除理由**，照常计分，只在报告里标注（`L15_v8_report.md` §2.4）：

| 情况 | 涉及的运行 | 为什么不排除 |
|---|---|---|
| GPT-6 在连续两次 API 超时后被降到 medium effort | as1、as2、bcm14、rr11、rr29、rc10、rc22（7/18） | r 系四次降级都发生在全部 lab 花费之后（按 ledger 时间戳核对），实验设计都在 high effort 下完成 |
| Claude 的时间片空转：回合比 tick 长时被打断，思考丢失 | r 系 Fable 运行空 tick 占比 0/8–22/36；sonnet-5 rr11 为 20/25 | 这是 harness 缺陷，不是截断：agent 进程每次都正常恢复并完成交付。和通过/失败不相关（空转最重的 rc22 通过了）。方向上对被测方不利，所以 Fable 的 r 系结果按下限解读 |
| 实跑实例不在证书池内 | `bcm5__fable__r1`、`bcm5__gpt6__r1`（b_control_min s5） | s5 通过了该族的逐实例闸门（screen + gate accept），只是没被池级选择挑中；选它是为了和 b_control s5 同实例配对。池级闸门管题库多样性，不管单实例正确性（报告 §2.3） |

---

## 6. 判分器修正后重评的运行

| 运行 | 首评 | 修正 | 重评 |
|---|---|---|---|
| `bcm5__fable__r1` | fail（R2 区间宽度 0.4500000000000002 > 0.45） | 所有宽度比较改为 `> CAP + 1e-9`（b_control / b_control_min / a_span / a_stack） | PASS |

这类错误不能记为模型失败。v8 因此规定：**每个前沿 fail 都要先做边界核查**（距阈值倍数、最近诱饵、轨迹机制），见 `L15_v8_report.md` §3。
