# v10 排除与标注清单

本文件列出所有**不计入**前沿成绩的运行和配置，以及计入但需要注意的条件。规则沿用 v7/v8，并且在看到结果之前就已定下，之后没有改过。

## 1. 排除规则（事先定）

只有以下三类运行被排除：

1. **通道故障（channel_outage）**：模型通道在运行早期就持续报错（至多 2 次成功调用），没有产生可判分的交付物。
2. **供应商分类器终止运行（classifier_block）**：分类器在运行中途结束会话，没有留下可判分的交付物。这类运行用**完全相同的 prompt** 重跑一次，prompt 不改写、不规避。
3. **harness 冒烟测试（smoke）**：本来就不判分。

以下情况**不构成**排除理由，这些运行照常计分：

- **API 错误 / 超时（api_errors）**：GPT-6 的 Responses API 单次延迟 40–150 s，常有超时。harness 会重试，错误次数记在 `grade.json` 的 `api_errors` 里。
- **运行中途被分类器拒绝一次、但会话随后继续并正常结束**：见 §3 的 su_L2_s6_fcc。
- **分数接近阈值**：例如 su_L2_s6_gpt6 差 0.0054、su_L2_s6_fcc2 超出 0.003，都照实判。

v7 时我曾把 api_errors 当作排除理由，后来已回滚。v10 从一开始就不这样做。

## 2. 通道故障：8 个 `_fable` 运行

这 8 个运行走 Messages-API 版的 `claude_agent.py`，模型是 claude-fable-5-1：

| 运行 | n_calls | api_errors |
|---|---|---|
| sab_L0_s0_fable | 0 | 3 |
| sab_L0_s4_fable | 0 | 2 |
| sab_L1_s1_fable | 0 | 2 |
| sab_L1_s3_fable | 0 | 2 |
| sab_L1_s4_fable | 2 | 3 |
| sab_L2_s3_fable | 0 | 2 |
| sab_L2_s7_fable | 2 | 3 |
| sab_L2_s9_fable | 0 | 3 |

原因是中转站对 Claude 的 Messages-API 通道不可用，出现过三种情况：

- 返回 503 "no available accounts"；
- 另一条通道返回 403，只服务官方 Claude Code CLI；
- 约 120 s 后以空流关闭连接。

处理办法是让 Fable 改走**官方 Claude Code CLI 二进制**（`harness/cc_agent.py`，运行名后缀 `_fcc`）。harness 没有伪装 CLI 请求头。

这 8 个运行对应的同名 `_fcc` 运行全部完成，所以没有丢失任何任务实例。其中 sab_L1_s4_fable 和 sab_L2_s7_fable 各有 2 次成功调用（只读了数据），随后通道超时并以空流关闭，都没有交付物。

## 3. 供应商分类器（Fable 5.1，类别 reasoning_extraction）

Fable 的 cc 运行共 24 个：serve_ab 9、judge_audit 3、scaleup L1 4、scaleup L2 8。其中 3 个遇到过分类器拒绝：

| 运行 | 发生在 | 结局 | 处理 |
|---|---|---|---|
| sab_L1_s3_fcc | 第 36 回合（stream 006） | 会话终止，没有交付物 | **排除**；用相同 prompt 重跑为 sab_L1_s3_fcc2（PASS） |
| su_L1_s1_fcc | 第 11 回合（stream 001） | 会话终止，repo 停在半途；判分 0.7343 < T 0.785，仅作记录 | **排除**；用相同 prompt 重跑为 su_L1_s1_fcc2（PASS 0.8505） |
| su_L2_s6_fcc | 第 8 个 tick（stream 007，stream 从 000 编号） | 拒绝一次后，CLI 会话在下一个 tick 继续，正常结束并交付 | **不排除**，照常计分（FAIL 0.7053） |

统计：

- 终止运行的拦截率是 **2/24**；
- 出现过任意一次拒绝的比例是 **3/24**。

上述每个运行都**没有改写 prompt**。

## 4. 冒烟测试

`smoke_fable` 和 `smoke_gpt6` 用于验证 harness 和网关连通性，不判分。

## 5. 未认证、被取代、或未开跑的配置（不产生任务）

| 配置 | 状态 | 原因 |
|---|---|---|
| scaleup EPOCHS = 64（e64） | 被取代 | 生产副本 CPU 时间超过每次 bash 调用约 178 s 的上限。改为 EPOCHS = 32，旧证书留在 VM 的 `/tmp/v10/scaleup/scaleup_cert_e64.json`，不进交付 |
| scaleup L1 种子 3 | 未认证 | 诱饵余量 0.0132 < 0.015：团队配置原样（`ship_as_is`）就有 0.7968，离 T 0.81 太近。参考解（规则 wd 0.1）是 0.832，而 wd 0.3 能到 0.864 |
| scaleup L2 种子 1/2/3/4/8/9 | 未认证 | 团队 wd 本身已接近最优，ship 与参考只差 0.017–0.034。种子 3 只因 T 取 0.005 格差 0.0003，种子 8 诱饵余量 0.0082。规则不事后改 |
| serve_ab 第一版 L2（`sab_L2_state`） | 证书 ok = false | fluid 诱饵在饱和点不连续，p_ablation 漏过 1/4；mutation 的 q1_width 小扰动失败。改用 Erlang-B loss 模型作 P3 消融后得到 L2b（ok = true），任务取自 L2b |
| serve_ab L1（`sab_L1_state`） | 被 L1b 取代 | L1 的 P3 诱饵是 `fluid_no_drill`，L1b 改为 `loss_no_drill`，两者都 ok = true。任务 s1/s3/s4 取自 L1b 的准入池 [1, 3, 4, 8, 9, 12, 13, 15]；s4 在 L1 池里被策略闸门拒绝，在 L1b 被准入 |
| serve_ab L0 | 证书 ok = false | **只**因为没有声明 search 策略。L0 是校准档，朴素答案按设计就对，所以不做搜索闸门 |
| judge_audit 种子 3（`tasks/ja_L1_s3`） | 已认证，未开跑 | L1 在 s0–s2 上已饱和（Fable 3/3），没有再花预算 |

## 6. 计入成绩、但需要标注的条件

- **GPT-6 超时**：api_errors 每个运行 0–11 次，逐个列在 `harness/results_table.py` 的输出里。超时会消耗 agent 的墙钟时间，计分照常。
- **cc 时间片**：agent 只在 `drive.py` 的 tick 内推进。每个 tick 约 158 s，每次 bash 调用约 178 s，后台进程随调用一起结束。`MAX_RUN_MIN` = 100。没有运行因为达到上限而被截断：`grade.json` 里 `truncated` 为 true 的只有 §3 中两个被分类器终止的运行，其余 41 个判分运行全部为 false。
- **scaleup 单独判分**：判分和证书都不与 agent 或其它作业并行，因为同机争用会把 CPU 读数从 17.5 s 抬到 48 s（17.5 s 见 `certs/scaleup_l2_cert.json` 的 `job_cpu_s`；48 s 是开发时在 VM 上观测到的，交付里没有日志）。所有 scaleup 判分都是在两次 drive 调用之间单独执行的。
- **判分器修复（前 13 个判分运行之后加入）**：`run_job` 增加了"跨进程总 CPU"检查（`scaleup.py` 第 218–223 行）。RLIMIT_CPU 按进程计，一个 fork 多个 worker 的作业可能总共超过 120 s。加入前已判分的 13 个运行里，最大总 CPU 是 53.4 s，所以不改变任何已有判分；之后的前沿运行里也没有交付物触发这条检查。
- **边界核查的一处日志不一致**（su_L2_s6_gpt6 的 distinct-N 反事实）：只改了 `train()` 里计算 wd 的那一行，而 member 打印行用的是另一处计算，所以日志里仍显示旧的 wd。准确率从 0.7996 升到 0.8366，证明改动生效了。结果文件原样保留。

## 7. 饱和档（照实计分，但不算"困住前沿"的证据）

| 档 | 有效运行结果 | 说明 |
|---|---|---|
| serve_ab L0（校准） | 4/4 PASS | 按设计朴素答案就对，用来确认题面可读、判分正确 |
| judge_audit L1 | Fable 3/3，GPT-6 2/3 | 默认卫生（随机 pilot 验证 → 分层 verify）即可解 |
| scaleup L1 | 6/6 PASS（0.843–0.893） | 正交改进（噪声鲁棒损失、平均、集成）就能买下余量 |
