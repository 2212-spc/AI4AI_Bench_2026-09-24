# L1.5 Bench v8（2026-09-28 分支）

本目录是 v8 的完整交付，**自包含**，不依赖上层目录的任何文件。上层旧版本目录（v2–v7）没有做任何修改。

v8 的主题是一条"可迭代加难、但正确性不退化"的造题 pipeline：

- **两张证书**：
  - 正确性证书：闸门全自动，零模型调用，可从头重跑。
  - 难度证书：前沿实跑，并对每个 fail 做边界核查。
- **轨迹驱动硬化（S7）**：用上一版题的前沿轨迹，把旧 oracle 变成新诱饵。
- 示例是 r_retune → r_crit 这一轮。

## 从哪读起

| 文件 | 内容 |
|---|---|
| `L15_v8_report.md` | **主报告**（中文）。§0 是一页结论，§1 逐条回答五点要求，§2 是全部实测分数和运行条件，§3 是每个前沿 fail 的边界核查，§7 是局限 |
| `DESIGN.md` | 设计与 pipeline 文档：S0–S7 各阶段、闸门清单（§4）、定律（§5）、r_retune → r_crit 的完整硬化示例（§6）、背诵覆盖率（§7）、从 MLS-Bench / TB-Science 借鉴的部分（§8）、异质性矩阵（§9） |
| `EXCLUDED.md` | 排除与作废记录：哪些 run 没进统计及原因（provider 分类器拦截 3 次）；被裁掉的题族（a_stack）；饱和族；作废配置；不排除但需要标注的运行条件；判分器修正后重评的运行 |

## 本轮结果（详见报告 §0、§2）

| 族 | Fable 5.1 | GPT-6 | 状态 |
|---|---|---|---|
| **r_crit**（硬化后） | 2/6（4 个 fail：39.8×、33.5×、3.5×、29.4×） | 0/4（204–960×） | 唯一对两个前沿都有区分力的族；两次 Fable 通过都在 batch16x 上，混杂未排除 |
| r_retune（第一版） | 2/4（两个 fail 在阈值 2–3× 内） | 0/6（22–557×） | 被 r_crit 取代 |
| a_span / a_span_min | 4/4 | 2/4 | 饱和，列入退休候选 |
| b_control / b_control_min | 4/4 | 4/4 | 饱和，作对照保留 |

## 目录

- `l15/`：生成器、闸门与评分器，均以模块方式运行（`python3 -m l15.<模块>`）。
  - `l15/tasks/`：题族文件，每个文件包含世界、oracle 和诱饵。v8 的六个族是 `r_crit.py`、`r_retune.py`、`a_span.py`、`a_span_min.py`、`b_control.py`、`b_control_min.py`。
  - `l15/tasks/a_stack.py`：已裁掉的族，保留作搜索闸门的标定用例。
  - `l15/tasks/k*.py`：v7 原样继承，不属于 v8 结果。
  - `v8gates.py`：v8 族闸门（p_ablation / item_activity / mutation / search / pool）。
  - `v8cert.py`：可续跑、并行的证书驱动。
  - `gates.py`：v7 闸门。
  - `build.py` 出题；`grade.py` 判分；`server.py` 是 lab 服务。
- `tasks/`：已构建的 21 个实例。
  - 每个实例包含题面 `instruction.md`、`task.toml`、环境 `environment/`、隐藏真值 `hidden/` 和提示 `hints/`。
  - 其中 r_crit 7 个、r_retune 6 个，另有 a_span、a_span_min、b_control、b_control_min 各 2 个。
- `v8_reports/`：六个族的正确性证书，均为 `ok: True`。
  - `<族>.json` 是证书报告。
  - `<族>.state.json` 是可续跑状态，其中包含接受池。
- `harness/`：跑模型用的沙箱、网关和驱动。
  - `cc_agent.py`：Claude 系 agent。
  - `gpt_agent.py`：GPT 系 agent。
  - `gateway.py`：本地 API 网关，唯一持有上游 key 的进程。
  - `drive.py`：按时间片推进各个 run。
  - `launch.py`：启动一个 run。
  - `test_no_key.py`：断言"沙箱启动参数和环境里不得出现真实 key"。交付前跑过，ALL PASS。
- `run_evidence/`：46 个 run 的证据，包括 `grade.json`、`traj.jsonl`、`lab_ledger.jsonl`、`app/` 产物、`state.json` 和 `cc_stream_*.jsonl`。
- `gate_evidence/`：沙箱完整性审计（报告 §5）。
  - `probe_audit.py` 是审计脚本，`probe_audit.json` 是审计结果。
  - 审计内容是逐 run 统计"找判分器 / 读进程环境 / 扫 lab 目录"类探测。
- `research/`：调研笔记。
  - `r1`：MLS-Bench 与 ML 研究类 bench。
  - `r2`：TB-Science 的 rubric 与自动出题。

## 已做的脱敏

- **不含**：`home/`、`sbx_tmp/`、`app/.lab_token`。
- **lab token 字段**：`run_evidence/*/lab.json` 的 `token` 字段一律改为 `<redacted>`。字段保留，schema 不变。
- **agent 自己打印出的 lab token**：有 14 个文件里 agent 自己 `cat` 出了 lab token，已就地替换为 `<redacted-lab-token>`。
- **外层代理凭据**：`rc10__gpt6__r1` 的 `traj.jsonl` 和 `state.json` 里，GPT-6 读进程环境时带出了外层代理凭据，已替换为 `<redacted-proxy-cred>`。
  - 泄漏途径是沙箱 PID 1 的环境变量，已在 harness 里修复（报告 §5）。
- **sk- 扫描**：全树扫过 `sk-[A-Za-z0-9_-]{16,}`，剩下的命中只有三类：
  - 字面占位符 `sk-sandbox-placeholder-not-a-credential`，出现在 `harness/gateway.py`、本 README 和主报告中；
  - `test_no_key.py` 里的假 key；
  - 两个 `state.json`（`rc10__gpt6__r1`、`rr10__gpt55__r1`）中 base64 `encrypted_content` 字段内部偶然出现的子串。正则匹配到的连续串长 1476 和 9023 字符，所在字段全长 4388 和 17548 字符。
- **真实 key**：网关用的真实 key 不在本目录任何文件中，已按字符串比对确认。
- **`__pycache__/`**：这些目录没有删掉（沙箱不允许删除），内容无害。

## 复现

**正确性证书**：零模型调用，可从头重跑。

```bash
python3 -m l15.v8cert <族> <lo> <hi> <target> <state.json> [time_limit_s]
# 例：python3 -m l15.v8cert r_crit 0 240 12 /tmp/r_crit.state.json 150
# 输出 INCOMPLETE 时重复同一命令即可续跑；最终打印报告与 OK / NOT OK
```

六个族的参数（`lo`、`hi`、`target`）记录在 `v8_reports/<族>.state.json` 中。例如 r_crit 是 `0 240 12`。

**出题**：

```bash
python3 -m l15.build <题族模块> <seed> <输出目录>
```

`tasks/` 下已构建的实例目录是写保护的，重新构建时请换一个新名字。

**判分**：

```bash
python3 -m l15.grade <run_dir> [...]
```

交付前对 `run_evidence/` 里 43 个有交付物的 run 全部重新判分，与原始 `grade.json` 0 处差异。另外 3 个被拦截的 run 没有交付物。

**跑模型**：需要自备网关凭据，凭据不随交付分发。

```bash
python3 harness/launch.py <task_dir> <cc|gpt> <model> <hint 0|1|2> <run_name> [effort]
SEC_DIR=<凭据目录> timeout 165 python3 harness/drive.py 158 [run_dir ...]
```

- run 目录默认在 `LAB_RUN_ROOT=/tmp/v7/runs`。
- `MAX_PAR` 控制并行数，`MAX_RUN_MIN` 控制单个 run 的 agent 时间上限。
- 已知缺陷：Claude 系 harness 有时间片空转（报告 §2.4、§6 第 6 条）。在修掉之前，r 系 Fable 的结果应按下限解读。
