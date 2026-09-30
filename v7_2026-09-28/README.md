# L1.5 Bench v7（2026-09-28 分支）

本目录是 v7 的完整交付，**自包含**，不依赖上层目录的任何文件；上层旧版本目录（v2–v6）未作任何修改。

## 从哪读起

| 文件 | 内容 |
|---|---|
| `L15_v7_report.md` | **主报告**（中文）。0 节是一页结论；2 节是全部实测分数；4 节明确写了本轮没做到的部分 |
| `DESIGN.md` | 设计与 pipeline 文档，含九条闸门（§4.1 + §4.4）、难度杠杆实测（§4.5）、交付前第三轮自查（§4.6） |
| `EXCLUDED.md` | 排除与作废记录：哪些 run 没进统计、为什么，以及我自己写错又改回来的规则 |

## 目录

- `l15/` — 生成器与评分器。`l15/tasks/k*.py` 是六个题族的世界+oracle+诱饵；`gates.py` 是闸门；`build.py` 出题；`grade.py` 判分。均需以模块方式运行，例如 `python3 -m l15.gates k8_post 27 --salts 4`。
- `tasks/` — 已构建的 35 个实例（题面 `instruction.md`、环境 `environment/`、隐藏真值 `hidden/`、提示 `hints/`）。
- `harness/` — 跑模型用的沙箱与网关。`test_no_key.py` 是"启动参数里不得出现真实 key"的断言测试，交付前跑过，全部 PASS。
- `run_evidence/` — 151 个 run 的证据（`grade.json` / `traj.jsonl` / `lab_ledger.jsonl` / `app/` 产物 / `state.json`）。
- `gate_evidence/` — 闸门输出与判分快照。
- `research/` — 对已有 AI4AI bench 的调研笔记。

## 已做的脱敏

- 不含 `home/`、`sbx_tmp/`、`app/.lab_token`。
- `run_evidence/*/lab.json` 的 `token` 字段一律改为 `<redacted>`（保留字段而非删除，schema 不变）。
- 另有 13 个 run 的 `traj.jsonl` / `state.json` / `cc_stream_*.jsonl` 里，agent 自己 `cat` 出了 lab token，已就地替换为 `<redacted-lab-token>`。
- 全树扫过 `sk-[A-Za-z0-9_-]{16,}`：剩下的命中只有两类，一是 `harness/` 里两个字面占位符（`sk-sandbox-placeholder-not-a-credential`、`sk-FAKE-REAL-KEY-...`），二是 `state.json` 里 base64 `encrypted_content` 块内部偶然出现的子串（所处的连续 token 长达数千字符）。网关真实 key 不在本目录任何文件中，已按串比对确认。

## 复现

跑分用 `python3 harness/launch.py <task_dir> <cc|gpt> <model> <hint 0|1|2> <run_name> [effort]`，需要自备网关凭据（不随交付分发）。
注意：`tasks/` 下已构建的实例目录是写保护的，重新构建请换新名字。
