# v10（2026-09-29）：AI4AI benchmark 第十版

本版包含三个异质题族，每个实例都有正确性证书，并附一条从机制卡到轨迹硬化的造题 pipeline。

| 文件 | 内容 |
|---|---|
| **`V10_REPORT.md`** | **先读这个**：结论、前沿成绩、每个失败的边界核查、通过路线分析、教训、L3 下一步 |
| `DESIGN.md` | pipeline 设计：S0 机制卡 → S1 算子 → S2/S3 证书 → S4–S6 实跑与归类 → S7 硬化；异质性矩阵 |
| `EXCLUDED.md` | 哪些运行不计分、为什么（规则事先定）；计分但要标注的条件 |

## 一句话结果

- **正确性**：三族都有证书，所有计分项都是确定性判分。
- **题族**：
  - serve_ab：上线实验设计，交付日程文件 + 内嵌区间 QA；
  - judge_audit：评测方法学，纯 QA；
  - scaleup：MLS 式 repo，执行判分。
- **难度**：没有做到稳定困住前沿。
  - 全部有效运行：Fable 5.1 17/22，GPT-6 13/19；
  - 最难档 scaleup L2：Fable 4/8，GPT-6 3/5；
  - 失败都能精确归因到设计的机制，通过大多走事先已知的路线。
  - 详见报告 §0。

## 目录

```
l15/            lab 服务、判分器（l15/grade.py）、v8 闸门与可续跑证书（v8gates.py / v8cert.py）
l15/tasks/      三族的世界、证书与诱饵（serve_ab*.py、judge_audit*.py、scaleup*.py、scaleup_l2_decoys/）
tasks/          19 个已打包的任务实例（instruction.md / environment/app / hidden / task.toml）
certs/          证书：serve_ab/*.report.json、judge_audit_cert.json、scaleup_cert.json、scaleup_l2_cert.json
certs/boundary/ 边界核查脚本（dup_holdout_cf / line_cf / cfg_cf）与结果 results/*.jsonl
harness/        launch / drive / cc_agent（官方 Claude Code CLI）/ gpt_agent / gateway / test_no_key /
                export_evidence / results_table
run_evidence/   53 个运行目录（41 个有效 + 8 个通道故障 + 2 个分类器终止 + 2 个冒烟），已脱敏
```

## 复现

在本目录下运行：

```bash
python3 harness/results_table.py run_evidence            # 成绩总表
python3 -m l15.grade <run_dir>                           # 重新判分（读 tasks/<task>/hidden）
python3 -m l15.tasks.judge_audit_cert 20                 # judge_audit 证书
python3 -m l15.tasks.scaleup_l2_cert 150 2 all 0 5 6 7   # scaleup L2 证书（机器空闲时，可续跑）
python3 -m l15.v8cert serve_ab_l2 0 16 6 certs/serve_ab/sab_L2b_state.json 150   # serve_ab L2 证书（可续跑）
python3 certs/boundary/cfg_cf.py run_evidence/su_L2_s7_gpt6 7 test '{"weight_decay": 0.15}' 0   # 一个反事实
python3 harness/test_no_key.py                           # 密钥隔离测试
```

scaleup 的判分和证书要**单独**运行，不要和其它作业并行。同机争用会抬高 CPU 读数，而这个数会写进题面。

## 安全

- 上游 API key 只由本地网关进程持有，从不写入任何文件；沙箱里只有占位符。`harness/test_no_key.py` 断言这一点。
- `run_evidence/` 由 `harness/export_evidence.py` 生成：
  - 去掉了 `home/`、`sbx_tmp/`、`.lab_token`、`.npy/.npz`；
  - lab token 与任何 `sk-` 字符串都做了脱敏；
  - 导出后会重新扫描，残留为 0。
