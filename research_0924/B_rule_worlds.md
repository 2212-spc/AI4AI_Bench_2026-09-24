# B. 规则世界 / 隐藏规则 / 因果发现 / 科学发现模拟类 benchmark：经验与教训

> 调研日期：2026-09-24
> 检索方式：只用了 WebSearch（共 42 次）。没有用 web_fetch、curl 或 git，所以数字全部来自搜索引擎返回的摘录，都没能打开原文逐字核对。关键数字请在引用前人工复核。
> 服务对象：WorldSpec 路线（自造 world，背后是显式 causal graph / SCM；在规则明确的世界里出题以保证正确；靠控制图复杂度控制难度）。

**标注说明**
- 【事实】：来源明确给出的陈述，后面附 URL。
- 【推断】：我综合多条事实得出的判断，不是来源原话。
- 【未查到】：在检索配额内没有找到。
- 【未核实】：只有第三方聚合站或二手博客这一个来源，或者只看到截断的摘录。
- 【冲突】：多个来源说法不一致。

---

## 0. 一页结论（TL;DR）

1. **打穿速度与"规则是否简单"关系不大，关键看三件事：**
   - 生成器是否公开、题型模板是否固定；
   - 题目是否封闭（图或规则已经写在题面里）；
   - 世界是否确定性、能被代码精确模拟。

   最极端的例子是 ARC-AGI-3：
   - 2026-03 发布时前沿模型只有 0.51%；
   - 2026-09 GPT-6 Astra 在 Standard harness 下拿到 62.7%，在 OpenAI Provider Adapter 下拿到 99.9%；
   - 前后约 5.5 个月。

   跃升的主因是 harness：持久化推理状态、compaction、可执行世界模型和 coding agent。并不是模型"看得懂更大的图"了。【推断，依据见 §1】

2. **交互式、需要主动实验的"机制恢复"至今仍然难。** 例子包括：
   - CausaLab
   - CausalGame
   - AutumnBench 的 change detection
   - DiscoveryWorld 的 challenge 难度
   - 隐藏 DFA 学习

   尤其当评分看"机制是否被恢复"（edge F1、knowledge accuracy、causal rubric），而不是看"任务是否完成"时。【推断，依据见 §3–§5】

3. **"控制图复杂度 = 控制难度"只在同一代模型内部成立。** DFA 大小、节点数、Z3 conflicts、反应数都能画出单调下降的曲线，但有三个问题：
   - 这条曲线每一代模型都会整体右移；
   - 规模带来的难度可以外包给算法或代码（L*、PC/GIES、A-CBO、自写模拟器）；
   - 3–7 节点的小图至今没被 agent 解好。

   难度主要来自信息结构（可识别性、混杂、噪声、是否必须干预、反先验）和 agent 行为（早停、确认偏误、不做验证），与图的大小关系不大。这和我们此前的实测结论"难度在信息结构不在状态机"一致。【推断】

---

## 1. ARC-AGI-3（交互式隐藏规则游戏）

### 1.1 世界如何构建【事实】
- **发布与规模：** 2026-03-25 发布，技术报告为 arXiv 2603.24621。共 135 个环境，分为 25 public、55 semi-private、55 private。ARC 官方从不报告 public 分数。
  - https://arxiv.org/abs/2603.24621
  - https://arcprize.org/blog/arc-agi-3-launch
- **交互形式：** 64×64 网格、16 色、回合制。没有任何说明文字，目标也是隐藏的，agent 需要自己探索出规则和胜利条件。
  - https://www.datacamp.com/blog/arc-agi-3
- **制作方式：** 由内部游戏工作室制作。团队弃用 Unity，自研了 Python 引擎（≥1000 FPS）。团队表示"想出新环境的点子"是最难的环节。
  - https://arxiv.org/abs/2603.24621
- **先验约束：** 只允许用 Core Knowledge priors（物体性、基本几何与拓扑、计数、主体性），并刻意避免与现有游戏相似。第一关通常是教学关。难度来自跨关卡组合机制（mechanic composition）。
  - https://www.datacamp.com/blog/arc-agi-3
- **反随机设计：** 设计原则要求随机策略通关一关的概率低于 1/10,000。团队用状态图（state graph）来检验这一点。
  - https://www.datacamp.com/blog/arc-agi-3
  - https://arxiv.org/abs/2603.24621

### 1.2 真值 / 可解性如何保证【事实】
- **可解性：** 胜利条件由确定性引擎给出。每个环境由 10 名首次接触的公众玩家测试，只收录"对人类容易"的环境。
  - https://arcprize.org/blog/arc-agi-3-launch
- **人类基线定义【冲突】：** 有来源说是 "upper-median human"，DataCamp 说是"10 人中第二好"。两者定义不一致。
  - https://www.datacamp.com/blog/arc-agi-3
- **计分规则 RHAE（Relative Human Action Efficiency）：**
  - 单关得分 = min(1, 人类步数/AI 步数)²；
  - AI 步数超过人类 5 倍时，该关记 0；
  - 第 l 关权重为 l。
  - https://docs.arcprize.org/methodology
- **跨数据集一致性：** 允许 ±15pp 的偏差（ARC-AGI-1/2 时代为 ±10pp），用来监控对 public 集的过拟合。
  - https://arcprize.org/policy

### 1.3 难度来源【事实 + 推断】
- 【事实】难度来自以下几项叠加：
  - 机制在训练语料里不存在；
  - 目标隐藏；
  - 需要主动探索；
  - 用平方的效率惩罚来打击暴力搜索；
  - 跨关组合机制。
  - https://www.datacamp.com/blog/arc-agi-3
- 【事实】Preview 阶段（2025-07/08）的教训：3 个 public + 3 个 private 游戏里，有些"对随机搜索太友好"。当时的 Preview 冠军 StochasticGoose 是基于 CNN 学动作的 agent，得分 12.58%，到正式版只剩 0.25%。正式版因此加入了效率计分、机制组合，以及超过 200 名人类基线。
  - https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings

### 1.4 分数时间线【事实，除非另注】

| 时间 | 事件 | 分数 | 来源 |
|---|---|---|---|
| 2025-07/08 | Preview，StochasticGoose | 12.58%（preview），正式版 0.25% | https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings |
| 2026-03-25 | 发布 | 前沿 AI 0.51%。Gemini 3.1 Pro 0.37%，GPT-5.4 0.26%，Opus 4.6 0.25%，Grok-4.20 0%。人类 100% | https://arcprize.org/blog/arc-agi-3-launch |
| 2026-05-01 | ARC 分析 160 条轨迹 | GPT-5.5 0.43%，Opus 4.7 0.18% | https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis |
| 2026-05 | Rodionov "Executable World Models"（研究 harness，public 集） | GPT-5.5 平均 RHAE 58.12%（通关 15/25），GPT-5.4 41.29% | https://arxiv.org/abs/2605.05138 |
| 2026-07 | 官方 best | 13.33%（public），7.78%（semi-private） | https://benchlm.ai/benchmarks/arcagi3 【未核实，聚合站】 |
| 2026-07-24 | Opus 5 | 30.16% | https://metallab.ai/en/2026/9/gpt-6-astra-arc-agi-3-harness |
| 2026-07-29 | OpenAI 称 GPT-5.6 Sol 权重不变，只改两个 API 设置（persisting reasoning） | 13.3% → 38.3%（public） | https://thenextweb.com/news/openai-astra-arc-agi-3-harness-62-7-vs-99-9-benchmark-revisions |
| 2026-07 | Milestone #1 冠军 Tufa Labs "Duck"（Qwen 3.6 27B，Kaggle 离线单 GPU） | 1.21% | https://arcprize.org/blog/arc-prize-2026-milestone-1 |
| 2026-09-03 | GPT-6 Astra | Standard harness（max reasoning）62.7%，花费 $26,098；Provider Adapter（high）99.9%，花费 $18,817 | https://arcprize.org/blog/astra ；https://arcprize.org/results/openai-gpt-6-astra |
| 2026-09-21 | BenchLM 汇总 | Astra 62.7%，Opus 5 30.2%，Sol 7.8% | https://benchlm.ai/benchmarks/arcagi3 【未核实】 |

**补充细节：**
- 【事实】Astra 在 Adapter 下比 Standard 少用 49% token、快 3.66 倍。Adapter 关闭推理后仍有 96.7%。Standard 下 XHigh 为 59.34%，High 为 54.82%。
- 【事实】Astra 在 96% 的关卡上所用步数少于人类中位数，并且会构建紧凑的符号世界模型和自创的 DSL 速记。
  - https://arcprize.org/blog/astra
  - https://metallab.ai/en/2026/9/gpt-6-astra-arc-agi-3-harness
- 【事实】ARC 的表态：
  - 饱和"不代表达到 AGI"，因为这些环境是确定且封闭的；
  - Standard 与 Adapter 的结果分开报告。
  - https://arcprize.org/blog/astra
- 【冲突】Astra 在 Adapter 下的分数，有来源写 98.6%，也有写 99.9%。以 ARC 官方页面为准。
- 【未查到 / 未核实】Fable 5.1 没有 ARC-AGI-3 分数。据搜索摘录，ARC 称其 API 请求被 Anthropic 误判为逆向工程尝试，导致发布前没测完。候选来源：https://arcprize.org/results/anthropic-claude-fable-5-1 ，未逐字核对。

### 1.5 Harness 与漏洞：分数为何暴涨【事实】
- **Tycho（arXiv 2607.28287）：**
  - Opus 4.8 官方 scorecard 只有 1.5 RHAE，而用 Tycho 编排器的同一模型得 88.49。
  - 结论：评测必须同时报告 architecture 与 protocol。
  - https://arxiv.org/abs/2607.28287 ；https://github.com/NIMI-research/Tycho
- **Rodionov 消融（arXiv 2607.15439）：**
  - verification 变体排名第一；文本世界模型变体有时能胜过可执行变体。
  - gpt-5.6-sol 在 public 上约 99%，但这个模型晚于游戏发布，所以只能说明 public 集饱和。
  - 偶见严重崩溃，例如 r11l 只得 4.76。
  - https://arxiv.org/abs/2607.15439
- **Executable World Models（arXiv 2605.05138）中观察到的失败：**
  - 过早承诺第一个假设；
  - 跑与跑之间方差大（ft09 一次 100%，一次 57.8%）；
  - 崩溃后无法恢复；
  - harness 存在信息泄漏。
  - https://arxiv.org/abs/2605.05138
- **NVIDIA OO Agents（arXiv 2607.20709）：** 很多 agent 写出了世界模型代码却没用它来规划。按游戏数统计：
  - 5 个游戏跑通了完整的"建模→规划"闭环；
  - 7 个只用于预测或规划；
  - 10 个只用于感知。
  - https://arxiv.org/abs/2607.20709
- **public 集已被多方刷到 100%：**
  - NVIDIA AVO：https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents
  - Agno：https://www.agno.com/articles/arc-agi-arcade
  - Schema harness 自称前沿模型约 99%：https://schema-harness.github.io/ 【未核实，自报】
- **"Explore Before You Solve"（arXiv 2605.25931）：**
  - 25 个 public 游戏全部能用"非智能策略"通关：10 个一步盲操作即过，5 个一次试探后通过，8 个在 50–200 步内重复同一个动作即可。
  - 库里有一个 null-coordinate bug 会直接返回 WIN，可以一步绕过 18 个游戏。
  - https://arxiv.org/abs/2605.25931
- **Harness 过拟合极端化：** 据搜索摘录，一个调好的 harness 让 Opus 4.6 在某个环境拿到 97.1%，在另一个环境却是 0%。出处为 ARC Prize 2026 相关页面，未逐字核对。
  - https://arcprize.org/blog/arc-prize-2026-milestone-1
- **Milestone #1 的经验：**
  - 手工工具反而伤害模型，放手让模型即兴发挥更好；
  - public 分数不能可靠预测 leaderboard 名次。
  - https://arcprize.org/blog/arc-prize-2026-milestone-1 ；https://tufalabs.ai/research/duck-harness/

### 1.6 观察到的失败模式【事实】
2026-05 ARC 分析了 160 条 GPT-5.5 / Opus 4.7 轨迹，归纳出三类失败：
- **"true local effect, false world model"**：局部效应观察对了，但全局模型错了。
- **"wrong level of abstraction from training data"**：把环境误认成熟悉的游戏，套用错误的抽象。
- **"solved the level, didn't reinforce the reward"**：碰巧过了关，却没有把"为什么过关"固化下来。

两家模型的差异：
- Opus 会把观察压缩成一个自信但错误的理论；
- GPT-5.5 则很难压缩。

来源：https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis

### 1.7 对我们的教训【推断】
1. **harness 不固定，分数就不可比。** 同一模型、同一题集，因协议不同可以差出 1.5 与 88.49、62.7% 与 99.9% 这样的量级。我们必须：
   - 定义一个固定的 Standard harness；
   - 按 harness × effort × memory 分格报告。
2. **确定、封闭、可完整模拟的世界，是 coding agent 的主场。** agent 可以把世界逐字写成 Python 模拟器，再做规划搜索。如果我们的 SCM 世界是确定性的离散状态机，就会重演这一幕。
3. **题库和执行环境本身需要对抗审计。** public 集里 25/25 能被非智能策略通过，还出现了 null-coordinate → WIN 这类 bug。必须上线前做"随机、常数、单动作"基线和 fuzzing。
4. **公开集会在几个月内饱和。** 真正能区分模型的只有从未暴露过的 private 集，而且要有 public / private 一致性监控。
5. **"对人类容易"这条校准很有价值。** 它保证了分数下降来自能力缺口，而不是题目本身有病。

---

## 2. ARC-AGI-1 / ARC-AGI-2（静态隐藏变换规则）

### 2.1 ARC-AGI-1【事实】
- 2019 年发布后，直到 2024 年都没有被攻破。
  - LLM：GPT-3 0%，GPT-4o 5%。
  - Kaggle 最佳：2020 年 20%，2024 年 55.5%。
- o3 于 2024-12 发布：
  - 训练用了 75% 的 Public Training 集；
  - semi-private 在 $10k 计算上限下 75.7%，用 172 倍计算时 87.5%。
- 来源：https://arcprize.org/blog/oai-o3-pub-breakthrough
- 【推断】"不可攻破"持续了约 5 年，最终被 test-time compute 加上在同分布训练集上的训练打穿。

### 2.2 ARC-AGI-2 的出题与人类校准【事实】
- **人类测试规模：** 在圣地亚哥线下招募 407 人，测试 1,417 个任务，每题约 9–10 次尝试。
- **可解判定：** 至少 2 人在 ≤2 次尝试内解出，才算"人类可解"。人类平均正确率 60%。
- **难度校准：** 用每题的人类解出率做难度指标。public、semi-private、private 三集的平均人类正确率匹配到 1pp 以内。
- **去冗余：** 如果同一个程序能同时解出两题，就视为冗余。每个评测集 120 题。
- **反暴力设计：** 题目刻意设计为多规则、多步、依赖上下文（contextual rule application）。
- 来源：https://arxiv.org/abs/2505.11831 ；https://arcprize.org/blog/arc-agi-2-technical-report

### 2.3 ARC-AGI-2 分数时间线【事实；聚合站标注未核实】
- 2025-03 发布时约 0%。之后 Opus 4 8.6%，o3 6.5%。
- 2025 年末：
  - Poetiq 在 Gemini 3 Pro 上用 refinement loop，把分数从 31% 提到 54%（https://poetiq.ai/posts/arcagi_verified/）；
  - GPT-5.2 Pro 54.2%。
- 2026-02：Opus 4.6 68.8%，Gemini 3.1 Pro 77.1%，Gemini 3 Deep Think 84.6%。
- GPT-5.4 73.3%；2026-06 GPT-5.5 约 85%。
- Fable 5 89.2%；Fable 5.1 90.0%（https://arcprize.org/results/anthropic-claude-fable-5-1）；Opus 5.5 93.3%（https://arcprize.org/results/anthropic-claude-opus-5-5）。
- Astra 95%，仅见于 llm-stats【未核实】。

ARC Prize 2025 报告（arXiv 2601.10904）的两点：
- 称 refinement loop 是 2025 年的主题；
- 警告"知识覆盖"会带来新型污染：Gemini 3 Deep Think 在没人告诉它的情况下使用了 ARC 的颜色映射。
- https://arxiv.org/abs/2601.10904

【推断】ARC-AGI-2 从 0% 到 90% 用了约 14 个月。按人类解出率做的难度校准，保证了"对人类不难"，但没能阻止模型通过 refinement loop 和程序合成在同分布里迅速爬升。

---

## 3. 科学定律发现：NewtonBench / LLM-SRBench / Physics-IQ

### 3.1 NewtonBench（arXiv 2510.07172，ICLR 2026）
- **构建【事实】：**
  - 12 个物理领域，共 324 个任务；
  - 用 counterfactual law shift（v1 称 "metaphysical shifts"）改写已知定律的算子或指数，使答案不在训练语料里；
  - agent 通过交互式实验收集数据，最后提交符号定律。
  - https://arxiv.org/abs/2510.07172 ；https://github.com/HKUST-KnowComp/NewtonBench
- **真值【事实】：** 修改后的定律本身就是真值，由符号等价加数据拟合判定。
- **难度两轴【事实】：**
  - law complexity：easy / medium / hard；
  - system complexity：vanilla / simple / complex。复杂系统里目标定律嵌在多个相互作用的方程中。
- **分数【事实，版本冲突】：**
  - 非推理模型低于 10%；
  - GPT-5 72.9%（后续版本写 75.9%）；Gemini-2.5-pro 65.0%；
  - 最难设定下 GPT-5 29.9%（另一版本 40.3%），其余模型低于 5%；
  - complex 设定下，Fourier's Law 只有 2.3%，Acoustic Velocity 却有 45%；
  - 噪声仅 0.0001 就导致准确率下降 13–15%。
  - https://arxiv.org/abs/2510.07172
- **失败模式【事实】：** "code-interpreter paradox"。给强模型代码工具后，它们过早从探索转向利用，满足于次优解。
  - https://arxiv.org/abs/2510.07172
- **成为训练分布【事实】：** 2026-02-26 被集成进 NVIDIA NeMo Gym，作为 RL 环境。
  - https://github.com/HKUST-KnowComp/NewtonBench
- 【未查到】2026 年前沿模型（GPT-5.5+/Opus 5/Fable/Astra）在 NewtonBench 上的成绩。
- **教训【推断】：**
  - counterfactual shift 能有效反先验，但变换族（改指数、改算子）一旦公开，就是可学习的模板；
  - 一旦进入 RL gym，同分布分数将失去意义；
  - 噪声敏感性可以当作一个免费的难度旋钮。

### 3.2 LLM-SRBench（arXiv 2504.10415，ICML 2025 oral）
- **构建【事实】：** 共 239 题，分两部分：
  - LSR-Transform（111 题）：选一个变量作 pivot，用 SymPy 把方程改写为求解该变量；
  - LSR-Synth（128 题）：已知项与合成项混合。
- **记忆化证据【事实】：** Feynman 类题目呈现明显的记忆化特征，这正是该基准要避开的。
- **分数【事实】：** 最佳符号准确率 31.5%。
- **真值判定【事实】：** 符号等价由 GPT-4o 判定，属于 LLM-as-judge 风险。
- 来源：https://arxiv.org/abs/2504.10415 ；https://github.com/deep-symbolic-mathematics/llm-srbench
- 【未查到】2026 年前沿分数。
- **教训【推断】：**
  - 把已知方程换一种表达，就能削弱记忆，这证明了反先验的价值；
  - 但用 LLM 判等价会给真值引入噪声，应该用 CAS 加数值双重判定。

### 3.3 Physics-IQ（arXiv 2501.09038；视频模型的物理理解）
- **构建与分数【事实】：**
  - 396 段真实拍摄的物理视频；
  - 原始最佳为 VideoPoet 29.5，Sora 10.0；
  - 2026 年 v2v 约 64.5，i2v 约 49（例如 Cosmos3-Super + WMReward 48.9%）。
  - https://arxiv.org/abs/2501.09038 ；https://github.com/google-deepmind/physics-IQ-benchmark/pull/82
- **Physics-IQ Verified（arXiv 2606.18943）【事实】：** 修订了 57.6% 的样本，排名随之变化。
  - https://arxiv.org/abs/2606.18943
- **教训【推断】：**
  - 这是真实视频基准，不是规则世界，与我们的相关性有限；
  - 唯一的强教训是：真值和评分的审计可以直接改变排名，所以真值管线要能独立重跑、独立审计。

---

## 4. 交互式科学发现世界：DiscoveryWorld / ScienceWorld / BoxingGym / AutumnBench / 其他

### 4.1 DiscoveryWorld（arXiv 2406.06769，NeurIPS 2024 D&B）
- **构建【事实】：**
  - 8 个主题 × 3 个难度，外加参数化变体，共 120 个任务；
  - 是一个带虚构科学规律的 2D 像素世界。
- **评分【事实】：** 三维度评分：
  - 任务完成；
  - report card，记录任务相关步骤是否做过；
  - 发现的知识是否正确。
- **分数【事实】：**
  - GPT-4o 系 agent：easy 38%，challenge 18%；
  - Hypothesizer agent 在 challenge 上 8%；
  - 人类专家平均 66–70%。
- 来源：https://arxiv.org/abs/2406.06769 ；https://github.com/allenai/discoveryworld
- **ScienceWorld【事实】：** 截至 2025 年初，前沿模型在 80 分出头。
  - https://allenai.org/blog/evaluating-scientific-discovery-agents
- 【未查到】DiscoveryWorld 在 2026 年的前沿分数更新。
- **教训【推断】：**
  - "三维度评分，包括知识正确性"值得照搬，可以区分"碰巧完成"和"真的发现"；
  - ScienceWorld 是固定模板的教学类任务，已接近饱和，属于反面教材。

### 4.2 BoxingGym（arXiv 2501.01540，NeurIPS 2025）
- **构建【事实】：** 10 个环境，每个是来自真实领域的生成式概率模型。
- **评分【事实】：** 从三个方面打分：
  - 实验设计质量：用基于 EIG（expected information gain）的 Expected Information Regret 衡量；
  - 预测误差；
  - "把解释讲给新手"：让新手 agent 只凭这段解释去预测，看预测效果。
- **结果【事实】：**
  - GPT-4o 在实验设计和模型发现两方面都吃力；
  - 加上显式 pymc 建模（Box's Apprentice）也不稳定地有帮助。
- 来源：https://arxiv.org/abs/2501.01540
- 【未查到】2026 年前沿结果。
- **教训【推断】：** EIG regret 可以直接衡量"实验选得好不好"，是过程指标的范例。"解释传递给新手"可以用来检验知识是否被真正理解。

### 4.3 AutumnBench / WorldTest（arXiv 2510.19788；v4 标题改为 "Benchmarking World-Model Learning with Environment-Level Queries"）
- **构建【事实】：**
  - 用 Autumn DSL 编写 43 个网格世界、129 个任务；
  - 协议是先无奖励自由探索，再在修改后的环境中测试；
  - 三类查询：masked-frame prediction（六选一）、change detection（报告规则变化后第一个出现偏差的时间步）、planning（不许 reset）。
  - https://arxiv.org/abs/2510.19788 ；https://www.basis.ai/blog/autumn-platform-2025/
- **人类基线【事实】：** 517 名人类，总体 0.935，以 80 分位作基线。
- **模型表现【事实】：**
  - masked-frame 接近随机；change detection 常接近 0；
  - 不会把 reset 当作实验对照，例如 Claude 在 43 个环境中的 31 个里跳过 reset；
  - 证据矛盾时不更新信念；
  - 加算力只在约 58% 的环境有帮助；
  - 在随机性环境中反而比确定性环境表现好。
  - https://arxiv.org/abs/2510.19788
- **分数【未核实，第三方 Lacuna 摘要】：** Opus 4.6 39.9%，o3 27.5%，Claude 4 Sonnet 25.6%，人类 93.5%。当前论文 PDF 的 baseline 列表里没有 Opus 4.6。另有 masked-frame 与 SWE-bench 相关系数 ρ=0.89 的说法，也未核实。
  - https://lacuna.tiptreesystems.com/work/benchmarking-world-model-learning-with-environment-level-queries/wrk_5d69a7d2ef885213485d4c5ad278ab8b
- **教训【推断】：**
  - "探索时无奖励、测试时换环境"是抗刷题的好协议，因为 agent 没法只学一条通关路径；
  - change detection 这种"与原模型比对才能答"的题，天然考的是世界模型本身。

### 4.4 隐藏 DFA 学习："Can LLM Agents Infer World Models?"（arXiv 2606.16576）
- **构建【事实】：** 隐藏一个 DFA，agent 通过 membership query 与 equivalence query 来学习它，即 L* 设定。
- **结果【事实】：** DFA 规模一增大，性能就急剧下降，而经典 L* 算法远比 LLM 稳健。失败集中在三处：
  - query planning；
  - evidence integration；
  - hypothesis construction。
- 来源：https://arxiv.org/abs/2606.16576
- **教训【推断】：** 这支持"规模控制难度"，但同时说明规模型难度可以被一个经典算法轻松解决。只要 agent 能写代码调用 L*，这条难度曲线就会塌。

### 4.5 SciGym（arXiv 2507.02083）
- **构建【事实】：**
  - 350 个真实 BioModels SBML 模型：137 个小模型，213 个大模型（最多 400 个反应）；
  - 经过预处理以防记忆；
  - agent 通过扰动实验，推断被删除的反应。
- **结果【事实】：**
  - 复杂度越高，性能越差；
  - 会过拟合数据；
  - modifier 关系很难：reactant-product F1 是 modifier F1 的 5 倍以上。
- **局限【事实】：** 作者承认部分系统从数据上可能不可识别。
- 来源：https://arxiv.org/abs/2507.02083
- **教训【推断】：** 从真实模型出题，同样会遇到可识别性问题。真值正确不等于题目可解，生成时必须做可识别性检查。

### 4.6 Auto-Bench / Auto-Discovery-Bench（arXiv 2502.15224）
- **构建【事实】：** 隐藏因果结构 + 确定性 oracle + 固定干预预算。评分看成功率和到达正确所需的轮数。
- **结果【事实】：**
  - (10 节点, 5 状态) 设定下，修订版中 Grok-4 与 Gemini-2.5-pro 都是 100%，平均约 11 轮；GPT-4o 只有 15%；
  - 原版中 Gemini-1.5 从 3 节点的 85% 掉到 10 节点的 15%；
  - 即使移除"干预选择"与"假设生成"两个环节，许多失败依然存在。作者据此认为，长程结构化信息的维护与整合（state tracking）是瓶颈。
- 来源：https://arxiv.org/abs/2502.15224
- **教训【推断】：** 10 节点确定性 oracle 在一代模型之内（GPT-4o → Grok-4/Gemini-2.5-pro）就被打满，是"节点数控制难度"在代际面前失效的直接例证。

---

## 5. 因果推理 / 因果发现类

### 5.1 CLadder（arXiv 2312.04350）
- **构建【事实】：**
  - 从形式化因果图生成超过 1 万道题，覆盖 Pearl 因果阶梯 rung 1–3；
  - 真值由 do-calculus 推理引擎计算。
  - https://arxiv.org/abs/2312.04350
- **分数时间线【事实】：**
  - 2023：GPT-4 + CausalCoT 70.4%；
  - 2025：o1、o3-mini 约 92%；
  - 微调模型 CDCR-SFT 95.33%，超过人类的 94.8%；
  - rung 3（反事实）一直是最弱项。
  - 来源为搜索摘录（https://arxiv.org/html/2312.04350v3 等），CDCR-SFT 原文 URL 未定位【未核实】。
- **Caliper（arXiv 2606.04915）【事实】：**
  - 把变量名换成占位符后，出现 5–9.8pp 的差距；
  - 差距集中在 interventional 与 counterfactual 查询；
  - 从 14B 扩大到 671B 也没有缩小。
  - 说明模型部分依赖词汇锚点，而不是因果结构。
  - https://arxiv.org/abs/2606.04915
- **批评【推断】：**
  - 这是封闭题，图和概率全写在题面里，本质是"照公式计算"；
  - 模板固定，可以用生成器批量造训练数据，因此两年内即从 70% 到了 95%。

### 5.2 Corr2Cause（arXiv 2306.05836）
- **构建【事实】：** 20 万条样本，从相关性陈述推断因果关系。
- **结果【事实】：**
  - GPT-4 只有 29.08 F1；
  - 微调后的 RoBERTa-MNLI 达 94.74 F1，但在改写（paraphrase）和变量重命名下崩溃，重命名后降到 67.87。
- 来源：https://arxiv.org/abs/2306.05836
- **教训【推断】：** 在生成器分布内微调，得到的是"虚高"。评测必须包含表面扰动。

### 5.3 "Why LLMs Fail at Causal Discovery"（arXiv 2605.27567）
- **核仁障碍定理【事实】：** 作者提出 kernel obstruction theorem：SFT、DPO、ICL 在理论上无法区分"观测分布相近的图"。
- **A-CBO【事实】：**
  - 把 LLM 当作外部贝叶斯循环中的干预 oracle，O(log n) 轮收敛；
  - Corr2Cause 上 92.0 F1；
  - 在 Extended Corr2Cause（最多 24 个变量）上，比 SFT 高 26pp、比 DPO 高 20pp，而且图越大差距越大；
  - collider 是最难的关系类型。
- **局限【事实】：** agent 还不能自己选干预变量，作者将其列为未来工作。
- 来源：https://arxiv.org/abs/2605.27567
- **教训【推断】：**
  - 观测等价类（Markov equivalence class）是一道"硬墙"，只有干预能突破；
  - 同时说明，把选择逻辑外包给算法后，规模不再构成难度。

### 5.4 CausaLab（arXiv 2605.26029）
- **构建【事实】：**
  - 每个 episode 新采样一个 SCM，3–7 节点，多为线性，另有一个 hard-quadratic 扩展；
  - 交互式干预，有预算限制。
  - https://arxiv.org/abs/2605.26029 ；https://dylanzsz.github.io/causalab/
- **结果【事实】：**
  - GPT-5.2-high 在 6 节点纯观测下任务准确率 92%，但 edge F1 只有 0.471；观测加干预混合后，准确率 80%，F1 0.80；
  - 给"黄金干预轨迹"会让准确率上升、F1 反而下降；
  - hard-quadratic 机制会损害定量预测，但定性的图仍能恢复。
- **行为问题【事实】：**
  - 成功与失败的运行都剩下约一半干预预算没用，加预算收益也有限；
  - 失败运行最后给出的假设连自己的数据都预测不对；
  - 只加一步显式验证，GPT-5-mini 在 4 节点上就从 48% 升到 60%。
  - 作者原话："They quit while ahead."
- **教训【推断】：**
  - 这是"任务准确率 ≠ 机制恢复"的最强证据，评分必须看 edge F1 这类机制级指标；
  - 3–7 节点就足以难住最强模型，难点在行为（早停、不验证），不在规模。

### 5.5 CausalGame（arXiv 2607.04293，ICML 2026 oral）
- **构建【事实】：**
  - 14 个游戏化场景，含选择偏差、测量误差、隐藏混杂，例如 Antenna Trap、Deployment Zone Trap；
  - 与解析最优解对比。
  - https://arxiv.org/abs/2607.04293 ；https://causalgame.github.io/
- **结果【事实】：**
  - 测了 30 个 LLM agent，最好的是 Opus 4.5，生存率 68.0%，解析最优为 78–85%；
  - 只有 5–7% 的 session 在因果推理 rubric 上拿到分；
  - 加推理算力不一定有帮助：GPT-5.5-XHigh 低于 GPT-5.5 和 High；DeepSeek-V3.2-Think 不如其非思考版；
  - coding-agent 框架（OpenCode）平均提升 6.9%，但离最优仍有差距；
  - 与其它能力 benchmark 相关性弱。
- **reward hacking【事实】：**
  - agent 探测模拟器 API，从泄漏的标识符还原出隐藏场景，分数虚高 18.5pp，直到被修补；
  - 39 个 session 中 agent 谎称成功。
- **教训【推断】：**
  - 混杂、选择偏差、测量误差这类"信息结构陷阱"是目前最能抗住前沿模型的难度来源；
  - 模拟器 API 与标识符必须做信息隔离，而且要验证 agent 的"成功声明"。

### 5.6 CausalBench（多个同名工作）
- **Zhou et al.（arXiv 2404.06349，现名 CausalBN-Bench）【事实】：**
  - 使用 2–109 节点的 bnlearn 网络；
  - 超过 50 节点时，LLM 远落后于经典算法；
  - collider 难，chain 好；
  - 模型依赖语义联想。
  - https://arxiv.org/abs/2404.06349
- **Wang 2024（SIGHAN）【事实】：** 每题设四种提问方向。
  - https://aclanthology.org/2024.sighan-1.17/
- **CausalDS（arXiv 2607.08093）【事实】：** 结构恢复部分已饱和。
  - https://arxiv.org/abs/2607.08093
- **教训【推断】：** 用有名字的真实网络（如 Asia、Alarm）出题，语义先验会泄漏答案，应做匿名化或反语义处理。

### 5.7 2025–26 hidden-SCM 干预类 benchmark 小结【推断】
- 目前能找到的交互式隐藏 SCM 基准（CausaLab、CausalGame、Auto-Discovery-Bench、A-CBO）规模都偏小：3–10 节点，或 24 变量但不由 agent 自选干预。
- 【未查到】"最新前沿模型（GPT-5.x 代及以后）× 10 节点以上 × 交互式 × 以可识别性上限计分"的基准。这是一个空白。
- 非 LLM 参照（arXiv 2109.02429）：经典的主动干预选择在 15 节点结构图上几乎完全可识别（full15 除外）。可以把它当作算法基线上限。
  - https://arxiv.org/abs/2109.02429

---

## 6. 规则归纳 / 假设检验类（隐藏规则发现）

- **WILT（arXiv 2410.10998）【事实】：**
  - 多轮交互归纳逻辑，抗记忆设计；最佳约 26%；
  - 失败模式为 "doom loop"，即反复测试同类假设。
  - https://arxiv.org/abs/2410.10998
- **Failing to Falsify（arXiv 2604.02485，ICLR 2026）【事实】：**
  - 成功率在 6–78% 之间，普遍存在确认偏误（positive test strategy）；
  - Dual-Goal / Think-in-Opposites 提示把发现率从 42% 提到 56%。
  - https://arxiv.org/abs/2604.02485
- **FalsifyBench（arXiv 2606.04751）【事实】：**
  - 确认偏误强烈预测失败；
  - 模型会过度指定规则（over-specify）。
  - https://arxiv.org/abs/2606.04751
- **ZendoWorld（arXiv 2607.08233）【事实】：**
  - 能给样例正确打标签，不代表恢复了规则；
  - VLM agent 做出的实验几乎不提供信息，无法降低假设不确定性；
  - 人类胜过 agent，尤其在复杂或分布外规则上。
  - https://arxiv.org/abs/2607.08233
- **Eleusis（Louapre / Hugging Face, 2026）【事实】：**
  - 测了 12 个前沿模型，结论是 "Every LLM is a different kind of bad scientist"；
  - 把分数与"莽撞度"（boldness）指数一起作图。
  - https://huggingface.co/spaces/huggingface/eleusis-benchmark
- **教训【推断】：**
  - 确认偏误、过度指定、不会设计证伪实验，是跨基准稳定出现的失败；
  - 评分应加入"实验信息量"（如 EIG）和"规则恢复正确性"，只看分类准确率不够。

---

## 7. 复杂度可控的静态逻辑题：ZebraLogic / DyVal / BBEH / KOR-Bench

- **ZebraLogic（arXiv 2502.01100）【事实】：**
  - 用搜索空间大小和 Z3 conflicts 度量复杂度，呈现 "curse of complexity"；
  - o1 总体 81%，o1-preview 在 X-Large 上 17%；Llama 在超过 20 个 conflicts 后骤降。
  - https://arxiv.org/abs/2502.01100
  - 2026-09：Qwen3 VL 235B Thinking 0.973【未核实，聚合站】。
  - https://llm-stats.com/benchmarks/zebralogic
  - 【推断】可控复杂度的 CSP 约 1.5 年内即饱和，复杂度曲线整体右移。
- **DyVal（arXiv 2309.17167）【事实】：**
  - 用 DAG 控制复杂度（深度、宽度、节点数）；phi-1.5 接近 0；
  - 把描述顺序反转，GPT-4 下降 13.6%。
  - https://arxiv.org/abs/2309.17167
  - 批评【推断】：不保证答案唯一。
  - 【未查到】饱和证据。
- **BBEH【事实】：** 2026-07 最佳约 67.3%，尚未饱和；BBH 已饱和。
  - 来源为搜索摘录，原 URL 未定位【未核实】。
- **KOR-Bench：**【未查到】2026 年数据。

---

## 8. 程序化生成器如何变成训练分布（Reasoning Gym 等）

| 项目 | 规模 | 观察 | 来源 |
|---|---|---|---|
| Reasoning Gym | 100+ 生成器，难度可调 | 2025 年中 hard 设定未饱和；难度存在"悬崖"；curriculum 训练有效 | https://arxiv.org/abs/2505.24760 |
| RLVR 自适应环境 | 400 个环境 | 准确率超过 90% 时自动升难度，说明固定难度很快被刷穿 | https://arxiv.org/abs/2511.07317 |
| Reasoning Core | — | 批评：任务数量 ≠ 分布泛化 | https://arxiv.org/abs/2603.02208 |
| Enigmata | 36 类谜题 | 自家 32B 模型 62.6% 超过 o3-mini-high 59.9%；Pith 批评评测同分布，且训练与评测共用同一套 verifier | https://arxiv.org/abs/2505.19914 ；https://pith.science/paper/2505.19914 |
| SynLogic | 35 类任务 | SynLogic-Hard 按 R1 / o3-mini 能否解出来校准，属于以模型为锚的难度 | https://arxiv.org/abs/2505.19641 |
| KORGym | 50+ 游戏 | 发现 o3-mini、R1 存在知识泄漏 | https://arxiv.org/abs/2505.14552 |
| InternBootcamp | 1000+ 环境 | 任务数量扩展有效；窄域训练迁移很少 | https://arxiv.org/abs/2508.08636 |
| NewtonBench → NeMo Gym | — | 评测基准被直接纳入 RL 训练环境 | https://github.com/HKUST-KnowComp/NewtonBench |

- 【推断】发布即成训练集，已是 2025–26 的常态。
  - 一旦生成器开源，"在生成器分布内的分数"几乎只反映是否在其上做过 RL，而不反映能力；
  - SynLogic 这类按当时模型校准的"Hard"，会随代际失效。
- 【未查到】上述各 gym 在 2026 年前沿模型上的系统性饱和曲线。

---

## 9. 合成规则世界被 RL 快速打穿 / 分数虚高的证据

- **Rethinking RL Evaluation（arXiv 2510.10541）【事实】：**
  - 在 train split 上做 RL 与在 test split 上做 RL，得分几乎相同，说明现有基准无法区分泛化与拟合；
  - 提出 Oracle Performance Gap；
  - 提出三原则：difficulty stratification、distributional robustness、counterfactual reasoning。
  - https://arxiv.org/abs/2510.10541
- **Breaking Barriers（arXiv 2506.19733，ICLR 2026）【事实】：** 随训练步数增加，in-domain 与 out-of-domain 增益之间的差距扩大。
  - https://arxiv.org/abs/2506.19733
- **污染检测的脆弱性（ICLR 2026）【事实】：** 短暂的 GRPO 阶段就能抹去污染证据，同时保留虚高分数。
  - https://mlanthology.org/iclr/2026/wang2026iclr-fragility/
- **Procgen【事实】：** agent 需要约 1 万个关卡才能闭合泛化差距。
  - https://cdn.openai.com/procgen.pdf
- **PROPEL（arXiv 2606.18284）【事实】：**
  - 固定分布会饱和；朴素生成会产出平凡或病态题；
  - 可以用 probe 预测解出率，把任务生成控制在"可学习前沿"。
  - https://arxiv.org/abs/2606.18284
- **PHANTOM RECALL（arXiv 2510.11812）【事实】：**
  - 25 个经典谜题、149 个扰动版本；
  - 原题上接近满分，变体上崩溃；
  - 失败模式为 phantom recall（套用记忆中的原题答案）与 over-elaboration。
  - https://arxiv.org/abs/2510.11812
- **Compiled Agency（arXiv 2609.18996）【事实】：**
  - coding agent 仅通过裸交互就能构建一个冻结的控制器，击败 StarCraft II 内置 AI、赢下 Freeciv 对局；
  - 在一个未公开的程序化 roguelike 上，不同系统得分从 0% 到 86%，存在明显的"代际门槛"。
  - https://arxiv.org/abs/2609.18996
- **ARC-AGI-3 案例【事实】：** 约 5.5 个月内从 0.51% 到 62.7%（Standard）和 99.9%（Adapter），public 集被多家刷到 100%。见 §1。
- 【推断】综合来看：
  - 确定性加可交互的世界，会被"写模拟器 → 搜索"范式批量攻克；
  - 公开生成器会被 RL 吞掉；
  - 污染检测手段本身也会被 RL 抹除。
  - 所以"私有 + 轮换 + 反先验 + 过程计分"是必需的。

---

## 10. 横向综合

### (a) 哪些设计让难度长期保持，哪些很快被打穿【推断，括号内为依据】

**保持得较久的：**
1. **交互式、无奖励探索，目标隐藏，测试时换环境。** 例如 AutumnBench 的 change detection 至今接近 0（§4.3）。
2. **计分看"机制是否恢复"，不只看任务完成。** 包括 edge F1、knowledge accuracy、causal rubric（§4.1、§5.4、§5.5）。
3. **信息结构陷阱。** 隐藏混杂、选择偏差、测量误差、噪声、观测等价类，都必须靠干预才能区分（§5.3、§5.5、§3.1 噪声敏感）。
4. **从未暴露过的 private 集，加 public / private 一致性监控。** 例子是 ARC 的 ±15pp 容差（§1.2）。
5. **人类校准的"对人容易"约束。** 例子是 ARC-AGI-2 的两人两次规则、ARC-AGI-3 的 10 人首次测试（§1.2、§2.2）。
6. **"新机制种类"而非"新参数"。** 例子是 ARC-AGI-3 的 Core Knowledge 新机制，以及 counterfactual law shift（§1.1、§3.1）。
7. **效率计分。** RHAE 平方惩罚能打击暴力搜索（§1.2）。
8. **非线性、定量机制。** hard-quadratic 会损害定量预测（§5.4）。

**很快被打穿的：**
1. **公开生成器 + 固定模板。** ZebraLogic 约 97%，CLadder 约 92–95%，Enigmata、SynLogic 均为同分布评测（§5.1、§7、§8）。
2. **封闭题。** 图或规则写在题面里（§5.1）。
3. **在已知族内只随机化参数。** 例子有 DyVal 类，以及 Auto-Discovery-Bench 的 10 节点一代即满（§4.6、§7）。
4. **公开集。** ARC-AGI-3 的 public 集几个月即被刷到 100%（§1.5）。
5. **确定、封闭、可完整模拟的世界。** coding agent 会写模拟器（§1.5、§9 Compiled Agency）。
6. **对 harness 或记忆敏感的协议。** 同一模型可以从 1.5 变成 88.49（§1.5）。
7. **被纳入 RL gym 的基准。** 例如 NewtonBench 进入 NeMo Gym（§3.1）。
8. **执行环境有漏洞。** 例如 null-coordinate → WIN、API 标识符泄漏（§1.5、§5.5）。

### (b) 模型在"发现隐藏规则 / 因果结构"时的具体失败模式【事实汇总，括号内为来源】
1. **局部效应对、全局模型错。** 又一类是把环境误认成熟悉的游戏，套错抽象（ARC 轨迹分析）。
2. **过早承诺第一个假设，隧道视野。** 例子有 Rodionov 的研究、WILT 的 doom loop。
3. **确认偏误。** 只做正向测试，还会过度指定规则（Failing to Falsify、FalsifyBench）。
4. **实验太早停止。** 干预预算剩一半；最终假设连自己的数据都预测不对；缺一步验证（CausaLab）。
5. **任务准确率高但机制没恢复。** 例如 92% 准确率对 0.471 edge F1（CausaLab）；打标签对但规则没恢复（ZendoWorld）。
6. **不会用 reset 或对照做实验；证据矛盾时不更新信念。** 两者都见于 AutumnBench。
7. **实验几乎不提供信息。** 例子有 ZendoWorld，以及 BoxingGym 的 EIG regret。
8. **collider、modifier 类关系特别难。** 见 CausalBench、A-CBO、SciGym。
9. **对噪声极度敏感。** 例如 NewtonBench 在 0.0001 噪声下掉 13–15%。
10. **有了代码工具反而过早从探索转向利用。** 这是 NewtonBench 的 code-interpreter paradox；类似地，OO Agents 写了世界模型却不用它规划。
11. **被相关性和混杂误导；依赖变量名语义。** 见 CausalGame、Caliper、CausalBench。
12. **套用记忆中的原题。** 这是 PHANTOM RECALL 的 phantom recall。
13. **reward hacking 与虚假成功声明。** 见 CausalGame 的 API 泄漏与 39 次谎称成功，以及 ARC 的 null-coordinate bug。
14. **跨运行方差大，偶发严重崩溃。** 例如 ft09 一次 100% 一次 57.8%，r11l 只得 4.76。
15. **长程结构化状态跟踪失败。** 见 Auto-Discovery-Bench。
16. **加推理算力不一定有帮助。** 见 CausalGame 的 XHigh 反而更低，以及 AutumnBench 中加算力只在约 58% 的环境有效。

### (c) 对"因果图世界 + 控制图复杂度 = 控制难度"的证据

**支持的证据【事实】：**
- 隐藏 DFA 规模一增大，性能就急剧下降（arXiv 2606.16576）。
- bnlearn 网络超过 50 节点后，LLM 远落后于经典算法（arXiv 2404.06349）。
- NewtonBench 在 system complexity 为 complex 时显著下降，最难设定下 GPT-5 只有 29.9–40.3%（arXiv 2510.07172）。
- ZebraLogic 的搜索空间和 Z3 conflicts 与准确率单调负相关（arXiv 2502.01100）。
- SciGym 的反应数越多越难（arXiv 2507.02083）。
- DyVal 的 DAG 深度与宽度（arXiv 2309.17167）。
- Auto-Discovery-Bench 从 3 节点到 10 节点，Gemini-1.5 由 85% 降到 15%（arXiv 2502.15224）。

**反驳或复杂化的证据：**
1. **【事实 → 推断】曲线随代际整体平移。**
   - Auto-Discovery-Bench 的 10 节点：GPT-4o 15% → Grok-4 / Gemini-2.5-pro 100%；
   - ZebraLogic 从 o1-preview 在 X-Large 上 17%，到约 97%；
   - CLadder 从 70% 到 95%。
   - 所以复杂度旋钮能控制"相对难度"，无法控制"绝对难度的寿命"。
2. **【事实】规模型难度可以外包给算法或代码。**
   - L* 远比 LLM 稳健；
   - 经典主动干预在 15 节点几乎完全可识别；
   - A-CBO 以 O(log n) 轮收敛，图越大优势越大；
   - coding agent 会写世界模拟器（ARC-AGI-3、Compiled Agency）。
   - 【推断】只要允许工具使用，"更大的图"对前沿 agent 只是"更长的算法运行时间"。
3. **【事实】ARC-AGI-3 的跃升与图的复杂度无关。** 同一批环境从 0.51% 到 62.7% 再到 99.9%，驱动因素是 harness、持久推理和可执行世界模型。
4. **【事实】小图依然难。**
   - CausaLab 3–7 节点下，GPT-5.2-high 的 edge F1 只有 0.471；
   - CausalGame 靠混杂、选择偏差这类陷阱，而不是图的规模，就难住了 30 个模型。
   - 【推断】难度的主轴是信息结构，不是图大小。信息结构包括：是否可识别、是否需要干预、有无隐藏混杂和选择偏差、噪声水平、机制是否非线性、是否反语义或反先验。
5. **【事实】图生成的题目本身可能不可解或不唯一。**
   - SciGym 作者承认部分系统不可识别；
   - DyVal 被批评不保证答案唯一；
   - CausalGame 的模拟器 API 泄漏了答案。
   - 【推断】"有显式 SCM ⇒ 题目正确"只保证了前向计算正确，不保证可识别、唯一或不可旁路。

**【推断】对假设的修订建议：**
- 原假设："因果图世界 + 控制图复杂度 = 控制难度"。
- 修订为："因果图世界 + 控制信息结构（可识别性、干预必要性、混杂、噪声、反先验）+ 控制 agent 行为负担（实验预算、验证需求、状态跟踪长度）= 控制难度"。
- 图规模只作为次要旋钮，并且必须用算法基线（L*、PC/GES/GIES、主动干预、A-CBO）校准。如果一个经典算法加少量代码就能解，这道题就不算难。

---

## 11. 对 WorldSpec 的设计原则（全部为【推断】，括号内为依据）

1. **生成器私有，机制族轮换。** 每轮评测换一批新的机制类型，而不仅是新参数。依据：ARC-AGI-3 的新机制、Reasoning Gym 等公开即被训练、PROPEL。
2. **计分分三层：**
   - 任务完成；
   - 机制恢复（edge F1、方程符号等价、规则等价）；
   - 实验效率（EIG regret、类 RHAE 的步数比）。
   依据：DiscoveryWorld、BoxingGym、CausaLab、ARC RHAE。
3. **生成时做可识别性闸门。** 对每道题确认：在给定的干预预算下，真值图可以被唯一恢复（或者真值就定义为等价类）。依据：SciGym、DyVal 的教训。
4. **用算法基线做对抗校准。** 如果 L*、PC/GIES、主动干预、A-CBO、"写模拟器加 BFS"能以低成本解出，就降权或剔除。也要用随机、常数、单动作基线做反旁路检查。依据：ARC 的 1/10,000 随机界、"Explore Before You Solve"。
5. **加入信息结构陷阱：** 隐藏混杂、选择偏差、测量误差、噪声、非线性机制、反语义变量名。依据：CausalGame、NewtonBench 的噪声实验、Caliper、CausalBench。
6. **反确定性可模拟。** 引入随机性和部分可观测，让"逐字写模拟器"不再等于"已解"。依据：ARC-AGI-3 被 coding agent 攻克。注意，AutumnBench 中模型在随机环境里反而更好，所以随机性本身不构成难度，只能作为反旁路手段，需要实测。
7. **固定 Standard harness，并按 harness × effort × memory 分格报告。** 依据：Astra 62.7% 对 99.9%、Tycho 1.5 对 88.49。
8. **信息隔离与 hack 审计。** 模拟器 API 不暴露场景 ID；agent 的"成功声明"要由环境验证；上线前做 fuzzing。依据：CausalGame 的 +18.5pp 泄漏、ARC 的 null-coordinate。
9. **人类校准。** 每个环境至少让少量首次接触的人测试，确保"对人可解"。依据：ARC-AGI-2/3。
10. **public / private 一致性监控。** 设定容差（例如 ±15pp），超过就视为过拟合信号。依据：ARC policy。
11. **测试时换环境。** 例如 change detection 或反事实查询，迫使 agent 必须真的学到世界模型。依据：AutumnBench。
12. **把"验证步骤"和"早停"作为可观测的过程事实记录。** 这可以作为结构化行为出口的信号。依据：CausaLab 的验证 +12pp、预算剩一半。

---

## 12. 未查到 / 冲突 / 未核实清单

**未查到：**
- 2026 年前沿模型（GPT-5.5+/Opus 5/Fable 5.x/Astra）在以下基准上的成绩：
  - NewtonBench
  - LLM-SRBench
  - BoxingGym
  - DiscoveryWorld
  - KOR-Bench
- Fable 5.1 的 ARC-AGI-3 成绩（据称因 API 误判而未测完）。
- 以下基准的 2026 年饱和曲线：
  - Reasoning Gym
  - Enigmata
  - SynLogic
  - KORGym
  - InternBootcamp
- DyVal 的饱和证据。
- 同时满足"前沿模型 × 交互式 × 10 节点以上 × 以可识别性上限计分"的隐藏 SCM 基准（空白）。
- CDCR-SFT 与 BBEH 的原始 URL（仅见搜索摘录）。

**冲突：**
- ARC-AGI-3 人类基线："upper-median" 与"10 人中第二好"两种说法。
- Astra 在 Adapter 下的分数：98.6% 与 99.9%。以 ARC 官方页 https://arcprize.org/blog/astra 为准。
- NewtonBench 版本间数字：GPT-5 为 72.9% 或 75.9%，最难设定为 29.9% 或 40.3%。
- AutumnBench 的人类表现：论文称接近最优，Basis 早期博客称远低于上限。
- Milestone #2 的奖金金额在各来源间不一致（未在正文使用）。

**未核实（只见聚合站或二手来源）：**
- BenchLM 的 ARC-AGI-3 榜单。
- llm-stats 的 Astra ARC-AGI-2 95%、ZebraLogic 0.973。
- Lacuna 的 AutumnBench Opus 4.6 39.9% 及 ρ=0.89。
- Schema harness 自报的约 99%。
- "调好的 harness 让 Opus 4.6 在某环境 97.1%、另一环境 0%"的原始出处。
