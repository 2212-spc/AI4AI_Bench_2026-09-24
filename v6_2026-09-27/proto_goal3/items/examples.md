# 已认证题示例（自动生成：`python3 render.py`）

每个模板取一题：search T2 中 clean、优先机制驱动、红队裕度最大的那题；另加一道 M1 的“T1 认证但被红队杀”的题，展示分层证书。路由表按离真值的距离排序；B 由离真值最近的计分路由决定。

**注意**：题面是草图。本原型只用门里的模拟参考运行来认证题目，还没有实现给 agent 调用的 lab API（买运行、读笔记本）；把这些题接到 v5 的 lab 服务上是下一步（README §7）。第一版渲染就查出了一类泄漏：M3 在笔记本恰好选中在位配方时，题面公布的在位值就是答案；M2 的删失阈值 c 可能紧挨中位数。修复后新增了 `shown` 登记和 G7 的 ≥1T 检查（README §5）。

## 例 M1 — K2_lrbasin × M1

### 题面（agent 看到的）

**Card**: final loss vs log2 learning rate (asymmetric basin) (x = log2(lr / lr_ref)).

**Notebook (free)**: ctx0 dense sweep (13 pts, 3-seed means) + one production-context pilot run at each bracket point (noise x3.8). Context: the notebook swept LR at batch b0; production uses 4*b0, which moves the optimum by beta*log2(4) (beta=0.5 is the sqrt rule, 1.0 the linear rule; the world's beta is unknown).

**Question**: what is the expected value in the *production* context at x = -1.488?

**Lab**: you may buy up to 227 production runs at any of x ∈ [-2.1, -1.68, -1.26, -0.84] (each bought run has the lab's standard seed noise; the notebook pilots' noise multiplier is stated above). Answer with one number.

### 判分侧（agent 看不到）

真值 2.7225；容差 T = 0.00272（参考方法 90 分位误差 × 2.25）；B_lo（含红队）= 4.64，B_lo（仅基础对手）= 7.71（认证要求 ≥3）；κ = 1.00；n = 227；现实性标记：无；驱动常数：a（mechanism）。

世界常数：ys=2.584, a=0.01799, b=0.005868, xo=-0.4672, beta=0.9626, dy=0.01767, s0=0.00421, tau=0.1095, tau2=0.02418, tau_r=0.004817

| 路由 | 层 | 值 | 离真值 / T | 代表的失败 |
|---|---|---|---|---|
| `R3:ctx0_slope@near` | 红队 | 2.7351 | 4.64 | 最近生产点 + 笔记本曲线的斜率做线性外推 —— 假设上下文不改变斜率 |
| `naive_ignore:ctx:ratio@near` | 红队 | 2.7354 | 4.74 | 同上，但假设上下文效应是常数比例 |
| `naive_ignore:ctx:offset@near` | 红队 | 2.7358 | 4.88 | 笔记本曲线 + 最近一个生产点做常数平移迁移 —— 假设上下文效应是常数偏移 |
| `R1:readoff` | 基础 | 2.7435 | 7.71 | 读最近网格点的值 |
| `naive_ignore:ctx:pool` | 红队 | 2.6697 | 19.38 | 把两种上下文的数据混在一起拟合二次曲线 —— 假设上下文可忽略 |
| `R2:endpoint` | 基础 | 2.6613 | 22.46 | 读区间端点 |
| `R8:saturated` | 基础 | 2.6613 | 22.46 | 假设最大配置已经到达下限（M6）/ 假设响应已饱和、读最远网格点（M1） |
| `B_prior:ctx0` | 基础 | 2.6028 | 43.97 | 直接用笔记本（旧上下文）在目标点的值 —— 忽略上下文变化 |
| `valid:R3:local@op` | 合法替代（不计分） | 2.7149 | 2.78 | 与参考同阶的其他正确方法 |
| `valid:R3:local@near` | 合法替代（不计分） | 2.7218 | 0.24 | 与参考同阶的其他正确方法 |
| `valid:R4:secant@op` | 合法替代（不计分） | 2.7296 | 2.62 | 与参考同阶的其他正确方法 |
| `valid:R4:secant@near` | 合法替代（不计分） | 2.7233 | 0.29 | 与参考同阶的其他正确方法 |
| `valid:R7:average` | 合法替代（不计分） | 2.7246 | 0.75 | 与参考同阶的其他正确方法 |
| `valid:control_variate` | 合法替代（不计分） | 2.7225 | 0.00 | 与参考同阶的其他正确方法 |
| `free:pilot_interp` | 免费数据（G7） | 2.7134 | 3.36 | 只用免费数据的诚实估计，必须 ≥2T |
| `shown:pilot@xl` | 题面数字（G7） | 2.7454 | 8.40 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |
| `shown:pilot@xr` | 题面数字（G7） | 2.6753 | 17.34 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |

## 例 M2 — K5_critbatch × M2

### 题面（agent 看到的）

**Card**: steps-to-target vs log2 batch (S = S_min (1 + B_crit / B)) (x = log2(B / B0)).

**Setting**: 9 units (eval domains / problem categories) at the target config x = 1.697. runs that do not reach the target inside the step cap are logged only as '> cap'; the threshold is c = 17.9362.

**Notebook (free)**: all 9 units measured at the older config x=0.10.

**Question**: the median over all 9 units of the expected value at x = 1.697.

**Lab**: up to 207 runs (23 per unit). Answer with one number.

附加条款（G8：该读法落在 T 外，所以题面要写明）：the median is over all 9 units, including those logged only as a bound

### 判分侧（agent 看不到）

真值 14.547；容差 T = 0.348（参考方法 90 分位误差 × 2.25）；B_lo（含红队）= 7.12，B_lo（仅基础对手）= 7.12（认证要求 ≥3）；κ = 1.00；n = 207；现实性标记：无；驱动常数：bc（mechanism）。

世界常数：Smin=4.019, bc=2.929, gam=1.838, srel=0.02563, tau=0.2782, tau2=0.0462, tau_r=0.0075

| 路由 | 层 | 值 | 离真值 / T | 代表的失败 |
|---|---|---|---|---|
| `B_guess:mean_uncensored` | 基础 | 11.886 | 7.65 | 未删失单元的均值 |
| `drop:censored` | 基础 | 11.578 | 8.54 | 丢掉删失单元后取中位数 —— 把“只知道下界”当成“没有信息” |
| `B_guess:threshold` | 基础 | 17.936 | 9.75 | 把删失阈值 c 当作中位数 —— 题面给出的数字被直接抄用 |
| `drop:censored_mean_nb_shift` | 红队 | 18.64 | 11.77 | 均值版的平移迁移，且丢掉删失单元 |
| `naive_ignore:unit_slope` | 红队 | 19.087 | 13.06 | 笔记本配置的中位数 + 未删失单元的中位平移 —— 假设各单元平移相同 |
| `R1:notebook_config` | 基础 | 36.274 | 62.48 | 直接用笔记本配置下的全体中位数 —— 忽略配置变化 |
| `valid:censor_as_c` | 合法替代（不计分） | 14.547 | 0.00 | 与参考同阶的其他正确方法 |
| `valid:symmetric_trim` | 合法替代（不计分） | 14.547 | 0.00 | 与参考同阶的其他正确方法 |
| `shown:threshold_c` | 题面数字（G7） | 17.936 | 9.75 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |
| `shown:notebook_median` | 题面数字（G7） | 36.274 | 62.48 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |

## 例 M3 — K4_passk × M3

### 题面（agent 看到的）

**Card**: pass@k vs log2 k under heterogeneous per-problem success rates (Beta mixture) (x = log2(k)), all recipes at x = 4.859 (higher is better).

**Notebook (free)**: 12 recipes, one run per recipe, eval noise sd 0.0937 (stated); recipe 0 is the incumbent. The notebook picked recipe 7.

**Question**: the true expected value of the recipe the notebook picked.

**Lab**: up to 283 runs of any recipe. Answer with one number.

附加条款（G8：该读法落在 T 外，所以题面要写明）：the true value of the recipe the notebook selected, not of the best recipe on the menu

### 判分侧（agent 看不到）

真值 0.41434；容差 T = 0.00487（参考方法 90 分位误差 × 2.25）；B_lo（含红队）= 7.38，B_lo（仅基础对手）= 9.81（认证要求 ≥3）；κ = 1.00；n = 283；现实性标记：无；驱动常数：pbar（mechanism）。

世界常数：pbar=0.08815, s=1.485, s1=1.637, tau=0.9831, tau2=0.0914, tau_r=0.02997

| 路由 | 层 | 值 | 离真值 / T | 代表的失败 |
|---|---|---|---|---|
| `R7:top2_mean` | 红队 | 0.45316 | 7.98 | 报告前两名的均值（半吊子收缩） |
| `B_guess:notebook_plus_1se` | 红队 | 0.37503 | 8.08 | 冠军数字 + 1 个标准误的“修正” —— 知道有偏但修正量拍脑袋 |
| `R1:incumbent_value` | 基础 | 0.36273 | 10.61 | 报告在位配方（已发表）的值 |
| `B_guess:notebook_value` | 基础 | 0.46869 | 11.17 | 直接报告笔记本里冠军配方的数字 —— 赢家诅咒 |
| `R7:menu_mean` | 基础 | 0.35331 | 12.54 | 报告全菜单均值（过度收缩） |
| `free:james_stein` | 免费数据（G7） | 0.36202 | 10.75 | 只用免费数据的诚实估计，必须 ≥2T |
| `shown:incumbent_published` | 题面数字（G7） | 0.36273 | 10.61 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |
| `shown:notebook_best` | 题面数字（G7） | 0.46869 | 11.17 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |

## 例 M6 — K1_compute × M6

### 题面（agent 看到的）

**Card**: loss vs log10 compute (saturating power law) (x = log10(C / C0)).

**Question**: going from x = 0.185 to x = 0.950, what fraction of the *reducible* value (above the floor) is removed?  Floor: the synthetic data generator's entropy can be probed directly (noisy probe run).

**Notebook (free)**: one run at xa, xb and one floor-estimate run (noise x2.0).

**Lab**: up to 160 runs at any of x ∈ [0.185, 0.95, 2.188] (floor instrument included). Answer with a fraction.

### 判分侧（agent 看不到）

真值 0.38582；容差 T = 0.00344（参考方法 90 分位误差 × 2.25）；B_lo（含红队）= 17.55，B_lo（仅基础对手）= 17.55（认证要求 ≥3）；κ = 2.00；n = 160；现实性标记：无；驱动常数：E（mechanism）。

世界常数：E=1.989, A=1.676, al=0.2767, D=0.03863, xr=1.721, s0=0.01139, tau=0.08631, tau2=0.03322, tau_r=0.0159

| 路由 | 层 | 值 | 离真值 / T | 代表的失败 |
|---|---|---|---|---|
| `B_prior:nominal_floor` | 基础 | 0.32139 | 18.75 | 背出版常数当下限（Chinchilla E=1.69）—— 不测本世界的下限 |
| `R4:secant_floor` | 红队 | 0.4509 | 18.94 | 用两点割线外推下限 |
| `R8:saturated` | 基础 | 0.53528 | 43.50 | 假设最大配置已经到达下限（M6）/ 假设响应已饱和、读最远网格点（M1） |
| `naive_ignore:floor` | 基础 | 0.16523 | 64.19 | 假设下限为 0 |
| `free:notebook_probe` | 免费数据（G7） | 0.37147 | 4.17 | 只用免费数据的诚实估计，必须 ≥2T |

## 例 M1-T1（分层证书：对基础对手认证、被红队杀） — K1_compute × M1

### 题面（agent 看到的）

**Card**: loss vs log10 compute (saturating power law) (x = log10(C / C0)).

**Notebook (free)**: ctx0 dense sweep (13 pts, 3-seed means) + one production-context pilot run at each bracket point (noise x1.0). Context: the notebook's sweep used fresh data; production is data-constrained and starts repeating tokens past x_rep, adding a softplus penalty of D nats per decade.

**Question**: what is the expected value in the *production* context at x = 1.723?

**Lab**: you may buy up to 137 production runs at any of x ∈ [1.545, 1.665, 1.784, 1.904] (each bought run has the lab's standard seed noise; the notebook pilots' noise multiplier is stated above). Answer with one number.

### 判分侧（agent 看不到）

真值 2.2144；容差 T = 0.0028（参考方法 90 分位误差 × 2.25）；B_lo（含红队）= 1.01，B_lo（仅基础对手）= 6.85（认证要求 ≥3）；κ = 1.00；n = 137；现实性标记：无；驱动常数：D（mechanism）。

世界常数：E=1.733, A=1.959, al=0.37, D=0.05928, xr=1.248, s0=0.0104, tau=0.1293, tau2=0.04553, tau_r=0.01903

| 路由 | 层 | 值 | 离真值 / T | 代表的失败 |
|---|---|---|---|---|
| `naive_ignore:ctx:offset@near` | 红队 | 2.2114 | 1.07 | 笔记本曲线 + 最近一个生产点做常数平移迁移 —— 假设上下文效应是常数偏移 |
| `naive_ignore:ctx:ratio@near` | 红队 | 2.2111 | 1.17 | 同上，但假设上下文效应是常数比例 |
| `R3:ctx0_slope@near` | 红队 | 2.2108 | 1.27 | 最近生产点 + 笔记本曲线的斜率做线性外推 —— 假设上下文不改变斜率 |
| `naive_ignore:ctx:pool` | 红队 | 2.2012 | 4.70 | 把两种上下文的数据混在一起拟合二次曲线 —— 假设上下文可忽略 |
| `R1:readoff` | 基础 | 2.2346 | 7.21 | 读最近网格点的值 |
| `B_prior:ctx0` | 基础 | 2.1842 | 10.80 | 直接用笔记本（旧上下文）在目标点的值 —— 忽略上下文变化 |
| `R2:endpoint` | 基础 | 2.1598 | 19.49 | 读区间端点 |
| `R8:saturated` | 基础 | 2.1598 | 19.49 | 假设最大配置已经到达下限（M6）/ 假设响应已饱和、读最远网格点（M1） |
| `valid:R3:local@op` | 合法替代（不计分） | 2.2093 | 1.82 | 与参考同阶的其他正确方法 |
| `valid:R3:local@near` | 合法替代（不计分） | 2.2138 | 0.23 | 与参考同阶的其他正确方法 |
| `valid:R4:secant@op` | 合法替代（不计分） | 2.2201 | 2.04 | 与参考同阶的其他正确方法 |
| `valid:R4:secant@near` | 合法替代（不计分） | 2.2149 | 0.19 | 与参考同阶的其他正确方法 |
| `valid:R6:loglinear` | 合法替代（不计分） | 2.2154 | 0.35 | 与参考同阶的其他正确方法 |
| `valid:R7:average` | 合法替代（不计分） | 2.2172 | 1.02 | 与参考同阶的其他正确方法 |
| `valid:control_variate` | 合法替代（不计分） | 2.2144 | 0.02 | 与参考同阶的其他正确方法 |
| `free:pilot_interp` | 免费数据（G7） | 2.2048 | 3.43 | 只用免费数据的诚实估计，必须 ≥2T |
| `shown:pilot@xl` | 题面数字（G7） | 2.2268 | 4.42 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |
| `shown:pilot@xr` | 题面数字（G7） | 2.1821 | 11.55 | 题面给出的同单位数字，必须 ≥1T（抄它不能判对） |
