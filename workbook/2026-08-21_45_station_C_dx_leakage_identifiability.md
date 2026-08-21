# 2026-08-21 (45) station–C_dx 泄漏算子与层级同时校准的统计不可实现性

## 任务

条目 44 已把 hierarchical alignment V1 的失败收缩为 **station ↔ `C_dx`
cross-level identifiability / leakage**，并判定**不冻结**联合层级。本条目
不再提交大规模物理 refit，不再增加任何 layer/module 自由度，只利用已有
iteration-00 FD Jacobian、cluster 999304 / 999493 / 999500 观测，做一次
定量终审：

1. 记 station Jacobian 为 `J_s`、layer contrast 列为 `j_c`，在
   train/validation × truth-selected/route-selected × start/heldout 上计算
   一阶泄漏算子
   `A = (J_s^T W J_s)^{-1} J_s^T W j_c`。
2. 给出每 1 µm `C_dx` 会伪装成多少 `dx/dy/rx/ry/rz` correction，并核对
   已观测的 `dx error / C_dx ≈ −59`、`ry error / C_dx ≈ −32` 是否是稳定的
   Jacobian 几何，而不是某个 operating point 的偶然结果。
3. 用已冻结的 station capture tolerance 反推 station solve 之前允许残留的
   最大 `|C_dx|`，并与已注册 `σ(C_dx)=6.91 µm`、反向路径 leftover 6–8 µm
   直接比较。
4. 只增加一个分析级、train-only 的 nuisance-robust control：不改 payload、
   不增加物理 DoF，用 `j_c` 对 station residual space 做 weighted
   projection / Schur-complement profiling，比较普通 station WLS 与
   「对未知 `C_dx` 一阶不敏感」的 station estimator。validation 只做一次
   冻结 transfer。
5. 据此回答失败究竟是可投影掉的 estimator cross-talk，还是物理观测空间
   本身的不可兼容精度要求；若是后者，正式关闭 hierarchical V1 rescue。

冻结不变：mode-0、V2、`physical_edge_deduplicated`、station 5-DoF + survey
`dz`、1-D `C_dx` capture、sealed test。禁止联合 Newton 写 payload。

## 算子定义

Jacobian 一律在 `iteration_00_reference`（全零）线性化；`W` 是对应观测集
的 anchor 协方差逆。`C_dx` 只作为分析列，不写入 Athena。

- **`A_5`**：用户指定的五维 `J_s = (dx, dy, rx, ry, rz)`，无 `dz`。
- **`A_6`**：与生产 station 步一致，六列含 survey `dz`（5 mm prior）。
  与条目 44 已观测的 `error / C_dx` 直接可比。

`dx/ry` 上 `A_5` 与 `A_6` 在 0.1% 内一致；survey `dz` 不改变泄漏几何。
下文数字默认 `A_6`（生产可比），并同时给出 `A_5`。

同单位读法：`A_dx`（mm / mm `C_dx`）等于每 1 µm `C_dx` 伪装成的 station
`dx` 微米数；`A_ry`（mrad / mm `C_dx`）等于每 1 µm `C_dx` 伪装成的 `ry`
微弧度数。

Schur 剖面：

```
N_ss^⊥ = N_ss − N_sc N_cc^{-1} N_cs
```

这是对未知 `C_dx` 一阶不敏感的 station 正规矩阵。`C_dx` 回收值只作 nuisance
诊断，不写 payload。

## 输入（全部已有，无新 Condor）

| 角色 | 路径 |
| --- | --- |
| FD Jacobian | `outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/`（cluster 999304） |
| 联合注入观测 | 同上，`iteration_00_start` / `heldout_00` |
| station-first remaining | cluster **999493** |
| reverse remaining | cluster **999500** |
| synthetics / V2 | iteration-00 + 两条 remaining 的已物化 sample；冻结 V2 只在 reference |
| station capture | `outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/` |
| `C_dx` σ | `outputs/mc24_ift_layer_contrast_2d_capture_criteria_train_v1/` 的 `C_dx` 行 |

24 个角：2 split × 2 观测种类 × 2 点 × 3 套 geometry（iteration-00 /
999493 / 999500）。`A` 本身由零点 Jacobian + 观测集 `W` 决定，与 remaining
operating point 几乎无关；remaining 用来验证经验比 `error/C_dx` 是否仍落在
同一几何上。

产物：`outputs/mc24_ift_hierarchical_v1_leakage_identifiability_audit_v1/`。

## 每 1 µm `C_dx` 伪装成的 station correction

生产可比算子（train route-selected、`iteration_00_start`，591 条去重边，
`A_6` + 5 mm `dz` prior）：

| DoF | `A`（native / mm `C_dx`） | 每 1 µm `C_dx` |
| --- | ---: | ---: |
| `dx` | −59.21 | **−59.2 µm** |
| `dy` | +1.70 | +1.70 µm |
| `dz`（survey） | +10.12 | +10.1 µm |
| `rx` | −2.03 | −2.03 µrad |
| `ry` | −31.91 | **−31.9 µrad** |
| `rz` | −0.37 | −0.37 µrad |

五维 `A_5`：`dx −59.27`、`dy +1.67`、`rx −1.99`、`ry −31.93`、`rz −0.26`。
与 `A_6` 的 `dx/ry` 无法区分。

`j_c` 落在 station 列空间的比例：train route-selected
**`R² = 0.99966`**（子空间余弦 0.99983）；train truth-selected
`R² = 0.9988`。`j_c` 与 station `dx` 列余弦 **−0.94**。这不是弱相关，而是
近共线。

## `−59 / −32` 是 Jacobian 几何，不是 operating-point 偶然

iteration-00 八个角（train/val × truth/route × start/heldout）的 `A_6`：

| 量 | 均值 | 范围 | 相对散布 |
| --- | ---: | ---: | ---: |
| `A_dx` | −57.70 | −59.21 … −55.10 | 7.1% |
| `A_ry` | −31.23 | −31.91 … −29.66 | 7.2% |
| 经验 `dx error / C_dx` | −57.72 | −59.23 … −54.42 | 8.3% |
| 经验 `ry error / C_dx` | −31.37 | −32.27 … −29.32 | 9.4% |

Train 上 `A` 钉在 **−59.16 … −59.21**（`dx`）和 **−31.76 … −31.91**（`ry`），
truth 与 route-selected 一致，start 与 heldout 一致。Validation 略弱
（`A_dx ≈ −55 … −57`），仍是同一几何，符合已知 validation 离群源
`mc24_100047_00150_00199` 的统计差异，不是另一种物理。

999493 / 999500 remaining 上 **`A` 与 iteration-00 相同**（Jacobian 仍在
零点）。经验比在 remaining 上仍为 −59 量级；validation remaining 因非线性
略偏到 −58 … −60，仍不是新机制。

线性模型在 train route-selected start 上的一阶残差：
普通 station `dx` 误差 5.429 mm，对 `A·C_dx` 的预测 5.431 mm，差
**−0.002 mm**；`ry` 2.934 vs 2.927 mrad。条目 44 看到的 −59 / −32
就是这个算子作用在注入 `C_dx = −0.0917 mm` 上的结果。

判定：**是稳定的 Jacobian 几何性质。**

## 反推 station solve 前允许的最大 `|C_dx|`

冻结 station capture（条目 36；不重调）：

| 约束 | 限制 | 反推 `max |C_dx|` |
| --- | ---: | ---: |
| `dx` 工程 0.1 mm | 0.1 / 59.21 | **1.69 µm** |
| `dx` 统计 3σ（σ=0.02886 mm） | 0.0866 / 59.21 | **1.46 µm** |
| `ry` 工程 1.0 mrad | 1.0 / 31.91 | 31.3 µm |
| `ry` 统计 3σ（σ=0.1114 mrad） | 0.334 / 31.91 | 10.5 µm |

绑定项是 **station `dx`**。要保持 `dx` 在 0.1 mm 工程门内，station solve
之前的 `|C_dx|` 残留必须 ≲ **1.7 µm**；框架统计门更紧，≲ **1.5 µm**。

对照层级另一侧已经注册、不可重调的层估计精度：

| 量 | 值 |
| --- | ---: |
| 已注册 `σ(C_dx)` | **6.91 µm** |
| 反向路径 leftover（999500，start / heldout） | **6.17 / 8.36 µm** |
| `0.1 mm / 59` | 1.69 µm |

`σ(C_dx)` 已经是工程预算的 **4.1 倍**、统计预算的 **4.7 倍**。反向路径
实际 leftover 6–8 µm 与层估计 1σ 同量级，同样超标。当前 layer estimator
达不到 1.7 µm。迭代次数不能改变这个信息比。

**在当前 reconstruction / sample 下，层级同时校准在统计上不可实现。**

## Nuisance 投影 control（分析级，不写 payload）

Train 上构造对未知 `C_dx` 一阶不敏感的 station estimator；validation
只做冻结 transfer，不参与投影/阈值选择。

Train route-selected（生产观测）：

| 量 | 普通 station WLS | `C_dx`-profiled |
| --- | ---: | ---: |
| 5-DoF 秩 | 5 / 5 | **5 / 5**（仍满秩） |
| 5-DoF 条件数 | 2.06×10³ | **1.86×10⁵**（×90） |
| `σ(dx)` | 0.044 mm | **1.35 mm**（×31） |
| `σ(ry)` | 0.053 mrad | **0.73 mrad**（×14） |
| 框架 capture | false | **false** |
| 工程 capture | false | **false** |
| 覆盖 capture | false | true（σ 膨胀把 3σ 门撑开） |

Profiled `dx` 误差仍为 **−1.24 mm**（真值 0.430 mm），`ry` 误差
**−0.66 mrad**。Nuisance `C_dx` 回收 −0.113 mm（真值 −0.092 mm）——层列
的正交残渣极小（`1−R² ≈ 3.4×10⁻⁴`），用它去「解开」station 会把噪声放
大成毫米级 station 误差。

Truth-selected train：`σ(dx)` ×27 到 0.30 mm，仍大于 0.1 mm 工程门；
profiled `dx` 误差 **−6.41 mm**。逐源 profiled `dx` 散布 **12.9 mm**
（普通 WLS 1.46 mm）；其中一个源回收 −11.7 mm。投影后 source 不稳定。

Validation 冻结 transfer：全部角 profiled capture 失败。不开放小规模
真实 physical verification。

结论：5-DoF 形式上仍满秩，但 `dx/ry` 的可分辨信息几乎就是 `C_dx` 的同一
个方向。投影没有消去 estimator cross-talk 后留下可用的 station `dx`；
它把那一维一起投掉了。失败属于 **物理观测空间的不可兼容精度要求**，
不是再换一个算法就能解开的 cross-talk。

## 正式关闭 hierarchical V1 rescue

- **允许小规模物理验证：** 否。
- **关闭 V1 rescue：** 是。
- **失败类型：** `incompatible_precision_of_physical_observation_space`。

冻结结论：

**station 5-DoF + survey `dz` 与 IFT 1-D `C_dx` 都各自可测，但在当前
FASER track sample / reconstruction 下不是可同时求解的层级参数，必须
作为独立 calibration mode 使用。**

不把它们放进同一个 Newton，也不再按 block hierarchy 顺序互为 leftover。
不增加 `C_rx` / relative ry / layer dy/rz / layer1 / module 来硬救。
不靠迭代次数。不重调 V2、mode-0、route、χ²、capture。

各自独立使用时：station 步要求几何中没有未模型化的 `C_dx`（或把内部
形变留给 survey / 另一套 dedicated `C_dx` 模式）；`C_dx` 步要求 station
六矢已经闭合。两者都已在条目 36 与 39/40 上单独冻结，这里不改。

条目 46 把上述泄漏预算固化为 mode-validity contract，并把两种独立
mode 写成可执行的 **FASER alignment operating protocol V1**。不再把
hierarchical V1 当作联合层级生产。

## 代码与回归

- `alignment/hierarchical_v1_leakage.py`：`A`、预算反演、Schur 剖面、σ
  膨胀 / 秩 / 条件数。
- `scripts/audit_hierarchical_v1_leakage.py`：24 角审计（无新 refit）。
- `tests/test_hierarchical_v1_leakage.py` + `tests/test_hierarchical_v1.py`：
  16 项通过。

无新 Condor。无 payload 变更。test 未打开。
