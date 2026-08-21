# 2026-08-20 (42) 第一份 C_dx+C_rx 二维 source-disjoint mini-curriculum：联合简并，停在两个 1-DoF

## 任务

目标不再是证明单自由度可辨识，而是检验已经分别通过 truth-selected 与冻结 V2
route-selected 闭合的 `C_dx`、`C_rx` 在**联合扰动**、**独立 validation source**、
以及真实 `associate → estimate → write layer payload → refit → reassociate`
之后是否仍能稳定解耦。

冻结不变：station 5-DoF、station dz、mode-0、V2 checkpoint/calibration/route
policy、`physical_edge_deduplicated`、sealed test。canonical 坐标只使用
`C_dx=(dx_L0-dx_L2)/2`、`C_rx=(rx_L0-rx_L2)/2`。实际 payload 始终
`L0=(+C_dx,+C_rx)`、`L1=0`、`L2=(-C_dx,-C_rx)`，station 六矢严格为 0。
禁止把冻结 station 的 `reference_layer` 当 gauge-equivalent fit。不注入
relative ry、layer dy/rz、layer1 或 module。

## 预注册 dual gate（validation 未参与）

在打开任何 10/8 closure 之前，只用已经闭合的 3-source **train** route-selected
1-D 结果注册。schema
`faser-ift-layer-contrast-2d-capture-criteria-v1`。框架门是
**statistical**（`|error|≤3σ_registered`）**且** **this-fit coverage**
（`|error|≤3σ_this-fit`）。engineering 只作记录。

| 参数 | σ_registered | 注册误差 | 边数 | 来源 |
| --- | --- | --- | --- | --- |
| `C_dx` | **0.00691 mm** | +0.00012 mm | 193 | 既有 `sum_to_zero` 三层协方差，映射 `C=(L0−L2)/2`；不重调 `C_dx` |
| `C_rx` | **0.146 mrad** | −0.00030 mrad | 195 | 既有 `outer_contrast` 1-D 解，layer0 σ ≡ `C_rx` |

engineering 信息量：`C_dx` 0.03 mm、`C_rx` 0.15 mrad（与 1-D truth-selected
internal_abs 同尺度）。3σ_registered 约为 0.021 mm / 0.437 mrad。
`validation_used_in_registration=false`，`test_data_accessed=false`。

产物：`outputs/mc24_ift_layer_contrast_2d_capture_criteria_train_v1/capture_criteria.json`。

## 既有 FD 解析构造的两列 contrast Jacobian（未重产 40 probe）

复用 identifiability 的 layer0/layer2 dx 与 rx 中心差分：
`J_C_dx = J_L0_dx − J_L2_dx`，`J_C_rx = J_L0_rx − J_L2_rx`。
3 个 train source，694 条 truth-selected IFT 边。

| | |
| --- | --- |
| 秩 | **2 / 2** |
| 条件数 | 222 |
| 信息余弦 | **−0.888**（阈值 0.9，故 `near_degenerate=false`） |
| 未加权列余弦 | −0.710 |
| 预飞行 | **通过** |

弱方向几乎是 `C_rx`（SVD 组成 0.99），与 1-D 上 `C_rx` 比 `C_dx` 粗一个量级一致。
0.888 的相关是**监视项**，不是预飞行否决。10/8 联合扰动才决定这两个 mode 是否
在真实联合注入下互相吸收。此 3-source Jacobian **不**搬到 18 源 Newton 步；
10/8 bank 自己测量四个 contrast-space FD。

产物：`outputs/mc24_ift_layer_contrast_2d_jacobian_from_existing_fd_v1/contrast_jacobian.json`。

## 联合随机采样（非两个独立轴向 scan）

椭圆 `|C_dx|/0.12`、`|C_rx|/0.70`，半径 r≤1，`min(|u|,|v|)≥0.25` 拒绝近轴点。
seed **20260820**。FD 仍是轴向（中心差分必须如此），步长落在包络内：
`C_dx` ±0.10 mm、`C_rx` ±0.50 mrad。

| 点 | 角色 | `C_dx` / mm | `C_rx` / mrad | r |
| --- | --- | --- | --- | --- |
| `iteration_00_reference` | 名义 / FD 锚 | 0 | 0 | 0 |
| `iteration_00_fd_C_dx_{p,m}` | 轴向 FD | ±0.10 | 0 | 0.83 |
| `iteration_00_fd_C_rx_{p,m}` | 轴向 FD | 0 | ±0.50 | 0.71 |
| `iteration_00_start` | 主联合起点 | **+0.0403** | **+0.389** | 0.65 |
| `iteration_00_linear_00` | 较小联合 | +0.0277 | +0.155 | 0.32 |
| `iteration_00_linear_01` | 较小联合 | +0.0550 | +0.100 | 0.48 |
| `iteration_00_stress_00` | 近边界 | +0.0493 | +0.607 | 0.96 |
| `iteration_00_heldout_00` | **保留闭环** | **+0.0429** | **−0.437** | 0.72 |

`heldout_00` 未用于 FD 步长、也未用于 capture/operating-point 选择。所有 18 个
source 共用同一套 payload；source-disjoint 按原始 xAOD。

## 生产

- 源：原 V3 10 train / 8 validation xAOD。test 未打开。
- bank：`outputs/mc24_ift_layer_contrast_2d_iteration00_trainval_physical_v1/`
  （10 点 × 18 源 = 180 次真实 `/Tracker/Align → SegmentFitRefit → SegmentsRefit → NtupleDumper → Acts(mode 0)`）
- Condor **cluster 999274**（bigbird24，flavour `tomorrow`，6000 MB），
  18 job 已提交。
- 每点 payload 由 contrast 坐标展开，station 六矢为 0。

回归：`tests/test_layer_contrast_2d.py` 与既有 layer/gauge/5-DoF/route-selected
layer 测试共 43 项通过。

## 生产完成

cluster **999274** 18/18 `completion_status=accepted`，零 `failure.json`，mode-0，
test 未打开。180 个转换 ROOT 中 **1** 个存储层截断：
`mc24_100043_00500_00599/iteration_00_start/tracklets.root`
（`TBasket` 空偏移表）。Athena `enhanced_tracklets.root` 完好（100 event / 375
tracklet）。本地重跑 `convert_ntuple_tracklets.py --include-truth` 与
`audit_tracklets.py --require-mc-labels`；新旧 audit 除时间戳外逐字段一致。
坏文件保留为 `tracklets.root.corrupt_20260820`。未重提 Condor。

`build_6dof_pilot_physical_corpus.py` 现把 `injected_layer_transforms` 交给完成审计
（station 仍全零、L1 仍全零）。语料 18 源 × 10 点。

## Truth-selected 2-D（validation 只评价）

同一套 bank 自测的 contrast FD；锚点 `iteration_00_reference`。station 六矢保持 0，
未解冻结 station 的 `reference_layer`。

主起点 `iteration_00_start`（注入 `C_dx=+0.0403` mm，`C_rx=+0.389` mrad）：

| | train | validation |
| --- | --- | --- |
| 边数 | 2352 | 1801 |
| 秩 | 2/2 | 2/2 |
| 条件数 | **926** | 35 |
| 信息余弦 / 后验相关 | **−0.969** | −0.654 |
| `near_degenerate`（阈值 0.9） | **true** | false |
| 回收 `C_dx` / mm | 0.0385 | 0.0397 |
| 回收 `C_rx` / mrad | 0.311 | 0.376 |
| 误差 | −0.0019 mm / −0.078 mrad | −0.00065 mm / −0.014 mrad |
| pull_fit | −2.67 / −2.55 | −1.10 / −0.90 |
| pull_registered | −0.27 / −0.54 | −0.09 / −0.09 |
| source spread | 0.017 mm / **0.729 mrad** | 0.009 mm / 0.391 mrad |
| post/pre RMS | 0.079 | **0.719** |
| dual / framework capture | true | true |

弱 SVD 方向几乎是 `C_rx`（组成 0.99），与预飞行监视项 0.888 同方向，train 上越过
0.9。**逐源 `C_rx` 不稳定**：`mc24_100043_00600_00699` 回收 −0.339 mrad、
`mc24_100044_00500_00599` 回收 −0.208 mrad（期望 +0.389，符号反了）；这两源的
`C_dx` 也偏低（0.024 / 0.027 vs 0.040）。其余 train 源接近注入。这是联合 2 列把
弱 mode 吃进噪声，不是 1-D `C_rx` 失效。

validation 的 3-DoF 已知离群源 `mc24_100047_00150_00199` **未**再以错误 0→1 边出现；
该源 start 回收 `C_rx=0.359` mrad。真正的 validation 离群是
`mc24_100047_00100_00149`（`C_rx≈0`）。未调 association。

保留点 `iteration_00_heldout_00`（注入 `C_dx=+0.0429` mm，`C_rx=−0.438` mrad；
未用于 FD/capture）：

| | train | validation |
| --- | --- | --- |
| 信息余弦 | **−0.969** | −0.654 |
| 回收 | 0.0411 mm / −0.506 mrad | 0.0430 mm / −0.438 mrad |
| 误差 | −0.0018 mm / −0.069 mrad | +0.00008 mm / −0.00023 mrad |
| source spread `C_rx` | **0.667 mrad** | 0.155 mrad |
| post/pre RMS | 0.048 | **0.549** |
| dual capture | true | true |

同一对 train 离群源 held-out 上把 `C_rx` 收到 −0.911 / −1.080 mrad（期望 −0.438）。
validation 残差只降到 55–72%，尽管 pooled 点估计可以过 dual gate。

产物：
`outputs/mc24_ift_layer_contrast_2d_truth_{train,validation}_{start,heldout}_v1/`。

## 冻结 V2 route-selected（unknown association）

ungated 冻结 V2；`anchor_selected_field_edge` + `physical_edge_deduplicated`；
只浮动 `C_dx C_rx`，**无** `--gauge`。锚点 association 只跑
`iteration_00_reference`；target/FD 读 pooled synthetic sample。

Association（MC audit，未调阈）：

| split | routes | 完整四站 | complete-track ε | purity | fake |
| --- | --- | --- | --- | --- | --- |
| train | 2823 | 2187 | 0.906 | 0.978 | 0.022 |
| validation | 2066 | 1474 | 0.780 | 0.978 | 0.023 |

主起点与保留点（同一 Jacobian，只换观测）：

| | train start | train held-out | val start | val held-out |
| --- | --- | --- | --- | --- |
| unique edges（replica） | 656（2466） | 656（2466） | 490（1890） | 486（1871） |
| 秩 / 条件数 | 2 / **938** | 2 / 938 | 2 / 34.5 | 2 / 34.5 |
| 后验相关 | **+0.969** | **+0.969** | +0.479 | +0.479 |
| `near_degenerate` | **true** | **true** | false | false |
| 回收 vs 注入 | 0.04027 / 0.38970 vs 0.04033 / 0.38945 | 0.04271 / −0.43806 vs 0.04289 / −0.43751 | 点估计几乎贴注入 | 点估计几乎贴注入 |
| pull_fit | −0.03 / +0.003 | −0.11 / −0.008 | ~0 | ~0 |
| post/pre RMS | **0.010** | **0.007** | **0.969** | **0.922** |
| dual / framework | true | true | true | true |
| hierarchy residual 门（post≤0.5 pre） | true | true | **false** | **false** |
| station / L1 / `L0=+C,L2=-C` | 全通过 | 全通过 | 全通过 | 全通过 |

train 上 pooled 点估计极准、pull 稳定，但 **2 参数后验相关 0.969**，与 truth-selected
同一简并。validation 上相关降到 0.48，点估计仍然贴注入，**残差几乎不降**：二维
Jacobian 不能解释独立源上的联合响应。这不是阈值/route policy 问题，不重调 V2。

stdout 摘要曾按 1-D `outer_relative` 字段打印而在 JSON 写完后崩溃；已改为
contrast 审计打印相关与 post/pre。`tests/test_layer_contrast_2d.py` 等 30 项通过。

产物：
`outputs/mc24_ift_layer_contrast_2d_{synthetic_trainval,v2_backbone_{train,validation},route_selected_{train,validation}_{start,heldout}}_v1/`。

## Held-out 闭环（associate → estimate；不启 remaining 生产）

冻结 V2 对 `iteration_00_heldout_00` 的 remaining 是
`C_dx≈0.00019` mm、`C_rx≈0.00056` mrad（train）以及更小的 validation 残差，
已在预注册 3σ 窗口内、接近名义点。按冻结规则二维子空间**不予准入**，因此
**不**再为 remaining≈0 提交新的 18 源真实 refit。未打开 test，未写入相对 ry /
layer dy/rz / layer1 / module。

## 判定：停在两个 1-DoF，不冻结二维子空间

预注册 dual gate 的 pooled 点估计可以过，但准入要求的是**联合扰动下两个已闭合
mode 仍解耦**。本条目否决原因：

1. **联合简并**：train truth-selected 与 route-selected 信息/后验相关 **0.969 ≥ 0.9**，
   弱方向是 `C_rx`。预飞行 0.888 的监视项在 10 源联合 bank 上成真。
2. **source-dependent `C_rx`**：至少两个 train 源在 truth-selected 上把 `C_rx` 收到
   反号或加倍；validation 另有 `100047_00100` 把 start 的 `C_rx` 收到 0。
3. **validation 残差不降**：truth-selected post/pre 0.55–0.72，route-selected
   0.92–0.97。独立源上二维线性模型不能同时解释两个 contrast。

**因此不冻结 `C_dx+C_rx` 联合子空间。** hierarchy 仍停在已经分别闭合的
**1-D `C_dx`（条目 39/40）与 1-D `C_rx`（条目 41）**。不增加 layer 参数、
不把 relative ry 拉进来补偿二维失败。下一阶段若再碰内部转动，只允许单独的
relative-ry 候选，不得与本次联合 2 列混拟合。

station 5-DoF、mode-0、V2、route policy、`physical_edge_deduplicated`、sealed
test 全部保持冻结。
