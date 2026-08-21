# 2026-08-20 (41) Gauge-equivalence 语义审计，以及 C_rx 准入

## 任务

条目 40 用冻结 station 的 `reference_layer`（外层 0.886 mrad、post/pre
0.634）把 relative rx 判死。本条目**先**做 payload transform 层的
gauge-equivalence 语义审计，**不**把 `C_rx` 判死，也**不**把
`C_dx/ry/dy/rz` 混进同一个 route-selected。问题是：在 station 六矢被
严格冻在 0 时，`sum_to_zero` / `outer_contrast` 与 `reference_layer`
是不是同一物理子空间的不同坐标？

## 解析映射

同一物理 outer-contrast 变形（station = 0）

\[
L = [+C,\ 0,\ -C],\qquad S = 0.
\]

等权 `sum_to_zero` 就是这个向量（均值 0）。要在 **layer0 = 0** 的
reference-layer 图上表示**同一套加法六矢**，必须把所有平面校正整体
平移一个 common rotation，并由 station 刚体提供等量补偿：

\[
L' = L - C = [0,\ -C,\ -2C],\qquad S' = +C.
\]

加法读出 `S + L_i` 与 `S' + L'_i` 逐层相等。把 station 冻在 0 而只
展开 `L'`，得到的是 `[0,\ -C,\ -2C]`，**不是**注入的
`[+C,\ 0,\ -C]`。因此冻结 station 的 `reference_layer` 拟合的是另一族
物理约束，它的 0.886 mrad 不能当作 `C_rx` 不可辨识的证据。

## 数值：detector-element transforms

合成约定与 `TrackerAlignDBTool` 一致：station L1 为全局
`T·Rz·Ry·Rx`（不额外共轭）；plane L2 先写成同一六矢再共轭到平面
全局 z，`Ad_{T(z)}(g_\text{layer})`；GeoModel 把全局 delta 以
`T_\text{new} = g·T_\text{nominal}` 施加，故

\[
g_\text{element} = g_\text{station}\cdot \mathrm{Ad}_{T(z_\text{plane})}(g_\text{layer}).
\]

平面 z 取 relative-rx held-out 的 station-0 tracklet z
（−1860.15 mm）加上 FASERNU `LAYERPITCH = 31.5` mm：
`[-1891.65, -1860.15, -1828.65]` mm。

| 比较 | dx（C=0.12 mm） | rx（C=0.7 mrad） |
| --- | --- | --- |
| 加法六矢：contrast vs 带 station 补偿的 RL | **逐位相等** | **逐位相等** |
| 加法六矢：contrast vs 冻结 station 的 RL | 不相等 | 不相等 |
| Calypso `g_element`：带 station 补偿 | **逐位相等** | **不相等**（layer0 原点像差 Δy ≈ **1.32 mm**） |
| Calypso `g_element`：冻结 station | 不相等 | 不相等 |

`dx` 平移与 z 共轭对易，所以正确补偿后 SE(3) 也等价。`rx` 即使加上
`S'=+C` 也不等价：station `rx` 绕全局原点转，layer `rx` 绕平面原点转，
杠杆臂 `C·z ≈ 0.7×10^{-3}×1892 ≈ 1.32` mm。这是 pivot/conjugation，不是
再拿冻结 `reference_layer` 去否定 `C_rx`。

官方审计：
`outputs/mc24_ift_layer_gauge_equivalence_audit_v1/gauge_equivalence_audit.json`
回归：`tests/test_gauge_equivalence.py`。

## 文档与判据更正

- 规范内部基定为零公共模 contrast：`C_dx=(dx_L0-dx_L2)/2`（已冻结）、
  `C_rx=(rx_L0-rx_L2)/2`，layer1=0，station 刚体保持冻结。
- 等权 `sum_to_zero` 是同一物理族的对照。
- 冻结 station 的 `reference_layer` **降级**为不同物理约束的负对照，
  不再进入 `gauges_agree` / residual / 准入。
- 闭合 schema `faser-ift-layer-linear-internal-closure-v3`。

用新判据重解条目 40 的同一 bank（不重跑 refit）：
`outputs/mc24_ift_layer_linear_internal_heldout_relative_rx_v1_truth_closure_v3/`
→ `rx_passed=True`，692 条 IFT 边。负对照仍报告外层 0.886 mrad、
post/pre 0.634。

## C_rx-only 冻结 V2 route-selected

Jacobian：identifiability 的 layer-rx FD。观测：relative-rx held-out。
`--gauge outer_contrast`，只浮动三层 `rx`，不混 `C_dx/ry/dy/rz`。

- 语料：`outputs/mc24_ift_layer_linear_internal_heldout_relative_rx_v1/physical_corpus_manifest_relative_rx.json`
- 合成：`outputs/mc24_ift_layer_relative_rx_synthetic_train_v1/`（8 个 train sample）
- 冻结 V2 仅 anchor：`outputs/mc24_ift_layer_relative_rx_v2_backbone_train_v1/iteration_00_reference`
  （859 routes，683 条完整四站，2343 条 selected field-aware edges）
- Update：`outputs/mc24_ift_layer_relative_rx_route_selected_contrast_v1/`

| | `C_rx` | 外层 `0−2` | 秩 | 条件数 | post/pre | station |
| --- | --- | --- | --- | --- | --- | --- |
| 期望 | 0.700 | 1.400 | — | — | — | 0 |
| truth-selected contrast | 0.702 | **1.404** | 1 | 1 | 0.015 | 0 |
| route-selected contrast | 0.700 | **1.399** | 1 | 1 | **0.005** | **全零** |

195 条 `physical_edge_deduplicated` 边；容差 0.20 mrad。**通过。**

## 因此准入 / 仍不做

- **`C_rx` 与已冻结的 `C_dx` 一起成为 hierarchy 的两个正式内部自由度。**
- 第一份 `C_dx+C_rx` 二维 source-disjoint mini-curriculum（10 train /
  8 val，幅度落在已有 FD 线性包络，associate → layer-align → payload →
  真实 refit → reassociate）**现在允许，本条目不启动**。
- 不混入 relative ry、layer dy/rz、module。
- 不重开 40 个 FD probe，不打开 sealed test，不重调 mode-0 / V2 /
  route policy / station 5-DoF。
- 条目 40 用冻结 `reference_layer` 否定 `C_rx` 的结论作废。
