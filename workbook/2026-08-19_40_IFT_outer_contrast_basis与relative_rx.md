# 2026-08-19 (40) 外层 contrast basis，以及 relative rx 的最小闭合

> **条目 41 更正：** 冻结 station 的 `reference_layer` 不是 gauge
> 交叉检验。条目 40 用它否定 `C_rx` 的结论作废；见
> `workbook/2026-08-20_41_gauge_equivalence与C_rx准入.md`。

## 任务

条目 39 已证明 IFT 外层相对 dx 在 truth-selected 与冻结 V2 route-selected
上都稳定。本条目**不**扩成完整 10+8 layer curriculum，而是把已准入内部
变形收成显式的 gauge-invariant contrast 坐标，并对第二个候选自由度
**relative rx** 做最小独立闭合。

拟合坐标不再用三层各自独立参数。定义

* `C_dx = (dx_L0 − dx_L2) / 2`
* `C_rx = (rx_L0 − rx_L2) / 2`

layer common mode 固定为 0，layer1 暂不浮动，station 5-DoF / mode-0 / V2 /
route policy / `physical_edge_deduplicated` / sealed test 全部冻结。`C_dx`
只作为已准入 baseline 复核，不重调。relative ry 仍不准入。

比较量统一为 gauge-invariant `rx_L0 − rx_L2`（以及 dx 的对应外层差）。
三种表示：`sum_to_zero`、`reference_layer`、`outer_contrast`。

## 生产

- 模板：`configs/physical_refit_ift_layer_linear_internal_heldout_relative_rx.yaml`
- 观测 bank：`outputs/mc24_ift_layer_linear_internal_heldout_relative_rx_v1/`
- Jacobian：复用 `outputs/mc24_ift_layer_identifiability_pilot_v1/` 的 layer-rx FD
- 三个 train source，每源只跑一点
  `iteration_00_closure_relative_rx`：layer0 rx = +0.7 mrad，layer1 = 0，
  layer2 rx = −0.7 mrad。不混 dx/ry/dy/rz。
- Condor **998855**（bigbird24，`testmatch`）：3/3 source、1/1 点、
  0 个 `failure.json`。payload L2 键 `00`/`02` 为 ±0.0007 rad，station 六矢全零。

## C_dx 冻结复核（不重跑 held-out）

在条目 39 的相对 dx bank 上增加 `outer_contrast` 后重解，
`outputs/mc24_ift_layer_linear_internal_heldout_v1_truth_closure_contrast_baseline/`：

| 表示 | 外层相对 `0−2` | 条件数 | post/pre RMS |
| --- | --- | --- | --- |
| 期望 | 0.240 mm | — | — |
| sum_to_zero | 0.241 | 37 | 0.037 |
| reference_layer | 0.242 | 16 | 0.035 |
| outer_contrast `C_dx=0.121` | 0.243 | 1 | 0.031 |

三种表示一致，layer1 在 contrast 下被固定为 0。**不重调 C_dx。**

## Relative rx：truth-selected

Pooled 3 source，692 条 IFT 边。注入外层相对 1.400 mrad。

| 表示 | layer0 | layer1 | layer2 | 外层 `0−2` | 条件数 | post/pre |
| --- | --- | --- | --- | --- | --- | --- |
| 期望 | +0.70 | 0 | −0.70 | 1.400 | — | — |
| sum_to_zero | +0.704 | +0.006 | −0.710 | **1.414** | 2.3 | **0.018** |
| outer_contrast `C_rx=0.702` | +0.702 | 0 | −0.702 | **1.404** | 1 | **0.015** |
| reference_layer | +0.455 | −0.025 | −0.431 | **0.886** | 4.4 | **0.634** |

- `sum_to_zero` 与 `outer_contrast` 给出同一物理解；逐源外层相对
  1.379 / 1.435 / 1.404 mrad（contrast：1.388 / 1.434 / 1.393），方向一致。
- `reference_layer` 把幅度吃进 layer 均值（−0.455 mrad）并软化外层相对到
  0.89 mrad，残差几乎不降。这与条目 39 的 relative ry 同类：该 gauge 对
  转动内部模参数化敏感，不是 station 泄漏（`station_common` 全零，
  station-only `ift_rx ≈ 0.057` mrad）。
- 满秩；contrast / sum_to_zero 条件数稳定。三种表示**没有**给出一致物理解，
  因此 **relative rx 不准入** 二维 mini-curriculum。

官方报告：
`outputs/mc24_ift_layer_linear_internal_heldout_relative_rx_v1_truth_closure/`

## 因此不做的事

- 不跑冻结 V2 route-selected（三条表示未同时通过）。
- 不启动 `C_dx + C_rx` 的 10/8 source-disjoint mini-curriculum。
- 不把 relative ry 当自由度硬救。
- layer dy/rz 与 module-level 继续冻结。

## 对那个问题的回答

**Contrast basis 本身成立。** 已准入的外层相对 dx 用 `C_dx` 直接拟合，与两种
旧 gauge 同解。relative rx 在 `C_rx` / `sum_to_zero` 下同样能回收 1.4 mrad
外层差，但 `reference_layer` 仍给出约 0.89 mrad 的不同物理解，所以它目前
仍处于参数化敏感区，不能作为 curriculum 第二自由度。若以后只把
`outer_contrast`（外加 `sum_to_zero` 作对照）当作正式拟合坐标、不再要求
`reference_layer` 对转动模给出同一标签展开，再单独决定是否重开 rx 的
unknown-association 闭合。
