# 2026-08-19 (39) 干净线性 IFT layer internal-deformation held-out 闭合

## 任务

条目 38 的 identifiability map 已经说明：原 held-out 含非线性 layer dy，
不能用来回答「真实重建能否恢复 IFT 内部相对变形」。本条目**只做**一个
干净、最小的线性闭合，回答唯一问题：

> 在不借助 station rigid motion、也不受 dy/rz 非线性污染时，真实 FASER
> reconstruction 是否能稳定恢复一个 gauge-defined IFT 内部相对变形？

明确不做：完整 10+8 layer curriculum、module 层级、重产现有 FD Jacobian、
本步长下的 layer dy/rz。station-level 5-DoF、mode-0、V2、route policy、
`physical_edge_deduplicated` 继续冻结。

## 注入（真实链，3 个 train source）

与 identifiability 相同的三个 source：
`mc24_100043_00200_00299`、`mc24_100043_00600_00699`、
`mc24_100044_00300_00399`。每源只跑新点：

| 点 | station | layer0 | layer1 | layer2 |
| --- | --- | --- | --- | --- |
| `iteration_00_closure_relative_dx` | 全零 | dx = +0.12 mm | 0 | dx = −0.12 mm |
| `iteration_00_closure_relative_ry` | 全零 | ry = +0.7 mrad | 0 | ry = −0.7 mrad |

两个点独立，不把 dx 与转动混在同一 payload。不注入 source-unstable 的
layer1 rx，不注入 dy/rz。链仍是
`/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit →
SegmentsRefit → NtupleDumper → Acts(mode 0)`。

Jacobian **复用**条目 37/38 已通过线性筛的 layer dx/rx/ry FD
（`outputs/mc24_ift_layer_identifiability_pilot_v1/`），不重跑 40 个 probe。

## 通过标准

Truth-selected、layer-only，两种 gauge（`sum_to_zero`、`reference_layer`）
各自求解后，统一转换成 `layer_i - weighted_mean(layer)` 再比较。看的是
**物理相对 deformation**（外层 `layer0 − layer2`）、源间稳定、rank/条件数、
post-fit residual，以及 station 槽是否被拉动；**不**比较不同 gauge 下参数
标签是否逐项相等。reference_layer 展开后的 layer 加权均值是 gauge 坐标，
不是 station common-mode 泄漏。

## 生产

- 模板：`configs/physical_refit_ift_layer_linear_internal_heldout.yaml`
  （`held_out_only: true`）
- 观测 bank：`outputs/mc24_ift_layer_linear_internal_heldout_v1/`
  （2 点/源，`q_over_p_mode: 0`，`test_data_accessed: false`）
- Jacobian bank：`outputs/mc24_ift_layer_identifiability_pilot_v1/`
- Condor **998766** 启动失败（held-out-only 无零点 baseline）；修复后
  cluster **998796** 完成：3/3 source、6/6 点、0 个 `failure.json`。
- 闭合报告：`outputs/mc24_ift_layer_linear_internal_heldout_v1_truth_closure_v2/`

## Truth-selected 结论

Pooled 3 source，layer-only 2 维 gauge，695 条 IFT 边。

### 相对 dx：通过

注入 `[+0.12, 0, −0.12]` mm。两种 gauge 的 gauge-invariant internals：

| | layer0 | layer1 | layer2 | 外层相对 `0−2` |
| --- | --- | --- | --- | --- |
| 期望 | +0.12 | 0 | −0.12 | 0.240 |
| sum_to_zero | +0.116 | +0.008 | −0.124 | 0.241 |
| reference_layer | +0.123 | −0.004 | −0.119 | 0.242 |

- 满秩 2/2；条件数 **37**（sum_to_zero）与 **16**（reference_layer）。
- post/pre residual RMS ≈ **0.035**。
- 外层相对逐源 0.230 / 0.251 / 0.240 mm（散布 0.021 mm）。`00600` 把一部分
  幅度放进弱约束的 layer1，但可辨识模式（外层反对称）稳定。
- layer-only 拟合中 `station_common ≡ 0`。station-only 替代要把 stereo 放大
  吃进 `ift_dx ≈ −7.5 mm`，且 post RMS 0.60 对 layer-only 0.22，**不是**
  0.12 mm 相对变形漏进 station 刚体。

**相对 dx 作为 hierarchy curriculum 第一项正式准入自由度。**

### 相对 ry：不准入（后续候选）

sum_to_zero 回收 `[+0.708, −0.005, −0.702]` mrad，几乎完美；reference_layer
给出 `[+0.443, +0.087, −0.531]`，gauge-invariant internals **不一致**
（外层相对 0.97 vs 1.41 mrad），post/pre 0.36 vs 0.009。与条目 38「外层
ry 勉强可辨识」一致。station-only 的 `ift_ry` 仅 0.027 mrad，不是 station
泄漏，而是 reference_layer 对 0.7 mrad 注入偏软。layer1 rx 仍不注入。

### layer dy/rz

继续冻结。若以后重开 dy，只能把 FD 从 0.2 mm 显著缩小后重做
occupancy/odd-linearity scan，不能沿用本轮中毒 Jacobian。

## 对那个问题的回答

**能。** 在冻结 station 5-DoF、不含 dy/rz 的真实 reconstruction 上，两种
显式 gauge 对 **IFT 外层相对 dx** 给出同一物理解，源间稳定、满秩、残差下降
约 30 倍，且没有把该变形写进 station 刚体槽。

## Route-selected

相对 dx 已通过，因此做冻结 V2、`anchor_selected_field_edge` +
`physical_edge_deduplicated` 的 unknown-association 闭合；station 解保持
固定。语料适配：`scripts/build_layer_internal_physical_corpus.py`（Jacobian
layer-dx FD + 本条目 held-out，不重跑 refit）。

- 合成语料：`outputs/mc24_ift_layer_relative_dx_synthetic_train_v1/`（8 个 train sample）
- 冻结 V2 backbone（仅 anchor）：`outputs/mc24_ift_layer_relative_dx_v2_backbone_train_v1/iteration_00_reference`
  （859 routes，683 条完整四站，2343 条 selected field-aware edges）
- Update 脚本扩展：`scripts/run_route_selected_multidof_update.py` 接受
  `ift_layer_hierarchy`、`--only-parameters`、`--gauge`、`--target-scan-root`、
  `--allow-nominal-anchor`；layer 条件仍走 `payload_transforms_with_parameter_values`，
  不写进 station 槽。比较的是 `layer_i − mean` internals，不是 gauge 标签。

Pooled train，193 条去重物理边，mode-0。注入仍是 `[+0.12, 0, −0.12]` mm，
外层相对 0.240 mm。

| | layer0 | layer1 | layer2 | 外层相对 `0−2` | 条件数 | post/pre RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 期望 | +0.12 | 0 | −0.12 | 0.240 | — | — |
| sum_to_zero | +0.120 | −0.000 | −0.120 | 0.240 | 54 | 0.005 |
| reference_layer | +0.118 | +0.001 | −0.118 | 0.236 | 28 | 0.005 |

两种 gauge 满秩 2/2；外层相对互差 0.004 mm；station 六矢保持全零。
reference_layer 展开后的 layer 加权均值（−0.118 mm）是 gauge 坐标，不是
station 泄漏。官方报告：

- `outputs/mc24_ift_layer_relative_dx_route_selected_v1/`（`sum_to_zero`）
- `outputs/mc24_ift_layer_relative_dx_route_selected_reference_layer_v1/`

**冻结 V2 route-selected 也对相对 dx 通过。** 不因此启动完整 10+8 layer
curriculum；外层相对 rx/ry 仍为后续候选。layer dy/rz 继续冻结。
