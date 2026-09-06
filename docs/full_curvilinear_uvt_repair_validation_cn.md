# 完整 CurvilinearUVT 分支协方差契约修复验证 V1

Workbook 86。**状态：已完成并冻结。**
**最终决策：`segmentfit_full_curvilinear_covariance_contract_validated`。**

本战役检验 WB85 剩余结构 `mean_cyy_inflation_on_large_angle_subset` 是否就是
`SegmentFitAlg::GetState` 缺失的一般分支（`|t·z|<0.99`）Athena
`CurvilinearUVT` 槽位映射。它在**同一** source-disjoint
construction/validation 划分上，用**同一** gate 重跑冻结的 WB83 Stage A，并比较
baseline / WB85 束流近似修复 / WB86 完整 CurvilinearUVT 修复。这**不是**
alignment 任务。

**继承的冻结状态（SHA 校验，不重开）：**

- WB81 `faseracts_propagated_covariance_not_calibrated` /
  `overestimated_transported_fit_covariance`
- WB82 `existing_mc_real_wide_ty_support_validated`（`floor_muon_100120`）
- WB83 `source_tracklet_fit_covariance_not_calibratable` /
  `position_xy_swap_with_slope_miscalibration`
- WB84 `deterministic_segmentfit_get_state_transform_bug` /
  `coordinate_convention_mismatch_and_jacobian_sign_error`
- WB85 `segmentfit_getstate_repair_insufficient` /
  剩余结构 `mean_cyy_inflation_on_large_angle_subset`

`geometry_write_allowed=false`，`real_data_alignment_authorized=false`，
`measurement_model_validated=false`。不读真实残差，不写 geometry/conditions，
不进入 FaserActs / Stage B，不打开 Frozen-V2。不加 scale、不调 chi2、不删
outlier、不按角度拒绝。

## 修复范围（封闭）

从 Athena `CurvilinearUVT` 定义在**两个**分支上重建 GetState 映射：

1. 完整槽位映射：由 `(curvU, curvV)` 解
   `loc1·curvU_xy + loc2·curvV_xy = (x, y)`。`|t·z|≥0.99` 时这只是近似
   `loc1≈−y`、`loc2≈+x`（`O(tx)` 修正）。`|t·z|<0.99` 时
   `curvU=−(t×ẑ)` **不是** −ŷ。
2. Jacobian 符号：`dφ/dtx = −ty/r²`（与 WB85 相同）。

禁止：把硬编码束流映射当作唯一修复、scale factor、chi2 调参、outlier /
角度拒绝、基于残差的修正、station 相关标定、改中心预测。

修复离线实施为
`C_rep = J_full J_buggy⁻¹ C_exp J_buggy⁻ᵀ J_fullᵀ`。拟合态
`[x,y,tx,ty]` 不变。

## Gate（与 WB83 完全相同，未改）

`chi2/ndof ≤ 4`，`Cov(z)` 特征值 ∈ [0.25, 4]，广义特征值 ∈ [0.25, 4]，
construction/validation 源不相交一致性。战役 gate 评在**完整** station
总体。`|t·z|` 子集单独报告；3 条大角度径迹**不删除**。

## 结果

3230 条 tracklet（3227 束流，3 大角度），两种修复均 0 次反演失败。

### 全样本（WB83 gate）

| cell | baseline med χ² | WB85 χ² / med / 校准 | WB86 χ² / med / 校准 |
| --- | --- | --- | --- |
| construction 0 | 409 | 1.00 / 0.838 / 是 | 0.999 / 0.833 / **是** |
| construction 1 | 397 | 3.68 / 0.851 / **否** | 1.02 / 0.851 / **是** |
| construction 2 | 415 | 0.987 / 0.849 / 是 | 0.987 / 0.849 / **是** |
| construction 3 | 377 | 0.973 / 0.824 / 是 | 0.973 / 0.824 / **是** |
| validation 0 | 391 | 1.11 / 0.826 / **否** | 0.988 / 0.826 / **是** |
| validation 1 | 360 | 0.932 / 0.784 / 是 | 0.932 / 0.783 / **是** |
| validation 2 | 361 | 0.910 / 0.820 / 是 | 0.909 / 0.820 / **是** |
| validation 3 | 437 | 1.02 / 0.844 / 是 | 1.02 / 0.844 / **是** |

Baseline 仍复现 WB83（8/8 失败，median χ² ~360–437）。WB85 仍在同样 2 个
cell 上失败（均值协方差本征 gate）。WB86 在两个不相交 split 上 **8/8 通过**。

先前失败的两个 cell 恢复，是因为 MEAN `C_yy` 不再被抬高：

- construction station 1：`Cov(z)` 最大 11.6 → 1.21；广义特征值最小 0.206 → 0.828
- validation station 0：广义特征值最小 0.172 → 0.833

典型 tracklet median χ² 仍约 0.83。`frac(χ²>4)` 仍约 0.01–0.03。

### 束流子集（`|t·z|≥0.99`）

8 个仅束流 cell 在 WB85 与 WB86 下全部通过。凡 `n_large=0` 的 station，两种
修复在报告精度内一致。WB85 已经是有效的束流分支契约。

### 大角度子集（`|t·z|<0.99`）

共 3 条（construction station 0、1；validation station 0）。`n=1` 低于冻结的
`min_tracklets=30`，因此**不对**该子集套用 WB83 总体 gate
（`wb83_population_gates_applicable=false`）。逐径迹诊断（不是拒绝切割）：

| 径迹 | `|t·z|` | `C_yy` WB85 → WB86 | χ² WB85 → WB86 |
| --- | --- | --- | --- |
| construction 0 | 0.9820 | 3.28×10⁻³ → 7.97×10⁻⁵ | 9.54 → 2.26 |
| construction 1 | 0.9887 | 0.182 → 8.29×10⁻⁵ | 4190 → 3.64 |
| validation 0 | 0.9890 | 0.150 → 1.03×10⁻⁴ | 227 → 3.17 |

construction station 1 上一般分支 `curvU=−(t×ẑ)` 主要沿 −x̂，硬编码束流映射
`loc1=−y` 会重新引入类似交换的 `C_yy` 膨胀。这一条径迹就足以造成 WB85 的
MEAN `C_yy` 失败。WB86 把 `C_yy` 恢复到已校准的 ~10⁻⁴ mm²。未使用 scale、
outlier 切割或角度拒绝。

## 决策

`segmentfit_full_curvilinear_covariance_contract_validated`。

在完整 Athena Curvilinear 坐标契约（两个 `|t·z|` 分支）下，SegmentFit 导出
协方差满足源级协方差闭合。这**不是** alignment 变好。

`measurement_model_v2_discussion_allowed=true`。
`propagated_covariance_validation_authorized=true`。

此处仍然冻结：

- `measurement_model_validated=false`
- `real_data_alignment_authorized=false`
- `geometry_write_allowed=false`
- `stage_b_entered=false`
- `faseracts_propagation_entered=false`
- `frozen_v2_alignment_authorized=false`

后续战役现在可以**讨论 / 进入** FaserActs 传播协方差验证（Stage B）、
Measurement Model V2，或 conditional Jacobian 战役。本战役不进入它们。
