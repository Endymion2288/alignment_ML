# Workbook 86: 完整 CurvilinearUVT 分支协方差契约修复验证 V1

日期：2026-09-06
状态：**完成并冻结** —— 单一交互会话内按冻结顺序执行；gate 在 confirmatory 计算前写入 config，执行后未修改。

**最终决策：`segmentfit_full_curvilinear_covariance_contract_validated`**

> **本 Workbook 86 是 WB85 剩余结构的完整 CurvilinearUVT 验证，不是 alignment。** 不重开 WB81–WB85。不读 real residual，不写 geometry/conditions，不进入 FaserActs / Stage B，不打开 Frozen-V2。

整个 Workbook 86：

- `held_out_accessed = false`
- `real_data_alignment_authorized = false`
- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `measurement_model_validated = false`
- `measurement_model_v2_discussion_allowed = true`（仅授权后续讨论，本战役不进入）
- `propagated_covariance_validation_authorized = true`（仅授权后续战役，`stage_b_entered = false`）

## 1. 科学问题（唯一）

**WB85 的确定性修复是否只覆盖了束流近似 `loc1≈−y`、`loc2≈+x`，而没有完整实现 Athena `CurvilinearUVT` 在所有径迹方向下的基变换？**

成功标准不是 alignment 变好，而是证明 SegmentFit 导出协方差在完整 Athena Curvilinear 坐标契约下满足源级协方差闭合。

## 2. 起始状态审计（已完成）

`load_config` 对 WB81–WB85 做 SHA 校验（config + 决策产物 + 冻结决策/机制）。WB85 的剩余结构写在 `remaining_structure.category`，不是 `failure_classification`：

- WB81：`faseracts_propagated_covariance_not_calibrated` / `overestimated_transported_fit_covariance`
- WB82：`existing_mc_real_wide_ty_support_validated`（`floor_muon_100120`）
- WB83：`source_tracklet_fit_covariance_not_calibratable` / `position_xy_swap_with_slope_miscalibration`
- WB84：`deterministic_segmentfit_get_state_transform_bug` / `coordinate_convention_mismatch_and_jacobian_sign_error`
- WB85：`segmentfit_getstate_repair_insufficient` / `mean_cyy_inflation_on_large_angle_subset`

Stage A 数据与 gate 与冻结 WB83 **逐字相同**（construction：100043×3+100044×2；validation：100047×2+100048×2；station 0–3；`chi2/ndof≤4`；`Cov(z)` 与广义特征值 ∈ [0.25,4]）。篡改 gate 或继承 SHA 会被 `ConfigError` 拒绝。

Config SHA：`04c563cad61485ce73746b50d201353841d99cd087a41f5d830525b6d33cc47b`

## 3. 修复范围（封闭）

只允许：

1. 完整 CurvilinearUVT 槽位映射（两个 `|t·z|` 分支）：解 `loc1·curvU_xy + loc2·curvV_xy = (x, y)`
2. Jacobian 符号：`dφ/dtx = −ty/r²`（与 WB85 相同）

禁止：scale factor、chi2 调参、outlier 删除、按角度拒绝、残差修正、station 相关标定、改中心预测、把硬编码 `loc1=−y, loc2=+x` 当作唯一修复。

实施：离线确定性变换 `C_rep = J_full J_buggy⁻¹ C_exp J_buggy⁻ᵀ J_fullᵀ`。`e_source` 使用未改的拟合态。三路比较：original GetState / WB85 束流近似 / WB86 完整 UVT。

## 4. 结果

3230 条 tracklet（3227 束流，3 大角度），0 次反演失败。

### 4.1 全样本三路（`source_covariance_closure_three_way.json`）

Baseline 复现 WB83：8/8 cell 失败，median chi2/ndof ~ 360–437。

WB85 束流近似：仍 6/8 通过；失败 cell 仍是 construction station 1（χ²=3.68，`Cov(z)` max=11.6，gen min=0.206）与 validation station 0（χ²=1.11，gen min=0.172）。

WB86 完整 UVT：

| split | station | chi2/ndof | median | frac>4 | Cov(z) | gen eig | calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| construction | 0 | 0.999 | 0.833 | 0.026 | [0.873, 1.16] | [0.812, 1.02] | 是 |
| construction | 1 | 1.02 | 0.851 | 0.023 | [0.905, 1.21] | [0.828, 1.02] | **是** |
| construction | 2 | 0.987 | 0.849 | 0.015 | [0.766, 1.16] | [0.713, 1.19] | 是 |
| construction | 3 | 0.973 | 0.824 | 0.022 | [0.829, 1.15] | [0.762, 1.01] | 是 |
| validation | 0 | 0.988 | 0.826 | 0.028 | [0.826, 1.15] | [0.833, 1.11] | **是** |
| validation | 1 | 0.932 | 0.783 | 0.022 | [0.787, 1.06] | [0.813, 1.05] | 是 |
| validation | 2 | 0.909 | 0.820 | 0.008 | [0.786, 1.07] | [0.756, 0.986] | 是 |
| validation | 3 | 1.02 | 0.844 | 0.025 | [0.738, 1.40] | [0.704, 1.18] | 是 |

construction/validation 源不相交一致：两边都是 WB86 全通过、WB85 仍不足。典型 tracklet median chi2/ndof 仍约 0.83。

### 4.2 束流子集 `|t·z|≥0.99`

8 个仅束流 cell 在 WB85 与 WB86 下全部通过。`n_large=0` 的 station 上两路修复在报告精度内一致。WB85 已经覆盖束流分支。

### 4.3 大角度子集 `|t·z|<0.99`

共 3 条，每 cell `n=1 < 30`，不对子集套用 WB83 总体 gate。未删除这些径迹。逐径迹诊断写入 `curvilinear_branch_subset_closure.json`：

| 位置 | `|t·z|` | (tx, ty) | C_yy baseline / WB85 / WB86 | χ² baseline / WB85 / WB86 |
| --- | --- | --- | --- | --- |
| construction 0 | 0.9820 | (−0.190, +0.026) | 0.196 / 3.28×10⁻³ / 7.97×10⁻⁵ | 925 / 9.54 / 2.26 |
| construction 1 | 0.9887 | (−0.048, +0.144) | 0.0211 / **0.182** / 8.29×10⁻⁵ | 3150 / **4190** / 3.64 |
| validation 0 | 0.9890 | (−0.095, +0.116) | 0.104 / **0.150** / 1.03×10⁻⁴ | 1.60×10⁴ / **227** / 3.17 |

construction station 1 的一般分支 `curvU=−(t×ẑ)` 主要沿 −x̂，硬编码 `loc1=−y` 把大的 loc1 方差写进 `C_yy`，一条径迹就把 MEAN `C_yy` 从 1.14×10⁻⁴ 抬到 5.00×10⁻⁴，从而打失败值 gate。这不是 XY swap 残余，也不是 scale。完整 UVT 后 MEAN 与 median `C_yy` 重新对齐。

## 5. 决策

**`segmentfit_full_curvilinear_covariance_contract_validated`**

- WB85 束流近似是主导但**不充分**的 GetState 修复（`wb85_beam_still_insufficient=true`）。
- 补上一般分支后，冻结 WB83 Stage A 的全部 station/split 通过。
- 源级协方差契约在完整 Athena Curvilinear 坐标下闭合。
- `measurement_model_validated = false`
- `measurement_model_v2_discussion_allowed = true`
- `propagated_covariance_validation_authorized = true`
- `stage_b_entered = false`，`faseracts_propagation_entered = false`
- 不写 geometry，不授权 real-data alignment，不打开 Frozen-V2。

## 6. 下一步（本战役未进入）

现已允许单独预注册并进入：

- FaserActs 传播协方差验证（Stage B）
- Measurement Model V2
- conditional Jacobian 战役

本战役不实施上述任何一项。仍禁止经验 rescale、chi2 调参、按角度删径迹。

## 7. 工程产物

- 配置：`configs/full_curvilinear_uvt_covariance_contract_repair_validation_v1.yaml`（SHA `04c563ca…`）
- 模块：`alignment/full_curvilinear_uvt_repair_validation.py`（SHA `c85549dd…`）
- 驱动：`scripts/report_full_curvilinear_uvt_repair_validation.py`（SHA `05772b0e…`）
- 测试：`tests/test_full_curvilinear_uvt_repair_validation.py`（24 个测试，SHA `2439a32b…`）
- 文档：`docs/full_curvilinear_uvt_repair_validation.md`、`docs/full_curvilinear_uvt_repair_validation_cn.md`
- 产物（EOS，不入 git）：`outputs/full_curvilinear_uvt_covariance_contract_repair_validation_v1/`
  - `config_validation.json` `d4ee5b2e…`
  - `source_covariance_closure_three_way.json` `274f25c1…`
  - `curvilinear_branch_subset_closure.json` `571aeccc…`
  - `full_curvilinear_uvt_repair_validation.json` `82831e0b…`
  - `covariance_repair_decision.json` `00944508…`
  - `campaign_summary.json` `ea1f1af9…`
