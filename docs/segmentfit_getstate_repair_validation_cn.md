# GetState 协方差变换修复验证 V1

Workbook 85。**状态：已完成并冻结。**
**最终决策：`segmentfit_getstate_repair_insufficient`。**

本战役验证 WB84 定位的两个确定性 `SegmentFitAlg::GetState` 修复，是否足以恢复
WB83 的源协方差闭合。它在**同一** source-disjoint construction/validation
划分上，用**同一** gate 重跑冻结的 WB83 Stage A。这**不是** alignment 任务。

**继承的冻结状态（SHA 校验，不重开）：**

- WB81 `faseracts_propagated_covariance_not_calibrated`
- WB82 `existing_mc_real_wide_ty_support_validated`（`floor_muon_100120`）
- WB83 `source_tracklet_fit_covariance_not_calibratable` /
  `position_xy_swap_with_slope_miscalibration`
- WB84 `deterministic_segmentfit_get_state_transform_bug` /
  `coordinate_convention_mismatch_and_jacobian_sign_error`

`geometry_write_allowed=false`，`real_data_alignment_authorized=false`，
`measurement_model_validated=false`。不读真实残差，不写 geometry/conditions，
不进入 FaserActs / Stage B，不打开 Frozen-V2。

## 修复范围（封闭）

1. Curvilinear 槽位映射：`loc1 = -y`，`loc2 = +x`（Athena 束流径迹坐标架）。
2. Jacobian 符号：`dφ/dtx = -ty/r²`。

禁止：scale factor、经验重标定、chi2 调参、基于残差的修正、station 相关标定、
改中心预测。

修复离线实施为确定性变换
`C_rep = J_rep J_buggy^{-1} C_exp J_buggy^{-T} J_rep^T`。拟合态
`[x,y,tx,ty]` 不变。

## Gate（与 WB83 完全相同，未改）

`chi2/ndof ≤ 4`，`Cov(z)` 特征值 ∈ [0.25, 4]，广义特征值 ∈ [0.25, 4]，
construction/validation 源不相交一致性。

## 结果

Baseline 复现 WB83：全部 8 个 cell 失败（median chi2/ndof ~ 360–437）。

两个 GetState 修复之后（3230 条 tracklet，0 次反演失败）：

| cell | chi2/ndof | median | frac>4 | calibrated |
| --- | --- | --- | --- | --- |
| construction 0 | 1.00 | 0.838 | 0.026 | **是** |
| construction 1 | 3.68 | 0.851 | 0.026 | **否**（mean C_yy） |
| construction 2 | 0.987 | 0.849 | 0.015 | **是** |
| construction 3 | 0.973 | 0.824 | 0.022 | **是** |
| validation 0 | 1.11 | 0.826 | 0.031 | **否**（mean C_yy） |
| validation 1 | 0.932 | 0.784 | 0.022 | **是** |
| validation 2 | 0.910 | 0.820 | 0.008 | **是** |
| validation 3 | 1.02 | 0.844 | 0.025 | **是** |

典型 tracklet 契约已恢复：median chi2/ndof = 0.83，仅做位置 XY swap 现在会把
chi2 **恶化**到 ~300，`var(ty)` 已校准。两个失败 cell 只失败在**均值**协方差
本征 gate：MEAN `C_yy` 被 1–2 条大角度径迹（`|t·z|<0.99`）抬高，对这些径迹
束流槽位映射不是 Athena Curvilinear 架。这些 cell 上的 median `C_yy` 仍然校准。
这**不是**残余 XY 交换，也**不是**加 scale factor 的理由。

## 决策

`segmentfit_getstate_repair_insufficient`。

WB84 的两个 bug 是 WB83 失败的**主导**原因（6/8 cell 通过；典型 tracklet
白化已恢复）。在冻结的均值协方差 gate 下，它们**还不足以**覆盖每个
station/split。

`measurement_model_v2_discussion_allowed=false`。Measurement Model V2、
conditional Jacobian、alignment closure 仍然关闭。

## 下一步（此处未授权）

后续战役可单独预注册**完整 CurvilinearUVT 槽位映射**（两个 `|t·z|` 分支），
仍保持确定性、不加 scale，并重跑同一 Stage A。本战役不实施该扩展。
