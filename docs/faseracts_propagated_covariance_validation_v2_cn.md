# FaserActs 传播协方差验证 V2

Workbook 87。**状态：已完成并冻结。**
**最终决策：`faseracts_transport_covariance_not_validated`。**
**机制：`q_over_p_uncertainty_semantics`。**

这是 WB86 之后获授权的 Stage B。它检验

    C_target = J_transport C_source_WB86 J_transport^T + process_noise

是否能在冻结的 WB83/86 source-disjoint MC 上描述
`e_target = 传播拟合态 − 目标面 truth`。这**不是** alignment 任务。WB81
mode 3 **不能**当作答案继承。

**继承的冻结状态（SHA 校验，不重开）：**

- WB81 `faseracts_propagated_covariance_not_calibrated` /
  `overestimated_transported_fit_covariance`
- WB82 `existing_mc_real_wide_ty_support_validated`（`floor_muon_100120` 仅预留）
- WB83 `source_tracklet_fit_covariance_not_calibratable`
- WB84 `deterministic_segmentfit_get_state_transform_bug`
- WB85 `segmentfit_getstate_repair_insufficient`
- WB86 `segmentfit_full_curvilinear_covariance_contract_validated`

`geometry_write_allowed=false`，`real_data_alignment_authorized=false`，
`measurement_model_validated=false`。不读真实残差，不写 geometry/conditions，
不打开 Frozen-V2，不用 truth q/p 作 real-data 解，不把 covariance 调到
chi2=1，不把删除 q/p 列当作最终方案。

## Part A — 确定性传输

`C = J_geom C_src_WB86 J_geom^T`，不加 process noise。`J_geom` 是无场 4D
力臂 Jacobian。生产 ACTS 没有 material process noise（WB81 provenance）。
经验上 `J_geom C_exported J_geom^T` 与 ntuple mode-3 `C_prop` 一致。

**Jacobian 自洽**（`J e_source` vs `J C J^T`）：两个 split 上 **6/6 cell
通过**（χ²/ndof ≈ 1.00，median ≈ 0.85，`frac(χ²>4)` ≈ 0.025）。已验证的源
协方差被正确传输。4D Jacobian **不是**失败原因。

**同一 `C` 对 `e_target`：** 全部 6 个 cell 失败。剩余
`e_target − J e_source` 在 `y/ty`，并随 `1/p` 缩放（无约束动量下的磁场传输），
而不是 100 GeV Highland 项。

## 四种 mode（`e_target` 全部失败；split 一致）

| mode | 模型 | 典型结果 |
| --- | --- | --- |
| 0 | 现有生产 `C_prop` | χ² ~ 800–1.5×10⁵；pencil **727–43890×**（复现 WB81） |
| 1 | WB86 源 + 现有 q/p ΔC | median χ² ~ 0.7–1.0，但 pencil 仍 **727–43890×** |
| 2 | WB86 源，**不**传输 dummy q/p | pencil **~1.06–1.22**（已修好）；χ² **4×10⁴–1.4×10⁶**（y/ty 低估） |
| 3 | mode 2 + Highland 5% X0，未调 chi2 | χ² 降约 10–50×，仍 **3.8×10³–5.4×10⁴** |

## q/p 审计

- **带** dummy q/p（mode 0）：虚假 pencil，与 WB81 同一机制。
- **不带** dummy q/p（mode 2）：pencil 回到 ~1；精密面 χ² 爆炸。
- SegmentFit dummy `q/p` 协方差是直线 tracklet **未约束输入**，不是物理动量不确定度。
- 因为 IFT→下游传输穿过磁场，**需要物理** `q/p`。删除 q/p 列**不是**最终方案。
- Truth q/p 只作诊断，不是 real-data 解。

## 决策

`faseracts_transport_covariance_not_validated` /
`q_over_p_uncertainty_semantics`。

源协方差不再是限制因素。FaserActs 对它的 4D 确定性传输相对 `J e_source`
已验证。生产 `C_prop` 以及所有预注册的 `J C_src J^T (+ Highland)` 重建都
不能闭合 `e_target`。并行事实：ACTS process noise 关闭；第一性原理 Highland
不足，且不提升为生产模型。

`measurement_model_v2_discussion_allowed=false`。
`measurement_model_v2_entered=false`。
`stage_b_entered=true`（仅本战役）。

后续战役可单独预注册磁场传输用的**物理动量先验**（不是 dummy 方差、不是删列、
不是 real data 上的 truth q/p）和/或 ACTS material process-noise 契约。本战役
不进入 Measurement Model V2 或 alignment。
