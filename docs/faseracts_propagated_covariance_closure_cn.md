# FaserActsExtrapolation 传播协方差来源与 MC 闭合验证 V1

Workbook 81。**状态：完成并冻结** —— 单一交互会话内按冻结顺序执行；所有验证 gate 在任何 confirmatory 计算前写入 config 并冻结，执行后未修改任何 gate。

**最终决策：`faseracts_propagated_covariance_not_calibrated`。**
**机制分类：`overestimated_transported_fit_covariance`。**

> **命名澄清：** 本 Workbook 81 **不是** "Gauge-Fixed Real-Data Alignment Diagnostic V2"。Workbook 80 已失败（`measurement_model_multiple_components_not_validated`），该条件成功分支从未触发。真正的 Alignment Diagnostic V2 须等 measurement model 完整验证后另分配编号。

本战役是**上游 covariance validation，不是 alignment**。整个 workbook：`held_out_accessed=false`、`real_data_alignment_authorized=false`、`geometry_write_allowed=false`、`official_conditions_write_allowed=false`、`external_constraint_ingest_authorized=false`，且第一阶段**不修改** `FaserActsExtrapolationTool`。

## 科学问题（唯一）

`FaserActsExtrapolationTool` 输出的 propagated covariance（`C_prop`）到底代表什么？它的 covariance transport / process noise 能否在 truth-known MC 上正确描述 source-tracklet → downstream-surface 的 prediction error？

WB80 已证明 frozen combined covariance 的巨大 χ² 不是数值 bug，`C_combined = C_prop + C_target` 语义正确，但 `C_prop` 来自外部 Calypso/ACTS 工具，其 pencil 方向 covariance 明显过强（过估真实方差 ~100–2000×）。本战役把 `C_prop` 从 `C_target` 与 real-data residual 中隔离，用 truth-known MC 单独验证 `C_prop`。

## Stage 0 — provenance 审计（`faseracts_covariance_provenance.json`）

直接审计本地 Calypso 源码（git `40892527e9c6…`，Athena `24.0.41`，ACTS `32.0.2`）与 runtime config。**15 个预注册问题全部 resolved（11 项来自源码、4 项来自 runtime config），0  unresolved，`provenance_closed=true`。** 关键结论（每条带 文件:行号 证据）：

- 输入为 native Athena `Trk::TrackParameters` 5×5 `(loc1,loc2,phi,theta,q/p)`。
- `actsTrackletParameters` 经中心数值 Jacobian 转到 ACTS bound 6×6；q/p 列**被传输**（除非 mode 3 suppress）。
- transport 为 `C → J C J^T`，由 ACTS `EigenStepper` 累积、`CovarianceEngine::transportCovarianceToBound` 施加。
- **material effects、multiple-scattering process noise、energy-loss uncertainty 全部禁用**（`MaterialInteractor` early-return）。
- particle hypothesis 硬编码为 `Acts::ParticleHypothesis::muon()`。
- `C_prop` 是 **prediction covariance**（source-fit covariance 的传输），**不是** conditional covariance，且**不含任何 stochastic transport noise**。

## Stage 1 — truth/reference target state（`truth_reference.json`）

reference target state 复用 `enhanced_tracklets.root` 现有 per-station Geant-truth state（`truth_stX_*`），按 `(run, eventID, truth_barcode)` join，直线修正到 propagation target 平面。不新增 exporter、不发明 branch、不改 reconstruction selection。mode-2 全-truth source 控制校准不可约 reference floor；预注册 `signal/floor ≥ 3` gate 对三个 pair 全过（28×、101×、118×），`reference_floor_ok=true`。

## Stage 2 — source-disjoint MC truth 闭合（`closure.json`，production mode 0）

observable：`e_prop = propagated source-tracklet state − truth state at target`（不含 `C_target`）。对 `z_prop = C_prop^{-1/2} e_prop` 按 station pair 与 residual-blind 运动学 slice 做 whitening/eigenmode/pencil/generalized-eigenvalue 分析，用两个 source-disjoint MC split（construction 5 源、validation 4 源，文件级互斥）。

**6 个 cell（3 pair × 2 split）全部失败（`calibrated=false`），两个 split 一致。** whitened χ²/ndof 范围 1142–149577（gate ≤ 4）；pencil 方向方差比（C_prop/经验）**310–3894×**，复现并放大 WB80 发现；`(C_empirical, C_prop)` 的 generalized 特征值同时 ≪1（三个方向，过估）与 >1（一个方向，欠估）——**orientation 与 scale 都错**。miscalibration 在 lever-arm / |tx| / |ty| 各 bin 弥漫存在。

## Stage 3 — q/p-mode diagnostic variants（`diagnostic_variants.json`）

4 个预注册 q/p 处理模式（全部 `diagnostic_only=true`、`alignment_authorized=false`）。决定性对比为 mode 0（传输 q/p 协方差）vs mode 3（suppress q/p 协方差）：

- **mode 0（production）**：pencil 过估 310–3894×。
- **mode 3（suppress q/p 协方差）**：虚假 pencil **完全崩塌**（约 4000× 掉到 ~0.1），证明 WB80 所见 pencil 过估是由直线 source fit 的（不约束的）dummy q/p 方差传输造成。
- **但** mode 3 / mode 1 随后在主导方向**欠估**真实误差 10–100×（pencil 比 ~0.01–0.10），χ² 更大——底层 source tracklet fit 协方差本身欠估真实 fit 误差。

因此 `C_prop` 在 scale 与 orientation 上、在两个方向上都错误：q/p 协方差传输造成过估的虚假 pencil，底层 source-fit 协方差又欠估弯曲面误差。

## 决策与后续

- 冻结决策：`faseracts_propagated_covariance_not_calibrated`；分类 `overestimated_transported_fit_covariance`。
- `propagated_covariance_model_validated=false`、`real_kinematic_jacobian_support_validated=false`、`measurement_model_validated=false`、`real_data_alignment_v2_preregistration_allowed=false`、`held_out_accessed=false` 全部保持。
- 按预注册规则，upstream bug **不在本战役内 patch**。需另开独立 **Propagated-Covariance Upstream Repair & MC Validation V1** campaign；只有 upstream `C_prop` 闭合通过后才允许重新定义 combined residual covariance。
- 独立的 **Workbook 82（Wide-ty Real-Support-Matched MC Coverage Feasibility V1）** 支线仍必须进行：WB80 的 real (0,1)/(0,2) `pred_ty` 支持不足是独立 blocker。两个 gate（`propagated_covariance_model_validated` 与 `real_kinematic_jacobian_support_validated`）都为 true 前不得做 measurement-model v2 预注册；两支线不得互相救援。

## Artifacts

存于 EOS `outputs/faseracts_propagated_covariance_provenance_closure_v1/`（不入 git）；SHA256 见 Workbook 81 workbook。工程文件：`configs/faseracts_propagated_covariance_provenance_closure_v1.yaml`、`alignment/propagated_covariance_closure.py`、`scripts/report_faseracts_propagated_covariance_closure.py`、`tests/test_propagated_covariance_closure.py`（40 项回归）。
