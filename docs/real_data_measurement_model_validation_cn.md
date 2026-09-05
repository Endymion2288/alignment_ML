# 真实数据 Measurement-Model 重建与跨 Run 验证 V1

Workbook 80。**状态：完成并冻结** —— 单一交互会话内按冻结顺序执行；所有验证 gate 在任何 confirmatory 计算前写入 config 并冻结，执行后未修改任何 gate（仅修复两处不影响 gate/科学的代码 bug）。

**最终决策：`measurement_model_multiple_components_not_validated`。** 两个独立关键 gate 全部失败：(A) residual covariance model 未通过 cross-fit 验证；(B) MC→real Jacobian-transfer model 的 real-data 支持不足。因此 `covariance_model_validated=false`、`jacobian_transfer_model_validated=false`，cross-run information 检查按预注册规则**结构性跳过**，`real_data_alignment_v2_preregistration_allowed=false`。真实数据继续 `residual_dq_monitoring_only`；**不开 Workbook 81**。

本战役只重建并验证 measurement model（covariance + Jacobian transfer），**不求最终 alignment correction、不打开 held-out、不写 geometry、不做 nonlinear iteration、不改变 gauge、不重新定义 V_id/V_null、不修改 S/rank_tolerance、不使用 external prior**。整个 Workbook 80：

- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `real_data_candidate_alignment_authorized = false`
- `external_constraint_ingest_authorized = false`
- `held_out_accessed = false`（held-out 继续完全关闭，代码级 guard）

## 科学问题（唯一）

Workbook 79 已冻结 `real_data_statistical_model_multiple_failures`。本战役问题：能否从第一性原理重建并验证一个可信的 measurement model——(A) residual covariance model，(B) MC→real Jacobian-transfer model——使二者各自通过独立的、residual-blind 的 cross-run / MC-source-disjoint 验证？只有两者都通过且 cross-run information 稳定，才允许另开 Workbook 81（Gauge-Fixed Real-Data Alignment Diagnostic V2）。本战役**本身不产生任何 deployable candidate**。

## 冻结输入（不重新选 population）

- **Calibration（唯一允许使用的真实数据）**：run 14973 + run 14974，以及被冻结的 route/event/source identity、Frozen-V2 selected-route 合同与 split SHA。
- **Held-out（继续完全关闭）**：14975/14976 + 7 个 monitoring run（14971/14972/14980/14981/14985/14989/15007）+ 14977 report-only。`assert_no_held_out_access` 代码级 guard 对任何 held-out run id 直接 raise。
- Tracker information / gauge / identifiable basis / S / rank_tolerance 完全继承 WB78（经 WB79 链加载并 SHA 验证）；frozen V_id/V_null 在 `load_config` 时注入，**绝不在真实数据上重新 SVD**。

## 阶段与结果

1. **复现 WB79 baseline（硬停止）**——通过。WB79 covariance eigensystem/whitening baseline 逐项精确复现（6/6 checks）：bank SHA `6e4eaae0...`、χ²_total=1999613.163574398、whitened χ²/ndof=1058.4585330827601、最小 eigenmode 占比 0.99741、condition 中位数 2.347e12、n_pairs=378。

2. **Covariance 生成语义追踪**——combined 4×4 covariance 为 `C_combined = C_propagated_source + C_target`（plain sum，逐位验证 max abs diff=0.0），假设 source/target 独立。`C_prop` 由外部 Calypso/ACTS 外推算出，经验上近奇异（median condition ~4.4e14）；**该工具是否已含 multiple-scattering process noise 从本 repo 不可确认**（unresolved external provenance）。marginal per-pair pull RMS（x=1.68、y=0.13、tx=0.42、ty=0.12）大致校准——问题在 off-diagonal 相关结构，不在 marginal。

3. **数值求逆审计**——四种方法（direct inv / Cholesky / eigendecomposition / SVD）对同一 frozen covariance 的总 χ² 一致到 ~2.2e-7，零 factorization 失败。巨大 χ² **不是数值 artifact**，而是真实的统计模型失配。本阶段严格是 method-A（同一 C⁻¹ 的稳定计算）比较，未修改 C 本身。

4. **Covariance model cross-fit 验证——失败。** 候选只来自第一性原理（绝不以 γ/rank/condition/χ² 为设计依据）：`C0_frozen`、`C1_ms_leverarm`（multiple-scattering 随机游走 process noise `C = C_combined + θ²·G_MS(L)`）、`C2_scaled_diagonal_floor`、`C3_variance_inflation`。nuisance scale 用 Gaussian NLL 在严格 14973↔14974 cross-fit 下估计（禁止同-run 自证）。C1、C2 把 whitened χ²/ndof 从 ~1000 降到 ~1（约 1000×、物理上合理的改善），最大 Cov(z) 特征值健康，**但最小 Cov(z) 特征值 ~0.001–0.009，远低于 gate 0.25**。根因：`C_prop` 沿 pencil 方向过估真实 residual 方差 ~100–2000×（其声称的强 x↔tx 相关在真实 residual 中并不那么强），叠加式修正无法去除该过相关。无候选复现 residual 相关结构，`covariance_model_validated=false`。

5. **MC-only conditional Jacobian model——MC 验证通过。** 三个候选（`J0_station_pair_mean`、`J1_kinematic_binned`、`J2_linear_regression`）在 3 个 construction MC source 上拟合、4 个互斥 validation MC source 上验证，均通过全部 MC gate（per-pair Frobenius 中位 ~0.5–1.6%、injection recovery ~5–6%、null leakage ~1%）。特征 residual-blind（仅 pred_tx, pred_ty）。

6. **Real-data transfer-support 验证——失败。** 冻结的 MC conditional-J model 应用到 14973/14974 kinematics（residual-blind）：仅 **80.6%** 的 (0,1)、**87.2%** 的 (0,2) real pair 落在 MC 99% Mahalanobis (tx,ty) 包络内（gate ≥90%）；real ty 分布比 MC 宽 ~2.7×。out-of-support pair 未外推进任何 solve。`jacobian_transfer_model_validated=false`。

7. **Cross-run information sanity check——结构性跳过**（两个组件模型均未过，检查无意义，未运行）。

## 决策与物理解读

`measurement_model_multiple_components_not_validated`。在当前 frozen 输入下，**无法**重建并验证一个可信的 measurement model：covariance model 无法复现真实 residual 的相关结构（propagated covariance 的 pencil 方向过相关），Jacobian-transfer model 的 real-data 运动学支持不足。所有冻结 flag 保持；继续 `residual_dq_monitoring_only`；不开 Workbook 81。

有科学依据的后续方向（各需另开独立预注册 campaign）：(1) 直接从外部传播工具的 provenance 确认其 covariance transformation 与 multiple-scattering process-noise 处理，若确认 pencil 方向方差系统性过估，应在上游修正 propagated covariance，而非在本 repo 叠加修正；(2) 获取覆盖宽 ty 真实拓扑的 MC/control 样本，把 conditional-J model 的 applicability domain 扩到真实支持。

完整执行记录与 artifact SHA 见 workbook 80 §6。
