# Workbook 81: FaserActsExtrapolation 传播协方差来源与 MC 闭合验证 V1

日期：2026-09-05
状态：**完成并冻结** —— 单一交互会话内按冻结顺序执行；所有验证 gate 在任何 confirmatory 计算前写入 config 并冻结，执行后未修改任何 gate。

**最终决策：`faseracts_propagated_covariance_not_calibrated`**
**机制分类：`overestimated_transported_fit_covariance`**

> **命名澄清：本 Workbook 81 不是 "Gauge-Fixed Real-Data Alignment Diagnostic V2"。** WB80 已失败（`measurement_model_multiple_components_not_validated`），那个条件成功分支**没有被触发**。连续编号 81 用于本 root-cause campaign。真正的 Real-Data Alignment Diagnostic V2 必须等 measurement model 完整验证后另分配新编号。

本战役是**上游 covariance validation，不是 alignment**。整个 Workbook 81：

- `held_out_accessed = false`（不读任何 real-data residual，held-out 完全关闭）
- `real_data_alignment_authorized = false`
- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `external_constraint_ingest_authorized = false`
- 第一阶段**不修改** `FaserActsExtrapolationTool`（`do_not_modify_faseracts_extrapolation_tool=true`）

## 1. 科学问题（唯一）

**`FaserActsExtrapolationTool` 输出的 propagated covariance（`C_prop`）到底代表什么？它的 covariance transport / process noise 能否在 truth-known MC 上正确描述 source-tracklet → downstream-surface 的 prediction error？**

WB80 已证明：frozen combined covariance 的巨大 χ² 不是数值 bug；`C_combined = C_prop + C_target` 语义已确认；但 `C_prop` 来自外部 Calypso/ACTS 工具，其 provenance 在 WB80 时未闭合，且其 pencil 方向 covariance 明显过强（过估真实方差 ~100–2000×）。本战役把 `C_prop` 从 `C_target` 与 real-data residual 中彻底隔离，用 truth-known MC 单独验证 `C_prop`。

## 2. 起始状态审计（STEP 1，已完成）

- Workbook 80 freeze commit：`ed9c33c`（单一 freeze commit，含 code+config+docs+workbook+tests；artifacts 存于 EOS，不入 git）。
- WB80 最终冻结：`measurement_model_multiple_components_not_validated`，并确认：
  - `covariance_model_validated = false`、`jacobian_transfer_model_validated = false`
  - `real_data_alignment_v2_preregistration_allowed = false`、`held_out_accessed = false`
- 本战役 `load_config` 对 WB80 逐项 SHA 验证（不一致即 `ConfigError` 拒绝）：
  - WB80 config `8f0f0445741611e2f4ddbe6889eb107eaa18b9a5af56f655aaac120c44fa74d2`
  - `measurement_model_decision.json` `c450a0b946961d1663f87bd0b4d5a26ccb8c5e7a21ab634aca8c415e05b52a90`
  - `campaign_summary.json` `75c04b987506bd1d4c9bfe64178afe2a9dfd64eed1bd604961345c1afb597421`
- WB80 冻结决策与四项 gate 已逐字核对一致。

## 3. 冻结数据角色与禁止事项

- **MC-only**：本战役原则上不需要任何 real-data residual solve。使用已有 hierarchical V1 MC source 的 `iteration_00_reference`（nominal geometry，alignment constants 全 0）点。
- **Source-disjoint MC**（config 预冻结，源文件级互斥）：
  - construction：`mc24_100043_00200_00299`、`mc24_100043_00300_00399`、`mc24_100043_00400_00499`、`mc24_100044_00200_00299`、`mc24_100044_00300_00399`
  - validation：`mc24_100047_00000_00049`、`mc24_100047_00050_00099`、`mc24_100048_00000_00049`、`mc24_100048_00050_00199`
- 禁止：covariance floor/inflation tuning、diagonal covariance 提升为 official W、J 外推、打开 held-out、nonlinear scan、改 gauge/S/rank_tolerance、删 event/source、按 residual/pull/condition 删 pair、用 external evidence 作 prior、把 diagnostic variant 提升为 production。

## 4. Stage 0 — Covariance provenance 审计（`faseracts_covariance_provenance.json`）

直接审计本地 Calypso（git `40892527e9c6…`，Athena `24.0.41`，ACTS `32.0.2`）源码与 runtime config，逐项回答预注册的 15 个问题。**结果：15/15 全部 resolved（11 项 resolved_from_source + 4 项 resolved_from_runtime_config），0 项 unresolved，`provenance_closed=true`。**

关键结论（每条均带 文件:行号/函数 证据，详见 artifact）：

1. **input basis**：native Athena `Trk::TrackParameters` 5×5 `(loc1,loc2,phi,theta,q/p)`。
2. **segment→ACTS**：`actsTrackletParameters` 用中心数值 Jacobian（5→6）转到 ACTS bound 6×6；q/p 列**被 transport**（除非 mode 3 suppress）。
3. **transport Jacobian 位置**：ACTS `EigenStepper` 逐步累积 `jacTransport`，最终 `C → J_full C J_full^T`（`CovarianceEngine::transportCovarianceToBound`）。
4. **output 回 `[x,y,tx,ty]`**：`globalActsTrackletCovariance` 用中心数值 Jacobian（6→4）。
5. **material effects**：**未启用**。
6. **multiple-scattering process noise**：**未加入**。
7. **energy-loss uncertainty**：**未加入**。
8. **process noise surface/step**：`MaterialInteractor` 三 flag 全 false → early-return，任何 surface/step 都不加。
9. **particle hypothesis**：硬编码 `Acts::ParticleHypothesis::muon()`。
10. **q/p mode 影响**：mode 0 传输 source fit 的 q/p 方差；mode 3 suppress；mode 1/2 用 truth q/p。
11. **field covariance**：FASER 场为确定性 map，无 field covariance；数值 transport 为 RK EigenStepper。
12. **surface frame/local-global Jacobian**：中心数值 Jacobian + 长 lever arm 可放大相关项（已经验表征）。
13. **prediction vs conditional**：`C_prop` 是 **prediction covariance**（source fit covariance 的传输），**不是** conditional covariance。
14. **确定性 transport 无随机噪声**：**确认**——fit covariance 被确定性传输（`J C J^T`），但**未加入任何实际 stochastic transport noise**。
15. **runtime config 实际取值**：`NtupleDumperAlgCfg` 只设 `MaxSteps=10000` 与 `TrackingGeometryTool`；`FieldMode='FASER'`；三个 material flag 全默认 false。

## 5. Stage 1 — truth/reference target state（`truth_reference.json`）

- **路径**：复用现有 `enhanced_tracklets.root` 的 per-station Geant-truth state（`truth_stX_x/y/z`、`truth_stX_px/py/pz`），按 `(run, eventID, truth_barcode)` 与 propagation record 的 `(run_id, event_id, truth_particle_id)` join。**未发明任何不存在的 branch，未新增 exporter，未改 reconstruction selection。**
- **构造**：把 target-station truth 位置用 truth 方向（`tx=px/pz, ty=py/pz`）直线修正到 propagation target 平面（`target_z_mm`），dz≈1 mm 修正已验证可忽略场曲率。
- **Reference floor（mode 2 全-truth source 控制）**：truth-source 传播的 `e_prop` RMS 即不可约 floor。预注册 gate：signal/floor ≥ 3。

| station pair | floor (mm) | signal (mm) | signal/floor | gate |
| --- | --- | --- | --- | --- |
| (0,1) | 9.17 | 258.79 | 28.2 | 过 |
| (0,2) | 4.19 | 421.36 | 100.6 | 过 |
| (0,3) | 5.01 | 592.78 | 118.2 | 过 |

`reference_floor_ok=true`：truth reference 足以分辨 reco-source 的主导 propagation 误差。

## 6. Stage 2 — source-disjoint MC truth 闭合（`closure.json`，production mode 0）

核心 observable：`e_prop = propagated source-tracklet state − truth state at target surface`（**不含** `C_target`）。对 `z_prop = C_prop^{-1/2} e_prop`（去均值涨落）做 whitening/eigenmode/pencil/generalized-eigenvalue 分析。**预注册 gate**：whitened χ²/ndof ≤ 4；Cov(z) 特征值 ∈ [0.25,4]；generalized 特征值 ∈ [0.25,4]；pencil 方差比 ∈ [0.25,4]。

**结果：6 个 cell（3 station pair × 2 split）全部失败（`calibrated=false`），且两个 source-disjoint split 一致。**

| split | pair | n | χ²/ndof | Cov(z) eig max | pencil 方差比 | generalized eig |
| --- | --- | --- | --- | --- | --- | --- |
| constr | (0,1) | 416 | 8017 | 31679 | 310 | [1.3e-4, 5.3e-4, 4.5e-2, 4.5] |
| constr | (0,2) | 408 | 46765 | 186958 | 1242 | [5.9e-5, 9.1e-5, 4.0e-3, 1.0] |
| constr | (0,3) | 393 | 149577 | 596814 | 2399 | [4.8e-5, 6.9e-5, 1.2e-3, 0.40] |
| valid | (0,1) | 304 | 1142 | 3173 | 2502 | [3.6e-5, 6.9e-5, 2.7e-4, 40] |
| valid | (0,2) | 305 | 3291 | 11551 | 2264 | [3.8e-5, 4.2e-5, 4.7e-5, 7.6] |
| valid | (0,3) | 294 | 48611 | 191339 | 3894 | [1.5e-5, 3.2e-5, 7.4e-5, 2.7] |

- **复现 WB80 发现**：`C_prop` 的 pencil 方向（大特征值方向）过估真实 physical propagation 方差 **310–3894×**（WB80 估计 100–2000×，本 truth-based 闭合确认并放大该结论）。
- **orientation 与 scale 都错**：generalized 特征值同时存在 ≪1（三个方向，C_prop 过估）与 >1（一个方向，C_prop 欠估）——既不是单纯 scale 错误，也不是单纯 orientation 错误，而是**两者皆错**。
- 相关结构：`corr(y,ty)` C_prop≈0.97–0.995 vs 经验≈0.95–0.98（方向大致对但 scale 错）；x-tx block 经验相关≈1.0（强 pencil），但 C_prop 把最大特征值放在 y-ty（q/p 诱导的虚假 pencil）。
- lever-arm / |tx| / |ty| 分 bin 后 χ²/ndof 在所有 bin 都 ≫1（数千至数十万），说明 miscalibration 是**弥漫性**的，不局限于某个运动学角落。

## 7. Stage 3 — q/p-mode diagnostic variants（`diagnostic_variants.json`，diagnostic_only）

预注册 4 个 q/p 处理模式（全部 `diagnostic_only=true`、`alignment_authorized=false`，绝不提升为 production）。决定性对比是 **mode 0（传输 q/p 协方差）vs mode 3（suppress q/p 协方差）**：

| mode | 说明 | (0,1) pencil 比 | (0,2) pencil 比 | (0,3) pencil 比 | χ²/ndof 范围 |
| --- | --- | --- | --- | --- | --- |
| 0 | reco source + reco q/p（**传输 q/p cov**，production） | **310** | **1242** | **2399** | 1.1e3–1.5e5 |
| 1 | reco source + truth q/p | 0.10 | 0.09 | 0.09 | 8.0e4–5.5e5 |
| 3 | reco source + reco q/p（**不传输 q/p cov**） | 0.10 | 0.09 | 0.09 | 2.5e4–1.0e6 |
| 2 | truth source + truth q/p（reference floor，无 cov） | — | — | — | — |

**机制定位（决定性）**：

- mode 0 的虚假 pencil（过估 310–3894×）在 mode 3（去掉 q/p 协方差传输）中**完全崩塌**（pencil 比从 ~10³ 掉到 ~0.1，约 4000×  collapse）。这证明：**WB80 所见的 pencil 方向过估，是由 source tracklet fit 的（直线拟合不约束的）dummy q/p 方差经数值 Jacobian 传输进 `C_prop` 造成的。**
- 但 mode 3 / mode 1 同时暴露**第二个问题**：去掉 q/p 协方差后，`C_prop` 在主导方向**欠估**真实误差 10–100×（pencil 比 ~0.01–0.10），χ²/ndof 反而更大（2.5e4–1.0e6）。即**底层 source tracklet fit 协方差本身欠估真实 tracklet fit 误差**。

**结论：`C_prop` 在 scale 与 orientation 上、在两个方向上都错误——q/p 协方差传输造成过估的虚假 pencil，而底层 source fit 协方差又欠估弯曲面误差。任何单一修复都不够。**

## 8. 决策与物理解读

- 冻结决策：`faseracts_propagated_covariance_not_calibrated`。
- 机制分类：`overestimated_transported_fit_covariance`（主导、WB80-复现的 pencil 过估，由 q/p 协方差传输造成；另伴底层 source fit 协方差欠估）。
- **回答 §1 科学问题**：当前 `FaserActsExtrapolationTool` 输出的 `C_prop` **不能**在 truth-known MC 上正确描述 source→target prediction error。它是 source fit 协方差的确定性传输（`J C J^T`），**不含任何 multiple-scattering / energy-loss process noise**，且把 source fit 的 dummy q/p 方差传输成主导虚假 pencil。
- `propagated_covariance_model_validated=false`、`real_kinematic_jacobian_support_validated=false`、`measurement_model_validated=false`、`real_data_alignment_v2_preregistration_allowed=false`、`held_out_accessed=false` 全部保持。
- 未执行（按规则禁止）：修改 `FaserActsExtrapolationTool` 后再跑 alignment、打开 held-out、real-data residual solve、geometry/conditions write、covariance floor/inflation tuning、diagonal covariance 提升、J 外推、用 external prior、把 diagnostic variant 提升为 production。

## 9. 后续路径（失败分支）

按预注册 §9，本战役确认 upstream covariance semantic mismatch，**不在本战役内 patch 后进入 alignment**。另开独立 campaign：

**Propagated-Covariance Upstream Repair & MC Validation V1**（`propagated_covariance_upstream_repair_mc_validation_v1`）

修复方向（需在该 campaign 内预注册并重新跑 source-disjoint truth 闭合）：

1. **q/p 协方差处理**：直线 source tracklet fit 不约束 q/p，其 dummy q/p 方差不应传输进 `C_prop`（或在传输前 suppress / 用物理 q/p 先验）。
2. **source fit 协方差校准**：底层 tracklet fit 协方差欠估真实 fit 误差 10–100×，需重新校准。
3. **process noise**：当前确定无 multiple-scattering / energy-loss noise；若物理上需要，应在修复 campaign 内预注册并验证。

**只有 upstream `C_prop` 闭合通过后，才允许重新定义 combined residual covariance。**

同时，独立的 **Workbook 82（Wide-ty Real-Support-Matched MC Coverage Feasibility V1）** 支线仍必须进行：WB80 的 real (0,1)/(0,2) `pred_ty` 支持不足（within-support 0.806/0.872 < 0.90）是**独立** blocker。两个独立 gate `propagated_covariance_model_validated` 与 `real_kinematic_jacobian_support_validated` **都为 true 前**，`measurement_model_validated=false`、`real_data_alignment_v2_preregistration_allowed=false`。两支线不得互相救援。

## 10. Artifact 清单与 SHA256（存于 EOS `outputs/faseracts_propagated_covariance_provenance_closure_v1/`，不入 git）

| artifact | SHA256 |
| --- | --- |
| config (`configs/faseracts_propagated_covariance_provenance_closure_v1.yaml`) | `5a13b0cc7273546ebbee4268a853c32c527a68fd263eb82483f3d6e581d5f2b9` |
| config_validation.json | `7075dab7ebe89c0cc6a95c054a4318967d8f47b1ddc59de26c96549b2547e487` |
| faseracts_covariance_provenance.json | `5a8c376e72ff250bc784d6287631590f24bba831c85ee479dc509fe2caafe9b9` |
| truth_reference.json | `4d6a270b34b070aeff9dce1f5623e35505153c3818edbfada570b9a79b62aa2c` |
| closure.json | `27ac8820d5e5dc455d63e08e4e037ecbb17847fedba20ca6771c484397e41cf7` |
| diagnostic_variants.json | `79e90e973cc2809601573afcc1b4a401600abefe653a7d0a9d70c9a1d62c296d` |
| propagated_covariance_decision.json | `fd605f0ea93dd239a06c4ae8f14a619e1735709963d657b4c9a68575e9db96c2` |
| campaign_summary.json | `97a92006163770ee3986a992813b3abb2a58843f7953f25134de1f8412645a4f` |

工程文件：`configs/faseracts_propagated_covariance_provenance_closure_v1.yaml`、`alignment/propagated_covariance_closure.py`、`scripts/report_faseracts_propagated_covariance_closure.py`、`tests/test_propagated_covariance_closure.py`（40 项回归）、`docs/faseracts_propagated_covariance_closure.md` / `_cn.md`。

冻结提交见本节末尾 git 记录；完整测试套件（含 40 项 WB81 回归）通过。
