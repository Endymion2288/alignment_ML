# 宽 ty 真实运动学支持匹配 MC 覆盖可行性 V1

Workbook 82。**状态：完成并冻结** —— 单一交互会话内按冻结顺序执行；所有 gate 在任何 confirmatory 计算前写入 config 并冻结，执行后未修改任何 gate。

**最终决策：`existing_mc_real_wide_ty_support_validated`。**
**validated candidate：`floor_muon_100120`（floor-origin muon gun）。**

> **命名：** 本 Workbook 82 是 WB81 明确保留的独立支线（连续编号 82），**不是** covariance repair。covariance repair 属于 Workbook 83（`Propagated-Covariance Upstream Repair & MC Validation V1`），只有在 WB82 冻结后才允许启动。

本战役**完全 residual-blind**。整个 Workbook 82：`held_out_accessed=false`、`real_data_alignment_authorized=false`、`geometry_write_allowed=false`、`official_conditions_write_allowed=false`、`external_constraint_ingest_authorized=false`、`measurement_model_validated=false`（不触碰）、`residual_blind=true`——**不读** alignment residual / FD derivative / Jacobian error / singular value / rank / final-correction 表现。

## 科学问题（唯一）

现有非 sealed MC/control 是否有足够的 track kinematic coverage，可以覆盖 WB80/81 real calibration population 的 (0,1)/(0,2) `pred_tx,pred_ty` applicability domain，从而未来训练/验证一个不需要对 real tracks 外推的 conditional-J model？

WB80 已证明 canonical 3-source construction subset 的 transfer support 失败（(0,1)=0.806、(0,2)=0.872，gate ≥0.90），并冻结 `jacobian_transfer_model_validated=false`。那是**特定 3-source subset** 的结论。本战役问更广的问题：**全部现有非 sealed MC/control 中，是否存在任何一个 candidate 覆盖真实 calibration support**。

## Real support target（`real_support_target.json`，residual-blind）

继承 WB80 frozen calibration population（run 14973、14974），只读 residual-independent track/pair metadata（不读 residual 值）。Primary kinematic 为 source-tracklet frame `source_tx/source_ty`（对所有 candidate 与 real target 统一）；`pred_tx/pred_ty` 作 cross-check。

| pair | n_pairs | source_tx_p95 | source_ty_p95 | pred_tx_p95 | pred_ty_p95 | gated |
| --- | --- | --- | --- | --- | --- | --- |
| (0,1) | 227 | 0.0635 | 0.0237 | 0.0653 | 0.0262 | 是 |
| (0,2) | 149 | 0.0417 | 0.0128 | 0.0448 | 0.0133 | 是 |
| (0,3) | 2 | 0.0293 | 0.0130 | 0.0247 | 0.0154 | report-only |

真实 (0,1) 群体在 ty 上明显比 canonical MC 宽（real source_ty_p95≈0.024 vs canonical≈0.0097），且呈 +ty 不对称（约 87% track ty>0）。这正是 WB80 发现的 "wide-ty" gap。

## Candidate inventory（仅 kinematic metadata）

系统性 inventory 5 个现有非 sealed candidate，admission 只用物理 metadata/kinematics。Sealed source（`mc24_100116_00030_00039`、`mc24_100117_00030_00039`）与历史 100012 test split（`mc24_00020_00024`）被结构性排除。旧 campaign 失败（100120 physically-distinct admission、100130 movable-station min_pairs）**不**自动排除 residual-blind coverage control，coverage 好也**不**重写这些 identifiability 结论。

## 覆盖度结果（`coverage.json`）

**Primary gate：每个 gated pair type 要求 Mahalanobis 99% envelope fraction ≥ 0.90 且 density bin-occupancy(h=0.01) ≥ 0.90 且统计充足（≥30 pair、≥2 source）。** 两个 coverage metric 互为冗余：Mahalanobis 是 WB80-consistent envelope，bin-occupancy 是 robust 密度测度（防 "wide-but-sparse" / outlier-inflation）。已预注册 physical acceptance filter（|tx|,|ty|≤0.2）去除非物理 vertical-track outlier，使 Mahalanobis envelope 不被虚增。

| candidate | (0,1) maha | (0,1) dens | (0,2) maha | (0,2) dens | n(0,1) | species(p_med) | coverage | particle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_hierarchical_v1 | 0.841 | 0.881 | 0.940 | 0.960 | 1431 | muon(486 GeV) | **败 (0,1)** | 兼容 |
| gaussian_theta_100012 | 0.749 | 0.529 | 0.893 | 0.651 | 17 | muon(100 GeV) | **败（统计）** | 兼容 |
| fluka2d_100116_100117 | 0.877 | 0.767 | 0.980 | 0.906 | 65 | muon(54 GeV) | **败 (0,1)** | 兼容 |
| **floor_muon_100120** | **0.943** | **0.965** | **0.993** | **1.000** | **36123** | **muon(265 GeV)** | **过** | **兼容** |
| kshort_100130 | 0.885 | 0.700 | 0.980 | 0.832 | 43 | pion 211(113 GeV) | **败（density）** | **不兼容** |

**frame cross-check：** 对所有 muon candidate，tracklet-frame 与 pred-frame 覆盖度一致（差 ≤0.03，如 canonical (0,1) 两 frame 均 0.841），验证 muon 的 source-slope frame 是 pred frame 的良好代理，从而 100120 的 tracklet-frame 覆盖度（无 propagations 可用）可信。pion（100130）两 frame 差异大（0.93 vs 0.78），正说明低动量 pion 的 frame 不可混用，但其 species 已不兼容。

**lever arm / origin 审计：** 所有 candidate 与 real target 的 source-z（−1860.15 mm，station-0）与 lever arm（(0,1) 1907.6 mm）完全一致——floor-origin 只改变 (tx,ty) 分布，不改变 station-pair 几何 lever arm，故不影响 conditional-J 的几何依赖。

## 决策（`wide_ty_mc_support_decision.json`）

`floor_muon_100120` 在 (0,1)、(0,2) 两个 gated pair type 上同时通过 Mahalanobis + density + 统计 gate，且 particle-domain 兼容（muon，265 GeV，lever arm 匹配）。其余 candidate 均在 (0,1) 失败；`kshort_100130` 另因 species 不兼容（pion）被标记。

**决策：`existing_mc_real_wide_ty_support_validated`**
- `validated_candidates = ["floor_muon_100120"]`
- `real_kinematic_jacobian_support_validated = true`
- `conditional_j_retrain_permitted = true`（仅指**未来另开** J-retrain campaign 的 kinematic-support 前提成立）
- `new_mc_generation_required = false`
- `measurement_model_validated = false`（**不变**；还需 WB83 covariance gate）
- `real_data_alignment_v2_preregistration_allowed = false`（**不变**）

## 结论与边界

1. **存在一个现有非 sealed MC（floor-origin muon 100120）覆盖真实 wide-ty calibration support**，(0,1)/(0,2) 双过 0.90 门，统计充足（36k pair、10 source，足够未来 construction/validation split），species/momentum/lever-arm 兼容。**不需要新生成 MC**。
2. **canonical hierarchical V1（18 source）在 (0,1) 仍失败**（maha 0.841、density 0.881 双败），与 WB80 的 3-source 结论方向一致；canonical 的 ty 支持确实太窄。
3. **kshort_100130 是 pion**，即使角度覆盖部分真实支持，也标记 `particle_domain_mismatch`，不能用于 muon conditional-J。
4. **本战役不重写任何旧结论**：100120 的 physically-distinct admission 失败、100130 的 movable-station min_pairs 失败、WB80 的 measurement-model 失败、WB81 的 covariance 未校准，全部继续冻结。WB82 只回答 kinematic coverage 一个问题。
5. **汇合条件**：未来只有 `propagated_covariance_model_validated == true`（WB83）**且** `real_kinematic_jacobian_support_validated == true`（本战役已建立 candidate）才允许新开 Measurement Model Reconstruction & Validation V2。当前 covariance gate 仍未满足，**不能直接跳回 real-data alignment**。

## 后续

- **Workbook 83**（`Propagated-Covariance Upstream Repair & MC Validation V1`）：先做 source-tracklet covariance truth closure（Stage A），再做 propagation covariance construction（Stage B），q/p covariance semantic repair 是独立合同。WB82 不修改任何 FaserActs covariance。
- 未来 conditional-J retrain campaign（另开预注册）可使用 `floor_muon_100120` 作为 kinematic-support-matched 训练/验证样本；其 propagations（pred frame）需在该 campaign 中生成并重新验证 frame 一致性。
