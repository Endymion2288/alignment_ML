# 条目 71 — RelativeRoute V4 Train→Development 泛化失败机制审计（只读诊断）

日期：2026-09-04
分支：`4station`
状态：**只读机制诊断完成。失败定位为 Case B（泛化失败），主因 = `C_unbounded_correction_scale_collapse` + `D_train_to_development_domain_shift`，次因 = `E_relative_representation_transfer_failure`。未发现 implementation bug。`training_authorized=false`、`final_blind_eval_authorized=false`、`continue_to_15d_relative_wls=false`。本条目不授权任何 V5 训练。**
前置：Workbook 69（配对训练 `1104860`/`1104861`）、Workbook 70（development 评估 `1108310`，scientific gate failure 关闭）
物理闭环合同：`continue_to_15d_relative_wls = false`
密封 Test：`sealed_test_accessed = false`
新 Final Blind：`new_final_blind_content_accessed = false`（`00800_00849` 从未打开）
Development：`development_accessed = true`（`mc24_100047_00350_00399` / `mc24_100048_00350_00399`，仅作机制诊断，不再视为 blind）

---

## 1. 唯一主问题

> 为什么 Workbook 69 中经过 solver-aware C/D objective 训练的 complete-route head，在 Workbook 70 development 上没有降低 `U_truth <= 0`，反而使 Mechanism C 从 Arm 0 的 200 条增加到 Arm 1 的 327 条、Arm 2 的 441 条？

本条目只做只读诊断，不训练、不调参、不 rescue V4、不设计/训练 V5。

---

## 2. 最新 git state（实际本地真值）

```text
branch                 = 4station
remote HEAD (已知)      = cc8d22036be5f0381fa7ad9441477dae155d5a66
本地 audit 前 HEAD      = cc8d22036be5f0381fa7ad9441477dae155d5a66（与远端一致，工作树干净）
本条目新增 commit：
  8043888  Add Workbook 71 audit script + Condor wrapper
  714671f  Add Workbook 71 analysis pass
audit 运行时 HEAD       = 714671fecd0c5f59b93a22df7df9ee0bc2889c4b（git_dirty=false）
```

本地无覆盖、无未提交的分析代码；audit 以 committed 代码版本运行。

---

## 3. Workbook 69 / 70 artifact 独立复核（replay）

不重训。直接从冻结 artifact 复核：

| 项 | 期望 | 复核结果 | 结论 |
| --- | --- | --- | --- |
| Arm 0 checkpoint SHA256 | `0c3a28704cc0…` | `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236` | 一致 |
| Arm 1 checkpoint SHA256 | `e8a6d6c6e265…` | `e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8` | 一致 |
| Arm 2 checkpoint SHA256 | `a52037ead075…` | `a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a` | 一致 |
| Workbook 70 dev C/D（冻结 JSON） | Arm0 C=200/D=170；Arm1 C=327/D=123；Arm2 C=441/D=135 | 本 audit 用同一冻结 solver + 同一 checkpoint 在完整 development 上**精确复放**：Arm0 C=200/D=170、Arm1 C=327/D=123、Arm2 C=441/D=135；selected 3008/2928/2802；complete_truth_chains=3378 | **复放通过** |
| train route candidate 总数 | Workbook 69 报告 ~59.7 万 | 本 audit 重建 = **597,781**（精确一致） | 一致 |

历史 JSON 符号命名勘误（`arm1_minus_arm0_C`）：真实 later-minus-earlier 为 **Arm1−Arm0 ΔC = +127**、**Arm2−Arm1 ΔC = +114**。旧 JSON 不覆盖，仅在此记录。

---

## 4. 审计预注册（preregistration）

- 只读：加载冻结 Arm 0/1/2 checkpoint，对六源 TRAIN overlay 与已打开的 reserved-blind DEVELOPMENT overlay 重跑 inference + 冻结 unit-capacity solver。
- 不训练、不改 loss weight / LR / threshold / Platt / unmatched penalty / route score composition，不 clamp delta，不加 regularizer，不 early stop，不多 seed，不 finetune backbone，不重训 head。
- 不打开 `00800_00849`，不打开 sealed test，不开 15D WLS。
- 输出合同：`outputs/mc24_four_station_relative_route_v4_generalization_audit_v1/`。
- 脚本：`scripts/audit_relative_route_v4_generalization.py`（inference + solver + 梯度审计）、`scripts/analyze_relative_route_v4_generalization.py`（route_rows → 判定）。

---

## 5. 实际训练 objective（从代码确认，非 workbook 摘要）

Workbook 69 total loss = V2 route body + Workbook-64 solver-aware auxiliaries。机器可读表（权重 / reduction / 梯度目标）：

| loss | weight | reduction | 作用于 | 梯度方向（对 delta） |
| --- | --- | --- | --- | --- |
| `route_consistency` | 1.0 | weighted focal BCE，`sum(bce·focal·w)/sum(w)`，pos_weight=`min(neg/pos,80)`，hard_negative×2.0 | truth + fake | truth↑ / fake↓ |
| `one_to_one_competition` | 0.25 | mean over truth routes；每 truth 对 4 端点 logsumexp | truth + 共享端点 fake | truth↑ / competitor↓ |
| `fake_route_penalty` | 0.25 | mean over fake_endpoint routes；softplus | 仅 fake_endpoint | fake↓ |
| `packing_route_competition` | 0.0706 | reduction=`max` over event truth routes | truth + best competitor | truth↑ |
| `dustbin_aware_route_margin` | 0.05 | margin over dustbin-risk truth | truth | truth↑ |
| `gauge_twin` | 1.0 | chart↔twin consistency | paired truth | 净 0 |

`L_corrected = L_edge_W64 + delta_route_logit`；solver `U_complete = clip(L_corrected, ±13.8155) + 4·unmatched_penalty`（unmatched=−1.0）。

---

## 6. route-class imbalance（§9）

| split | complete candidates | truth | fake | truth:fake | 全局 pos_weight |
| --- | --- | --- | --- | --- | --- |
| train | 597,781 | 9,893（1.68%） | 587,888 | 1:59.4 | 59.4 |
| development | 192,588 | 3,378（1.79%） | 189,210 | 1:56.0 | 56.0 |

share-class（按 route 4 端点与同一有效 truth particle 的最大共享数）：

| split | A_truth | B_fake_share3 | C_fake_share2 | D_fake_share1 | E_fully_unrelated |
| --- | --- | --- | --- | --- | --- |
| train | 9,893 | 106,323 | 366,220 | 115,006 | 339 |
| development | 3,378 | 35,422 | 118,985 | 34,768 | 35 |

imbalance 真实存在（1:57–59），但 `route_consistency` 的动态 pos_weight（56–59）+ focal + solver-aware auxiliaries 在 TRAIN 上达到 C=0（见 §8），证明 imbalance 本身不是 development 失败主因。

---

## 7. truth vs fake `delta_route_logit` 分布（§5，核心新审计）

对冻结 epoch-30 checkpoint 只读 inference，按 share-class 分层。核心分位：

**A_truth（完整 truth routes）：**

| split | arm | n | mean | median | q01 | q05 | q95 | min | frac(delta<0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | arm1 | 9,893 | +0.075 | +0.064 | −0.08 | −0.04 | +0.22 | **−0.44** | 0.149 |
| train | arm2 | 9,893 | +0.160 | +0.155 | +0.00 | +0.05 | +0.28 | **−0.38** | 0.009 |
| development | arm1 | 3,378 | −0.991 | +0.042 | **−33.85** | −4.79 | +0.20 | **−53.06** | 0.275 |
| development | arm2 | 3,378 | −1.305 | +0.106 | **−36.15** | −8.30 | +0.27 | **−53.89** | 0.282 |

**fake（pooled）：**

| split | arm | n | mean | median | frac(delta<0) |
| --- | --- | --- | --- | --- | --- |
| train | arm1 | 587,888 | −130.79 | −128.97 | 0.996 |
| train | arm2 | 587,888 | −141.04 | −138.74 | 0.999 |
| development | arm1 | 189,210 | −125.33 | −123.14 | 0.996 |
| development | arm2 | 189,210 | −136.26 | −133.20 | 0.999 |

**解读：** head 在 train 上把 truth（delta≈0，最差仅 −0.44）与 fake（delta≈−130）干净分开；在 development 上，truth 分布中位数仍≈0，但**负尾显著加重**（q01≈−34/−36，最差 −53，frac_neg 0.27–0.28）。即一部分 development truth route 被 head 判成“fake-like”并施加 fake 量级的负 correction。

---

## 8. train vs development C/D（§7，Case 判别）

同一冻结语义下：

| split | arm0 C/D | arm1 C/D | arm2 C/D | arm0/1/2 selected |
| --- | --- | --- | --- | --- |
| **train**（9,893 truth） | **0** / 38 | **0** / 33 | **0** / 25 | 9855 / 9860 / 9868 |
| **development**（3,378 truth） | **200** / 170 | **327** / 123 | **441** / 135 | 3008 / 2928 / 2802 |

**判定 = Case B（generalization failure）。** 在 TRAIN 上，V4 head 完全不增加 Mechanism C（C=0/0/0），反而单调改善 Mechanism D（38→33→25）、单调提升 selected（9855→9860→9868），且 Arm 2 在 train 上最好。solver-aware objective 在分布内**正确且有效**。失败只发生在 development：同一冻结 head 把 C 抬高 +127（Arm1）/ +114（Arm2）。**objective 不是失败原因；失败是 correction 不能跨 source/payload 泛化。**

---

## 9. truth-route transition matrix 与 +127 / +114 精确来源（§6）

development（n=3,378），`decision_class` 迁移：

**Arm 0 → Arm 1（ΔC = +127）：**
```text
arm0 的 200 条 C 全部保持 C（utility_nonpositive->utility_nonpositive = 200，无一被 rescue）
新增 C = 127：
    selected            -> utility_nonpositive : 72   （Arm0 本已正确重建的 truth route 被 Arm1 打入 dustbin）
    packing_competition -> utility_nonpositive : 55   （Arm0 中 D 的 route 被进一步压成 C）
```

**Arm 1 → Arm 2（ΔC = +114 净）：**
```text
新增 C（毛）= 123：
    selected            -> utility_nonpositive : 78
    packing_competition -> utility_nonpositive : 45
离开 C = 9（8 -> packing_competition，1 -> selected），净 +114
```

**头条结论：** 新增 Mechanism C 的**主导来源是 Arm 0 原本已正确重建（selected）的 truth route**——Arm 1 占 72/127（57%），Arm 2 占 78/123 毛（63%）。**V4 correction 在 development 上主动把冻结 backbone 本已评对的 truth route 摧毁（压到 U≤0）。** 且 Arm 0 的 200 条 C 没有一条被 head rescue。

---

## 10. per-loss 梯度方向审计（§8，autograd，不执行 optimizer step）

对 24 个代表性冻结 TRAIN batch（epoch-30 stream，model.train()，固定 seed），用 `torch.autograd.grad` 计算每个 loss 对 `delta_route_logit` 的梯度，按 truth/fake 分组（weighted net force = weight·Σgrad）：

**Arm 1（truth routes）：**

| loss | weight | truth mean grad | truth weighted net | 方向 |
| --- | --- | --- | --- | --- |
| route_consistency | 1.0 | −5.6e-09 | ~0 | 饱和中性 |
| one_to_one_competition | 0.25 | −7.9e-04 | **−0.294** | rescue（推 delta↑） |
| packing_route_competition | 0.071 | −2.3e-03 | **−0.241** | rescue |
| dustbin_aware_route_margin | 0.05 | −2.3e-03 | **−0.171** | rescue |
| fake_route_penalty | 0.25 | 0（nz=0） | 0 | 不触 truth |
| gauge_twin | 1.0 | 0（chart/twin 抵消） | 0 | 净 0 |

Arm 2 同型（one_to_one −0.176 / packing −0.153 / dustbin −0.109）。fake routes 上所有 loss 梯度为极小正值（收敛后饱和，descent 推 delta↓ = suppress），量级 1e-6–3e-5。

**结论：排除 `A_objective_gradient_imbalance`。** 收敛 checkpoint 上，truth route 只受到 solver-aware loss 的**向上 rescue 力**；`route_consistency` 饱和中性；`fake_route_penalty` 对 truth 完全无梯度。**没有任何 loss 把 truth delta 往下推**，更不存在“fake-suppression 梯度规模远大于 truth-rescue”压倒 truth 的现象。objective 的梯度方向是健康的；失败不在 objective。

---

## 11. correction scale vs 物理 utility 尺度（§10）

物理决策尺度：`L_edge` 与 unmatched penalty 为 O(1–10)，solver logit clip ±13.8。

| 对象 | split | arm | |delta| median | |delta|/|L_edge| median | frac(|delta|>5) | frac(|delta|>10) | p_complete<1e-6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| truth | train | arm1/arm2 | 0.06 / 0.16 | 0.011 / 0.026 | 0 | 0 | 0 |
| truth | development | arm1/arm2 | 0.07 / 0.16 | q95=**1.16 / 2.04** | 0.049 / 0.068 | 0.032 / 0.042 | **0.020 / 0.028** |
| fake | train+dev | arm1/arm2 | **123–141** | **26–29×** | — | — | **0.97–0.98** |

**判定：`C_unbounded_correction_scale_collapse` 成立。** fake route 的 |delta| 中位数 123–141，是 edge-logit 总和的 **26–29 倍**，97–98% fake route 被压到 p_complete<1e-6。**V4 已不是一个合理 residual corrector，而是一个叠加在 edge logit 上的独立、极端饱和的二分类器。** 该尺度是“放大器”：任何被判成 fake-like 的 truth route 会被压到远低于 U=0（development 上 2.0–2.8% truth route 饱和到 p<1e-6，对应 ~68/96 条，加上部分负 delta 的 route，正好解释 +127/+114）。

---

## 12. S3 / 2→3 连接（§11，对应 Workbook 65 主问题）

| split | truth routes with L23<0 | delta_arm1 mean | delta_arm2 mean |
| --- | --- | --- | --- |
| train | **0** | — | — |
| development | **50** | **−19.47**（median −12.26） | **−21.49**（median −16.97） |

train 上不存在 L23<0 的 truth route（2→3 边在训练源上全部自信）。development 上 50 条 truth route 的 2→3 边偏弱（L23<0，即 Workbook 65 的原始 Mechanism C 来源）。**对这些本就危险的 route，head 施加的是 −12 到 −21 的大负 correction，而不是正 rescue。** 这直接回答 §11：当原始 `L23` 已偏低时，route head **进一步给予负 correction**，放大了原始失败模式，而非修复。

---

## 13. Arm 2 vs Arm 1 paired representation（§12）

同一 route 的 `delta_arm2 − delta_arm1`（development）：

| 分层 | n | mean | median | q05 |
| --- | --- | --- | --- | --- |
| ALL | 192,588 | −10.75 | −9.38 | −39.5 |
| truth | 3,378 | −0.31 | +0.06 | −3.19 |
| fake | 189,210 | −10.93 | −9.74 | −39.7 |
| truth_L23_negative | 50 | −2.02 | −2.82 | −6.35 |

按 gauge role：None=−9.5、left_se3_control=−11.2、s0_sampling_chart=−10.7（**均匀，非 gauge 特异**）。按 payload family：reference=−9.5、draw_00=−11.7、draw_01=−10.6、hard_s3_ry=−10.6（**均匀，非 S3 特异**）。

**判定：`E_relative_representation_transfer_failure` 为次因。** Arm 2（relative）在 fake 上一致地比 Arm 1（absolute）多压 ~10–11，且 dev truth 负尾更重（q05 −8.30 vs −4.79），导致 Arm 2 一致更差。但该效应是**全局、温和**的，不集中于 gauge twin 或 S3，故 `F_gauge_common_mode_sensitivity` 被排除为主因。

---

## 14. truth label / route key 语义审计（§13）

对 train / development 各 6 个 event 独立重算 route label 与 key：

```text
label_mismatch_count                        = 0（train 439 routes / dev 534 routes）
fake_endpoint_mismatch_count                = 0
complete_truth_chain_missing_from_candidates= 0
complete_truth_chain_wrong_label            = 0
score_map_key_mismatch_events               = 0
ok                                          = True（两个 split）
```

route candidate tuple `(idx0,idx1,idx2,idx3)` 与 truth label、`route_query_score_maps_by_event`、solver `_route_hypotheses`、C/D audit 完全一致。**不存在把真实 complete route 错标成 fake 的 bug。** `implementation_semantic_bug = false`，`G_route_label_or_key_semantic_bug` 排除。

---

## 15. primary failure classification（§15）

主因排序（多选）：

```text
1. C_unbounded_correction_scale_collapse   （enabler：饱和分类器尺度，26–29× edge scale）
2. D_train_to_development_domain_shift      （trigger：truth/fake 边界在 dev 上错位）
3. E_relative_representation_transfer_failure（次因：relative 一致更差，全局温和）
```

排除：

```text
A_objective_gradient_imbalance   —— 梯度审计：truth 只受 rescue 力，C=0 on train
B_route_class_imbalance          —— imbalance 真实但被 pos_weight+focal+aux 补偿，C=0 on train
F_gauge_common_mode_sensitivity  —— arm2-arm1 差在 gauge role 上均匀
G_route_label_or_key_semantic_bug—— label/key audit ok=True，0 mismatch
H_head_only_information_limit    —— 非主因：head 在 train 上明显含辨识信息（C=0、D 改善），失败是迁移而非信息缺失
```

**一句话机制：** V4 head 在 train 上学成一个饱和 truth/fake 分类器（fake delta≈−130，truth delta≈0，C=0）。该边界在 development 上轻微错位，使 2–3% 的 development truth route（尤其 2→3 已偏弱的 route）收到 fake 量级负 delta；由于 correction 尺度无界（饱和分类器），这些 route 被压到 U≪0，把 Arm 0 本已正确重建的 truth route 打成 Mechanism C（Arm1 +127、Arm2 +114，主导来源 = 原 selected route）。

---

## 16. 下一阶段授权逻辑（§16）

本 Workbook 71 永远：

```text
training_authorized            = false
final_blind_eval_authorized    = false
continue_to_15d_relative_wls   = false
```

- 未发现 implementation bug → 不需要 `fix_bug_under_new_preregistered_workbook`。
- 明确涉及 objective/scale（C）→ 允许 `preregister_new_objective_or_route_parameterization`：**预注册一个有界 / residual 的 route correction 参数化**（把 delta 耦合到物理 utility 尺度，使 head 只能做小修正、无法饱和成独立分类器）。
- 明确涉及 representation/domain-transfer（D/E）→ 允许 `preregister_architecture_level_physical_relative_or_gauge_aware_representation`：但任何 gauge-aware feature 必须先从 FASER/Calypso 真实 transform semantics 推导验证，不得直接实现 “gauge-equivariant Transformer”。
- **附加必要条**：任何 V5 在信任 head-only correction 前，必须先在 TRAIN 侧的 held-out source split（**不是 development**）上证明 truth/fake 边界可迁移。
- **本条目不训练任何 V5。**

---

## 17. Condor / 执行记录（§17）

```text
提交脚本   scripts/submit_relative_route_v4_generalization_audit_condor.py
worker     scripts/run_relative_route_v4_generalization_audit_condor.sh
cluster id 1108802  schedd bigbird24.cern.ch  flavour tomorrow  request_gpus=1
状态       removed_before_scheduling
原因       condor_q -analyze：GPU hostgroup bi/condor/gridworker/gpu 仅 1 个匹配 slot 且 drained，
           池内 ~16.5k idle 作业，无法及时获得 GPU slot。
实际执行   只读 audit 改在交互 GPU 节点 lxplus909.cern.ch（Tesla T4, 15360 MiB）后台运行，
           完整记录 provenance：git_commit=714671fecd（dirty=false）、hostname、GPU、时间戳。
           本 audit 为只读（无训练、无调参），科学完整性由确定性脚本 + 记录保证。
           Condor 提交保留为记录；如需 Condor 复放，同一脚本同一 checkpoint 结果确定一致。
```

Condor return 0 ≠ scientific PASS；本条目结论来自 artifact 数字，不来自作业状态。

---

## 18. 输出合同（§14）

`outputs/mc24_four_station_relative_route_v4_generalization_audit_v1/`：

```text
audit_contract.json                    边界/标签审计/git/provenance
route_rows.jsonl                       790,369 行（train 597,781 + dev 192,588），含 L01/L12/L23、L_edge、
                                       delta_arm1/2、U_arm0/1/2、best_fragment、C/D/selected、share-class、
                                       S3 量（chi2_23/L23_calibrated）、gauge role、source
train_summary.json                     train C/D：arm0 0/38、arm1 0/33、arm2 0/25
development_summary.json               dev C/D：arm0 200/170、arm1 327/123、arm2 441/135（复放 W70）
loss_gradient_summary.json             6 loss × truth/fake 梯度方向（24 batch autograd）
route_class_balance.json               truth/fake 比例、share-class、pos_weight
delta_distributions.json               truth/fake × share-class × arm × split 分位
truth_route_transition_matrix.json     arm0->1->2 迁移 + 新增 C 精确来源
train_vs_dev_cd.json                   Case A/B/C 判别
correction_scale_audit.json            |delta|/|L_edge|、饱和比例
s3_connection_audit.json               delta vs L23/p23/chi2_23 分层
arm1_vs_arm2_paired_summary.json       paired delta 差，按 truth/fake/payload/gauge/source/S3 分层
decision.json                          primary classification + 授权逻辑
```

未写任何 final blind 数据。

---

## 19. 边界状态

```text
development_accessed              = true（00350_00399，仅机制诊断）
new_final_blind_content_accessed  = false（00800_00849 未打开）
sealed_test_accessed              = false
training_authorized               = false
final_blind_eval_authorized       = false
continue_to_15d_relative_wls      = false
```
