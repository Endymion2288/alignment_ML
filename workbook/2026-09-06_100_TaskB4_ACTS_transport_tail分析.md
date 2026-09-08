# Workbook 100: Task B4 ACTS transport tail / uncertainty 分析

日期：2026-09-06
状态：**完成** —— 冻结尾巴已分类。未修 closure，未删尾巴，未 clip，未放大 C，未调 Q，未改 gate，未进入 Measurement Model V2 / alignment / ML。未改 WB87–WB99 结论。

**最终判定：`ANALYZED`（主因 Case 1）**

- `decision = acts_transport_tail_uncertainty_analyzed`
- `primary_case = wrong_track_state_dominated`
- `secondary_cases = material_slope_correlated_tail`
- `next_step = reconstruction_association_quality_control`
- `closure_pass = false`
- `measurement_model_v2_entered = false`

冻结保持：

- WB96：`ckf_qoverp_covariance_export_established`
- WB98：`acts_q_materialized_closure_failed`
- WB99：`acts_transport_covariance_failure_diagnosed` / `high_chi2_tail_dominated`

## 起始状态

HEAD（任务开始时）：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

相对 WB99：只新增本任务文件。未重跑 WB87–WB99，未覆盖 WB98 dumps，未覆盖 WB99 `sbb3_acts_transport_diagnosis_20260906T180100Z_574d6429`。

输入：WB99 冻结 official pair top 1%（24 条）与 top 0.1%（3 条），只读同一 9 个 dump。未重新计算 χ² 分位数。

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| B4.1 尾巴分类 | 冻结名单；原始 residual / C / Q | **完成**。24/24、3/3 保留。`rescreened=false`。 |
| B4.2 重建质量 | construction / validation 比例；不删除 | **完成**。0 删除。 |
| B4.3 pull | 全样本 + 去尾诊断 | **完成**。去尾只作诊断。 |
| B4.4 Q 特征值 | 不调 Q，不用 Highland | **完成**。 |
| 机制 | Case 1 / 2 / 3 | **Case 1**。次因 Case 2。 |
| 禁止项 | 不进 V2 / alignment，不调 C/Q | 保持。 |

## B4.1 High-χ² tail 分类

冻结 1%：24 条。主类：

| 主类 | n |
| --- | --- |
| A 重建异常 | 18 |
| B 输运异常 | 0（作为主类；5 条同时带 Jacobian 旗标，但优先归 A） |
| C 几何/材料 | 5 |
| 未解释 | 1 |

旗标（可重叠，未删）：`qoverp_pull_anomaly` 16，`repeated_event` 10，`high_material_proxy` 14，`wrong_momentum`（\|log10 p 比\|≥1）6，`jacobian_anomaly` 5，`large_slope` 3，`low_truth_match` 3，`poor_ckf_fit` 1。

`Q` 非 PSD 在尾巴里很常见（18/24），但全样本也是 ~59%，不能单独当输运失败。

冻结 0.1% 三条全部是 A：

| source | run/event | pair | log10(p_reco/p_truth) | q/p pull | CKF χ²/ndof | 主类 |
| --- | --- | --- | --- | --- | --- | --- |
| 100043_00400_00499 | 100043/37 | (0,3) | −0.69（约 4.9 倍，未过十年门） | −13.8 | 1.42 | A |
| 100048_00050_00099 | 100048/86 | (0,1) | −1.46 | −6.8 | ntuple 无匹配 Track | A |
| 100044_00200_00299 | 100044/92 | (0,1) | +0.20 | −5.5 | 无匹配 Track | A |

`100043/37` 的 CKF 拟合本身合格（χ²/ndof=1.42，truth match=1），但 q/p 相对 truth 的 pull 极大：这是**错误动量状态**，不是坏的 surface 契约。该 event 同时出现在 `(0,1)/(0,2)/(0,3)`。

## B4.2 Tail 与 reconstruction quality

| split | n official | tail 1% | tail 分数 | 十年错误 p | 重复 event 行 | 跨 pair 身份 |
| --- | --- | --- | --- | --- | --- | --- |
| construction | 1309 | 14 | 1.07% | 3/14 | 5/14 | 2 |
| validation | 1043 | 10 | 0.96% | 3/10 | 5/10 | 2 |
| 全部 | 2352 | 24 | 1.02% | 6/24 | 10/24 | 4 |

跨 pair 身份（全部保留）：`100043/37`、`100043/22`、`100048/15`、`100048/61`。

validation 上，十年错误 p 贡献了该 split 尾巴 χ² 的 **79%**、官方 χ² 的 **67%**。construction 上十年错误 p 只占尾巴 χ² 的 2%；construction 的 A 类主要是 \|q/p pull\|≥5 与重复 event。

**这些尾巴首先是错误 track state，不是错误 covariance。** 未据此删除任何事件。

## B4.3 Pull distribution

官方 2352 条全部保留。去尾 2328 条只作诊断。

| 样本 | pull_x RMS | pull_y RMS | pull_tx RMS | pull_ty RMS | pull_q/p RMS | \|pull_x\|>5 |
| --- | --- | --- | --- | --- | --- | --- |
| 全样本 | 1.47 | 0.93 | 1.14 | 0.75 | 4.82 | 1.2% |
| 去尾诊断 | 1.03 | 0.62 | 1.02 | 0.61 | 3.43 | 1.1% |
| 冻结 1% | 10.5 | 6.83 | 5.18 | 4.37 | 33.7 | 17% |

尾巴是非高斯成分（q/p 的 67% 满足 \|pull\|≥5）。去掉冻结 1% 后，y/ty 仍 overcover（RMS 0.62/0.61）；x/tx 接近 1。全样本 mean χ² 被尾巴拉高，主体并不是“处处 undercover”。

## B4.4 Material / Q diagnostic

官方 pair dump 2399 条：`Q=C1−C0` 的 **58.7%** 非 PSD。最小特征值中位 −2.1。与 \|q/p\| 相关 ≈ 0，与 slope 相关 0.05。三个 station pair 的非 PSD 比例都在 0.58–0.60。这是两个独立 propagate 线性化差值的普遍性质，**不是尾巴专属，也未调 Q，未用 Highland 替换**。

slope 富集：尾巴相对全样本 **18.4** 倍（3 条 large slope）。高 Q 富集只有 1.42，未过 Case 2 主门。因此 Case 2 只作次因。

## 机制判定

| Case | 结论 | 依据 |
| --- | --- | --- |
| 1 错误 track state | **主因** | 冻结 1% 的 75% 为 A 类（主要是 \|q/p pull\|≥5，加上十年错误 p 与跨 pair 重复）。0.1% 三条全是 A。validation 十年错误 p 主导该 split 的尾巴 χ²。 |
| 2 材料 / 入射 | **次因** | large-slope 富集 18×；高 Q 不构成主因。Q 非 PSD 是全局现象。 |
| 3 真实 uncertainty 缺失 | **不是主因** | 去尾后 x/tx pull RMS≈1，y/ty 仍偏大。当前 mean χ² 失败由错误状态尾巴主导，不是主体 undercover。 |

下一步（允许路线，尚未执行）：

```
WB99
  → Task B4 (本 workbook)：Case 1
  → reconstruction / association quality control
  → 如仍需要：material diagnosis；再评估 MM V2 输入模型
  → Transport covariance V3
  → Measurement Model V2
  → Alignment estimator
```

禁止把“让 χ² 过 gate”当成下一步。禁止删尾巴、调 C、调 Q、进入 alignment。

## 工程记录

起始 HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

Diff（相对 WB99，未提交）：

- `configs/acts_transport_tail_analysis_v1.yaml`
- `datasets/acts_transport_tail_analysis.py`
- `scripts/audit_acts_transport_tail_analysis.py`
- `tests/test_acts_transport_tail_analysis.py`
- `docs/acts_transport_tail_analysis.md`、`docs/acts_transport_tail_analysis_cn.md`
- 本 workbook

Config SHA：`6b14c571c316ff3aefa55ecbbd16efe0d900e6dedd710025ed3879a99b48b0d8`

Run ID：`sbb4_acts_transport_tail_analysis_20260906T182102Z_a19d9ada`

继承 WB99 decision SHA：`3595c6ef3573947f6ed5ceb5cc835a6fa406cb160c0909f121e6f1cad8ec5fa2`  
继承 WB99 tail SHA：`c6ad96f9b019f0ef1f32ec6d5476ebbb99234aab8e9207a2a314ad89e94627ce`

产物（EOS，不入 git）：`outputs/acts_transport_tail_analysis_v1/sbb4_acts_transport_tail_analysis_20260906T182102Z_a19d9ada/`

| 产物 | SHA256 |
| --- | --- |
| `tail_failure_classification.json` | `4b87eeb9125942f6d6f1b535bc464b38756c50d0176ee3a775ea9aa5b71adf14` |
| `reconstruction_quality.json` | `237b61ff572948686afb74a859b8c5c1b49ebc1df34907430e0a7f37abbbc2fc` |
| `pull_distribution.json` | `cd5c22914353a307c004ae7feb026da0435e5c313a3d5c1fa2d5d0cdc64a54d0` |
| `q_eigenvalue_diagnostic.json` | `ae95e92f3ab93e26258bbe23b7699deb63ea37f06069dae1779f13c39db9d849` |
| `acts_transport_tail_analysis_contract.json` | `03c277c621feee891cc0608c05a78b2942180ec6918831624304444f18d8f3be` |
| `inherited_stage.json` | `42d4072b6a1a8f5bbf5f07a385cb2fb86d83fb35b31f8ce8175cc4be275fe6a5` |
| `COMPLETE.json` | `1af3e2613cf884a65a44d220e33cc7ef1150e00c6537ac55a878fb1c8261d69b` |

测试：`tests/test_acts_transport_tail_analysis.py`，9 passed。短审计，login 跑；未覆盖 Condor dump。

## 下一步

主因 Case 1：先做 **reconstruction / association quality control**，解释错误 q/p 状态与跨 pair 重复 event。不要调 covariance。

材料诊断仍可作为次级任务（slope 富集、全局非 PSD 的 Q）。Measurement Model V2 仍未进入。
