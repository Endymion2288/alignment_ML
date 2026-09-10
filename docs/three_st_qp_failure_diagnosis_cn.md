# 三站 \(q/p\) 失败机制诊断（Yasu-S2D）

Workbook 120。WB119 已冻结：官方 `CKFTrackCollectionWithoutIFT` 的
\(q/p\) 不是可信 reconstructed momentum observable。本阶段只问
**为什么**——不把 S2 翻成 PASS，不进入第 3 阶段 residual
conditionals、alignment 或 weak-mode fit。

Truth 仍只作校准参考。WB119 的 source-disjoint 分母、匹配、binning
和失败样本原样复用。

## 正式 run

Smoke：`yasu_s2d_three_st_qp_failure_diagnosis_smoke_20260908T182759Z_d4782b4a`

```
decision = three_st_qp_failure_diagnosis_contract_established
diagnosis_verdict = INCONCLUSIVE
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
```

Batch：`yasu_s2d_three_st_qp_failure_diagnosis_batch_20260908T190106Z_b5bf3e74`
（与 WB119 相同的 3 398 774 条主样本、243 758 条 sign-flip）

```
decision = three_st_qp_failure_diagnosis_recorded
mechanism_verdict = mixed/inconclusive
supported = curvature_information_loss, covariance_semantics_failure
excluded  = reference_definition_error, charge_asymmetric_reconstruction
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
new_reconstruction_dump_authorized = false
```

## 机制裁决

高 \(p\) 符号翻转符合曲率信息不足：\(p\ge 2\,\mathrm{TeV}\) 翻转率
18.2%，\(p<200\,\mathrm{GeV}\) 为 4.2%；
\(\lvert q/p_{\rm truth}\rvert/\sigma<0.5\) 时 40.1%，
\(\ge 10\) 时 1.66%。高 \(p\) 平均 significance 1.67。

这 **不能**解释 pull RMS 955 或 68%/95% coverage 0.415/0.659。原生
5×5 的 \(\langle\log\kappa\rangle=30.4\)，2038 条最小本征值
\(\le 0\)。均值错与 \(\sigma\) 错同时成立。

参考恒等式通过（S1 重算、\(q\cdot p\)、`front()` 不在 IFT）。改用
production \(q/p\) 或 GeV 单位救不了 coverage。电荷奇 \(\Delta\) 大于
偶项，但 \(\mu^+/\mu^-\) 翻转率只差 0.86 个百分点，因此排除
charge-asymmetric reconstruction，也 **不是**磁弱模。

短径迹 / 缺站集中翻转（`n_mot<6` 或缺两站约 45%）。只报告条件分布，
不是 cut。WB119 dump 没有 seed \(q/p\) 或 leave-one-station 增量；因为
① 与 ② 已由现有字段决定，**不授权**新的 reconstruction dump。

第 3 阶段保持关闭。
