# 三站 \(q/p\) 协方差 / 曲率信息根因审计（Yasu-S2E）

Workbook 121。WB119 已冻结：官方 `CKFTrackCollectionWithoutIFT` 的
\(q/p\) 不是可信 reconstructed momentum observable。WB120 支持两个
彼此独立的问题，停在 `mixed/inconclusive`。本阶段要把它们拆开：

1. \(q/p\) 本身不可辨识到什么程度（曲率信息）？
2. 当前报告的 \(\sigma(q/p)\) / 原生 5×5 为什么不能表示这种不可辨识性？

不把 S2 翻成 PASS，不进入第 3 阶段 residual conditionals、alignment、
\(R_y/d_x\) weak-mode、ACTS boundary scan、B14M/B15 或 Measurement
Model V2。不提交新的 reconstruction dump。Truth 仍只作校准参考。

## 正式 run

Smoke：`yasu_s2e_three_st_qp_root_cause_smoke_20260908T203419Z_42264e70`

```
decision = three_st_qp_root_cause_contract_established
diagnosis_verdict = INCONCLUSIVE
A/B/D = unresolved
C = rejected
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
new_reconstruction_dump_authorized = false
```

Batch：`yasu_s2e_three_st_qp_root_cause_batch_20260908T210534Z_f1fa87c2`
（与 WB119 相同的 3 398 774 条主样本、243 758 条 sign-flip）

```
decision = three_st_qp_root_cause_recorded
A = supported   intrinsic_curvature_information_limit
B = unresolved  topology_amplified_information_loss
C = rejected    covariance_basis_unit_provenance_bug
D = supported   covariance_model_calibration_failure
information_limit_remains_if_covariance_fixed = true
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
new_reconstruction_dump_authorized = false
```

## 结构化 A/B/C/D 检验（看 batch 前冻结）

Clean 子样本（只作条件诊断，不是新的物理样本）：完整 S1+S2+S3、
`n_mot==18`、`ndof>0`、无 outlier、\(\chi^2/\mathrm{ndof}<5\)、SPD、
最小本征值 \(>0\)。

| 代号 | 机制 | 支持 | 拒绝 |
| --- | --- | --- | --- |
| A | 本征曲率信息极限 | clean 内：高−低 \(p\) 翻转差 \(\ge 0.05\)，翻转率随 significance 单调下降，低−高 sig 翻转差 \(\ge 0.10\) | 适用且高−低 \(p\) 差 \(< 0\) |
| B | 拓扑放大的信息损失 | dirty−clean 翻转差 \(\ge 0.10\)（各 \(n\ge 50\)） | 适用且 gap \(< 0.05\) |
| C | 协方差基 / 单位 / provenance bug | 按 \(10^{\pm 3}\) 或 \(\sqrt{10^3}\) 重标后 RMS \(\in[0.60,1.40]\) **且** 68% coverage \(\in[0.50,0.80]\)（像高斯，不是胖尾过覆盖） | 否则拒绝“全局单位 / Jacobian 是 pull 失败的原因” |
| D | 无转换 bug 的协方差模型 / 校准失败 | C 未救援；clean 仍失配；\(\lvert z\rvert\ge 10\) 占 clean \(\sum z^2\) \(\ge 50\%\) | clean 已校准 |

非 SPD 与极端 \(\lvert z\rvert\) tail 留在样本里，只报告它们对 pull
RMS、coverage 失败和 sign-flip 的贡献，不得删除后重算“clean PASS”。

测量级曲率 significance 用 \(\lvert q/p_{\rm fit}\rvert/\sigma(q/p)\)。
\(\lvert q/p_{\rm truth}\rvert/\sigma\) 只作 MC diagnostic。禁止仅凭
condition number 大就认定 covariance 错。

## 转换链（源码锁定）

Calypso pin `40892527e9c65409afd2378a2abfc25ddbddac03`。

`CreateTrkTrackTool::ConvertActsTrackParameterToATLAS` 取 Acts bound
6×6，丢掉时间，对 \(q/p\) 行列乘 `1_MeV=1e-3`（故 \(C_{44}\times 10^{-6}\)，
一致 Jacobian），再把 5×5 **不经** bound→curvilinear Jacobian 塞进
`Trk::CurvilinearParameters`。源码注释写 “to GeV”，算术却是 Acts
\(1/\mathrm{GeV}\) → Trk \(1/\mathrm{MeV}\)。\(q/p\) 基不变，所以
\(C_{44}\) 不应因这个标签改变。

`CKF2` 可把 target-plane 拟合参数作为 Hole 放到 `front()`，再
`KalmanFitterTool::fit` refit。成功则重写 track 且不带 fittedParams
（`front()` = 第一条 MOT smoothed state）；失败则保留 CKF track。
dump 没有 `kf_refit_succeeded`。refit 把输入协方差 \(\times 10\)，
并用表面中心而不是径迹 loc 当 loc seed。

缺全局 \(10^3/10^6\) 会把**所有** pull 同标度缩放。WB119 中位数
\(\bar z\approx -0.07\)、41.5% 已在 1σ 内，反对这种全局错误。
bound 当 curvilinear 解释不了 pull RMS 955。

## Diagnostic-export contract

`authorized = false`。现有字段加上锁定的转换链已经能裁决 A / B /
C-单位 / D。Acts native 6×6 只用于给 CKF-vs-refit 贴标签，不改变这些
裁决。

以后只有书面 contract 证明“现有字段不足 **且** 新字段能区分两个具体
机制”才允许另开 dump。不得改 seed、fitter、hit selection、covariance
scale 或 geometry，不得把 truth 放进 fit。

## 批量结论

在 clean 的 18-hit / 完整三站 / SPD 子样本（\(n=1\,973\,802\)）里：
高 \(p\) 翻转 14.2%，低 \(p\) 仅 0.59%（差 +13.6 pt），并且仍随
曲率 significance 单调下降（\(\lvert q/p_{\rm truth}\rvert/\sigma<0.5\)
时 34.1%，\(\ge 10\) 时 0.065%）。这是 **本征信息极限**，不是拓扑
缺陷。全部 sign-flip 的 52.5% 已经落在测量级
\(\lvert q/p_{\rm fit}\rvert/\sigma<1\) 区。

全局 \(10^3/10^6\) 单位 / Jacobian bug **被拒绝**：把 pull 乘
\(10^{-3}\) 得到 RMS 0.955，但 68% coverage 0.999（过覆盖，不是高斯
救援）。转换链的 \(C_{44}\times 10^{-6}\) Jacobian 内部一致，且
\(q/p\) 基不变。

报告的 5×5 在**没有**这种转换 bug 的情况下仍然失准：clean RMS 21.9，
coverage 0.440 / 0.700，\(\lvert z\rvert\ge 10\) 占 clean \(\sum z^2\)
的 98.9%。2038 条非正本征值径迹只贡献 0.18% 的 \(\sum z^2\)，未删除。
7947 条 \(\lvert z\rvert\ge 100\) 拥有合并平方和的 99.995%。

拓扑会放大翻转（dirty 11.4% vs clean 4.1%，差 +7.2 pt），但不到
+10 pt 支持线，故 B 保持 unresolved。它解释不了 clean 内的高 \(p\)
翻转。

即使以后修正 covariance，高 \(p\) 物理信息极限仍然存在。

## 第 3 阶段状态

```
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
```

只有未来某个**独立**阶段同时证明 reconstructed \(q/p\) 的符号/均值和
uncertainty 在预注册范围内可用，才允许重新讨论 S3。
