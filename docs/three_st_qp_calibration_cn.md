# 三站 CKF \(q/p\) 校准（Yasu 第 2 阶段）

Workbook 119。第 0 阶段锁定 `CKFTrackCollectionWithoutIFT`、第 1 阶段
物化独立 3ST→IFT 预测之后，本阶段只回答：三站 CKF 带符号 \(q/p\)
本身是否无偏、符号是否可靠、uncertainty / covariance 是否校准。

**不**进入 IFT residual weak-mode、alignment、ACTS boundary 扫描、
B14M、B15 或 Measurement Model V2。Truth **只做校准参考**，不得进入
fit、seed、prior、prediction 或 geometry solve。

## 正式观测量与参考

```
q/p_fit     = WithoutIFT front() 原生 Trk::CurvilinearParameters q/p
              （S1，单位 1/MeV，带符号）
σ(q/p)      = 落盘原生 5×5 的 sqrt(C_44)
q/p_truth   = truthParticle.charge() / |p_S1|
              p_S1 = FiducialParticleTool::getTruthMomenta(barcode)[1]
```

匹配工具是 `TrackTruthMatchingTool`（`SCT_SDO_Map` 在 MOT 上的多数
barcode）。公式沿用 NtupleDumper tracklet 路径，参考站固定为 1，因为
WithoutIFT 的 `front()` 在 S1。产生点 \(p\) 与 0/2/3 站动量只作诊断，
**不是**正式参考。

斜率定义为 \(t_x = p_x/p_z\)、\(t_y = p_y/p_z\)。

## 预注册主量（已冻结）

- \(\Delta(q/p) = q/p_{\rm fit} - q/p_{\rm truth}\)
- pull \(z_{q/p} = \Delta(q/p)/\sigma(q/p)\)
- 符号一致率
- 68% / 95% coverage（\(|z|<1\)、\(|z|<1.96\)）
- pull 均值 / 宽度
- 电荷偶偏置 \(0.5(\langle\Delta\rangle_{\mu^+} + \langle\Delta\rangle_{\mu^-})\)
- 电荷奇偏置 \(0.5(\langle\Delta\rangle_{\mu^+} - \langle\Delta\rangle_{\mu^-})\)

**不要**只看 \(E[q/p]\)。正式条件期望是 \(E[\Delta\mid q/p_{\rm truth}]\)、
\(E[\Delta\mid t_x]\)、\(E[\Delta\mid t_y]\)。偶/奇分解留给后续区分
磁偏转标度与纯几何效应。本阶段不把它解释成 weak mode。

## 冻结的样本规则

保留 sign-flip、高动量尾巴和大 pull。未匹配与 S1 truth 动量为 NaN 的
径迹不进 \(\Delta\)/pull，但留在选择分母。match fraction 只诊断，不
当切割。禁止挑有利动量区间、rescale covariance、用 truth 替换 fit
\(q/p\)、或人为加 prior。

高动量 / \(q/p\to 0\) 明确对应 `qp_near_zero_p_ge_5tev` 与
`p_ge_2000gev`。

## 失败类（全部保留，不删除）

- `mean_bias`
- `charge_sign_failure`
- `kinematic_dependent_bias`
- `covariance_miscalibration`
- `candidate_selection_loss`
- `fit_failure`

FAIL 是负结果。禁止靠重新定义样本翻案。

## 允许的判定

- `three_st_qp_calibration_contract_established` —— 仅 smoke/契约；
  授权 HTCondor 批量，**不**授权第 3 阶段
- `three_st_qp_calibration_established` —— 物理 PASS；三站 \(q/p\)
  可作为后续 reconstructed momentum observable；授权第 3 阶段预注册
  \(E[r_{\rm IFT}\mid q/p]\) 与 \(E[r_{\rm IFT}\mid t_y]\)
- `three_st_qp_calibration_not_established` —— 契约或物理 FAIL

匹配径迹 \(n < 200\) 的 smoke 按冻结门控记为物理 INCONCLUSIVE。只有
batch campaign 才能宣布该观测量可信。

## 正式 run

Smoke：`yasu_s2_three_st_qp_calibration_smoke_20260908T150024Z_63aec9de`

```
decision = three_st_qp_calibration_contract_established
contract_verdict = PASS
physics_verdict = INCONCLUSIVE
batch_htcondor_authorized = true
residual_conditional_authorized = false
three_st_qp_trusted_observable = false
```

construction / validation 各 20 事件。WithoutIFT 中 20/20 与 19/19
条有有限 S1 truth 参考。IFT 泄漏 = 0。validation 有 1 个事件没有
WithoutIFT 候选（选择损失，与第 0–1 阶段相同）。2 条 validation
sign-flip 与高动量尾巴全部保留。\(n=39<200\)，按冻结门控物理
INCONCLUSIVE。Smoke 不能授权第 3 阶段。

Batch：`yasu_s2_three_st_qp_calibration_batch_20260908T175453Z_c6f5561a`
（HTCondor cluster `1112283`，9/9 return 0；3 500 000 事件，
3 398 774 条 truth-matched 主样本）

```
decision = three_st_qp_calibration_not_established
contract_verdict = PASS
physics_verdict = FAIL
mechanism = charge_sign_failure
failure_classes = charge_sign_failure, mean_bias,
                  kinematic_dependent_bias, covariance_miscalibration
residual_conditional_authorized = false
three_st_qp_trusted_observable = false
```

**裁决：** 三站 CKF \(q/p\) **不能**作为后续 Yasu weak-mode 研究中的
可信 reconstructed momentum observable。最严重区域是
\(p\ge 2\,\mathrm{TeV}\)（符号一致率 0.818）以及
\(\lvert t_x\rvert\ge 0.010\)、\(\lvert t_y\rvert\ge 0.010\)。
合并符号一致率 0.928；pull RMS 955；68%/95% coverage 0.415/0.659；
\(\bar z=-0.5002\)。选择与 fit 门控通过。sign-flip 与高动量径迹全部
保留。第 3 阶段保持关闭。

完整 denominator、电荷偶/奇分解以及
\(E[\Delta\mid q/p_{\rm truth}]\)、\(E[\Delta\mid t_x]\)、
\(E[\Delta\mid t_y]\) 见 workbook 119。
