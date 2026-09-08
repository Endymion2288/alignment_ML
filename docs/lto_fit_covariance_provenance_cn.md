# LTO 拟合协方差 provenance（Stage B / Task B14R）

Workbook 111。WB110 表明独立 LTO 状态已经物化，但导出的
5×5 未通过 uncertainty contract（覆盖不足）。本任务只问：
这个 5×5 到底是什么。

**不**追求 closure PASS。**不**进入 B15。

## 问题

1. `n_measurements_in_fit` 是 ACTS MeasurementFlag 计数，还是
   输入 hit 数的别名？
2. 导出协方差是否随 seed 尺度改变？
3. 导出的是哪一个 ACTS track state？
4. 去掉某个 target 后，5D 哪些方向真正被约束？
5. WB110 的 pull RMS 是典型过窄，还是少数灾难拟合主导？
6. 为什么 `100043/37` 仍保持官方 q/p 和 seed 尺度的 σ(q/p)？

## 分类

- A：导出协方差是 seed 或 predicted
- B：剩余 measurements 不足以约束 5D
- C：拟合协方差基本不随 seed，但经验误差仍系统性更大
- D：主体合理，少量拟合主导 pull
- E：混合

seed 尺度 `0.1 / 1 / 10` 只用于预注册敏感性。生产尺度保持 1。
不得按 truth / χ² 挑选尺度。

## 字段定义

`n_measurements_in_fit` 是 `fittedTrack.nMeasurements()`，即 ACTS
`calculateTrackQuantities` 对 `TrackStateFlag::MeasurementFlag` 的计数。
它不是输入 hit 数，也不是 `used_measurement_count`。

导出的 5×5 是 `KalmanFitter` 在 source 站平面上的 `fittedParameters`
（`KalmanFitterTargetSurfaceStrategy::first`）：第一个 smoothed
measurement 传输到参考面之后的结果，不是 `KalmanFitterTool.fit`。

`ndof`、filtered/smoothed/outlier 计数、seed covariance 以及
first/last predicted/filtered/smoothed 矩阵只存在于 B14R smoke
schema。冻结的 WB109 dump 保持 B13 schema。

## 判定 token

本任务只能落入：

```
lto_exported_covariance_is_seed_or_predicted_state
lto_fit_information_insufficient
lto_fitter_covariance_semantics_mismatch
tail_dominated
mixed_or_inconclusive
```

这些都不授权 B15、V4 C/D 或 Measurement Model V2。

## 正式 token

正式 B14R run 落在 mixed B+D：

```
decision = mixed_or_inconclusive
lto_cin_contract_established = false
b15_authorized = false
```

导出的 5×5 是 source 平面上的 Kalman fitted state，不是原始
seed / predicted 矩阵。典型拟合均值几乎不随 seed，但 5×5
宽度随 seed。y-pull RMS 由尾巴主导。不得从 smoke 挑选 seed 尺度。
