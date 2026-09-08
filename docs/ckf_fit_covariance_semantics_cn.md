# CKF 拟合协方差语义（Stage B / Task B11）

Workbook 107。解释 WB105 中 Cin 在传播前已经过宽
（`λ = 0.027 / 0.054 / 0.153 / 11.35`，pencil `6.80`）。
证据来自冻结 Calypso / ACTS 代码路径，不凭类名猜测。
不缩放、不裁剪、不重新对角化、不经验标定 Cin。

## 正式结果

```
verdict = DIAGNOSED
decision = ckf_fit_covariance_semantics_audited
primary_case = official_cin_is_global_kf_refit_front_state
secondary_cases = official_refit_marks_ift_as_outlier,
                  seed_covariance_inflated_before_refit,
                  target_stations_used_in_refit
suitable_independent_propagation_seed = false
measurement_model_v2_authorized = false
cin_modified = false
```

正式 run `sbb11_ckf_covariance_semantics_20260906T211908Z_596ac546`。

## 生产链

1. **CKF2** 组合卡尔曼滤波，MS 与能量损失打开，
   连接 `Acts::GainMatrixSmoother`。可选地把拟合参数放在
   seed-tool 目标平面（`addFittedParamsToTrack=true`）。
2. **CreateTrkTrackTool**：measurement TSOS 落 **smoothed**，
   outlier TSOS 落 **filtered**，hole TSOS 落 **predicted**。
   若提供 fitted params，则作为 Hole 插到最前面。
3. **KalmanFitterTool.fit** 在成功时把 `Trk::Track` 重拟合进
   `CKFTrackCollection`。它使用全部 `measurementsOnTrack`，
   把 IFT（`z < -100 mm`）标成 outlier，膨胀种子协方差
   （对角 ×100，再 ×10），并且调用 `createTrack` 时
   **不**带 fitted params。hit 协方差写死为 `0.08²/12`。
4. **CkfActsTransportDumpAlg** 导出
   `track->trackParameters()->front()` 作为 Cin。

官方 refit 成功后，`front()` 是最上游的 measurement 面。
对 IFT 而言该面是 **outlier**，因此落盘参数是
**filtered / predicted**，不是测量更新，也不是
leave-target-out 状态。1–3 站仍作为 smoothed measurement
留在 refit 里。

因此 `trackParameters().front()` **不适合**作为独立传播种子。

## Cin 形状（只诊断，不改）

663 条唯一契约 track，truth 只作源面诊断：

| λ(C_emp, Cin) | pencil |
| --- | --- |
| 0.027 / 0.054 / 0.153 / 11.35 | 6.80，主轴 `x` |

过覆盖在**传播前**已经存在。本任务不修 Cin。

当前 jsonl / ntuple 无法恢复逐 TSOS 的滤波/平滑旗标。
以后允许只做 provenance dump；不允许标定 Cin。
