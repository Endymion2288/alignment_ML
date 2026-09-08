# Leave-Target-Out 预测契约（Stage B / Task B10）

Workbook 106。WB105 留下 A+B 混合：官方 Cin 不是 leave-target-out
预测，并且传播前已经过宽。本任务问：排除当前目标站 measurement
之后，能否建立统计独立的预测状态。

不编造 `Cov(pred,target)`，不把 `KalmanFitterTool.fit` 当成
leave-target-out 导出，不删 `100043/37`，不重开 raw CKF，
不进入 Measurement Model V2。

## 正式结果

```
verdict = NOT_ESTABLISHED
decision = leave_target_out_prediction_contract_not_established
primary_case = leave_target_out_prediction_contract_not_established
next_step = independent_leave_target_out_helper_not_kalmanfittertool_fit
measurement_model_v2_authorized = false
independence_proven = false
lto_states_materialized = false
```

正式 run `sbb10_leave_target_out_20260906T211854Z_951d7052`。
输入仍是 WB103 契约样本：1989 行 / 1974 pair。

## Calypso 接口目录

冻结 Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`，
Athena 24.0.41，ACTS 32.0.2。

| 接口 | 存在 | 整站 LTO 源面 5×5 |
| --- | --- | --- |
| `KalmanFitterTool.fit` | 是 | **否** |
| `getUnbiasedResidual(cluster_z)` | 是 | **否** |
| `getUnbiasedResidual(IFT cluster list)` | 是 | **否** |
| 专用 perigee / leave-one-station-out 导出 | **否** | **否** |

`KalmanFitterTool.fit` 重建全部 `measurementsOnTrack`，
把 `FaserActsOutlierFinder.cluster_z` 设为 `-1e6`，因此
`z < -100 mm` 的 IFT hit 被标成 outlier。这是 leave-**source**-out
（0 站），不是 leave-**target**-out（1/2/3 站）。
种子协方差先乘 `SeedCovarianceScale=100`，再乘 10。

`getUnbiasedResidual(cluster_z)` 只特殊处理
`cluster_z < -10000`。真实目标站 z（47.4 / 1237.4 / 2427.4）
不会被排除。IFT cluster 列表重载会**追加**传入 cluster，
然后仍加入全部 track measurement。

## Dump 清单

官方 WB98 dump 没有
`leave_target_out_state` / `leave_target_out_covariance`。
`Cin` 是 `CKFTrackCollection` 的
`Trk::Track.trackParameters().front()`。

## 目标独立性

每个 official pair 的目标站都在同一条 CKF tracklet 集合里
（比例 1.0）。预测与目标测量不独立。未估计
`Cov(pred,target)`。

验收标准是预测状态的独立性，不是 χ² 变好。该证明不存在，
因此契约未建立。

以后若在 `alignment_ML` 增加独立 helper，**不能**原样包装
`KalmanFitterTool.fit`。那个接口不能同时纳入 0 站并排除目标站。
