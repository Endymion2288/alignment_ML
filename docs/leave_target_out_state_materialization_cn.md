# 独立 Leave-Target-Out 状态物化（Stage B / Task B13）

Workbook 109。WB108 封锁了 V4 的 C/D 两臂，因为独立
leave-target-out 状态尚不存在。本任务在冻结的 WB103
契约样本上物化该状态。

helper 只存在于 `alignment_ML`。**不得**调用
`KalmanFitterTool.fit`。它读取同一冻结 xAOD 的
`CKFTrackCollection`，显式移除当前目标站的全部 measurement，
用剩余 hits 做独立 ACTS Kalman 拟合（保留 IFT，不用 outlier
finder，MS/Eloss 打开），并导出 source/reference surface 上的
状态和完整 5×5。

验收标准是 LTO 契约，不是 covariance closure。
`100043/37` 必须保留。禁止 truth q/p、dummy q/p、经验
`Cov(pred,target)`、Cin rescale、Q tuning，以及 Measurement
Model V2。

## 必须导出的字段

```
event/source identity
source station
target station
used station IDs
used measurement IDs/counts
excluded target measurement IDs/counts
fit success / failure reason
reference/source surface
native track parameters
5×5 covariance
signed q/p
state/covariance units and frame
material / field / geometry / conditions hashes
```

每一行都必须能机器验证
`target_station_measurements_used = 0`。失败行全部保留。
不得根据拟合结果新增 selection。

## 六点契约

1. 目标站 exclusion 可证明。
2. 每个成功拟合都有 source/reference surface 上的 state + 5×5。
3. 协方差不是 WB107 的 `front()` Cin，不是 seed covariance，
   也不是经验构造。
4. q/p 是拟合结果，不是 dummy，也不是 truth。
5. construction / validation 的成功、失败和失败原因覆盖全部
   WB103 denominator。
6. 不使用 closure 信息选择状态。

正式 PASS token：

```
decision = leave_target_out_state_materialization_established
independence_proven = true
lto_states_materialized = true
b14_authorized = true
```

这**不会**把 `transport_covariance_validated` 设为真，也不授权
Measurement Model V2。只有 B13 PASS 才允许进入 Task B14。B15
仍然关闭。
