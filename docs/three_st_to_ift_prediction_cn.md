# 独立 3ST→IFT 预测链（Yasu 第 1 阶段）

Workbook 118。第 0 阶段把 `CKFTrackCollectionWithoutIFT` 锁定为官方
S1+S2+S3 CKF 之后，本任务从该 collection 的原生 5 维状态和 5×5
物化一条到 IFT 的独立预测。

**不**重新拟合，不调用 `KalmanFitterTool.fit`，不用 WB109 LTO（LTO
保留 IFT），不从 SegmentFit dummy `q/p` 出发。IFT measurement 来自
`SCT_ClusterContainer`，不得进入预测。

## 链路

```
S1+S2+S3 hits
  → 官方 CKFTrackCollectionWithoutIFT（已落盘）
  → 原生 (loc1, loc2, phi, theta, q/p) + 完整 5×5
  → 复制 WB98 helper 的 ACTS 转换
  → FaserActsExtrapolationTool，带磁场，反向
  → IFT 站平面 z = −1860.15 mm（WB87/WB98 值）
  → 独立 FaserSCT_Cluster loc0 residual（IFT wafer）
```

WB98 只从四站 `front()`（IFT）**向前**传到 S1/S2/S3，并且跳过
`targetZ <= sourceZ`。本链必须反向。`AllowBackwardToIft` 为必选项。

正式预测是 Model 1（打开 MS/Eloss）。Model 0 仅诊断。残差为

```
r = loc0_cluster − loc0_predicted
```

`loc0_cluster` 是 `FaserSCT_Cluster.localPosition` 的 `Trk::locX`，与
`ClusterLocalDumpAlg` 相同。`insideBounds` 只记录，**不**作为残差是否
可用的条件。BoundaryCheck / approach 扫描属于第 6 阶段，不是本阶段。

## 失败类（全部保留，不删除）

- `no_without_ift_candidate` —— 选择损失；不是 warning 计数
- `ift_measurement_leak` —— WithoutIFT 的 MOT 或 TSOS 上出现 IFT
- `acts_start_unavailable`
- `acts_s1_to_ift_propagation_failed` —— 跨磁铁平面 hop
- `independent_ift_measurement_absent` / `no_ift_cluster_in_event`
- `ift_prediction_available_residual_unavailable`
- `surface_not_in_identifier_map` / `propagate_surface_failed`

ACTS warning 计数**不等于**重建损失。

## 允许的判定

- `three_st_to_ift_prediction_established`
- `three_st_to_ift_prediction_not_established`

PASS 只授权第 2 阶段（MC q/p 校准）。仅当 construction 与 validation
都至少有一条独立 IFT residual 时，才打开 residual 条件期望。这**不**
声称 q/p 无偏，**不**声称存在 `q/p ↔ R_y/d_x` 弱模，也不重开 B14M /
B15 / Measurement Model V2 / alignment。

## 正式 run

`yasu_s1_three_st_to_ift_prediction_20260908T131812Z_b30ae4b2`

```
decision = three_st_to_ift_prediction_established
mc_qp_calibration_authorized = true
residual_conditional_authorized = true
```

construction / validation 各 20 事件。反向 IFT 平面 Model 1 在
WithoutIFT 径迹上 20/20 与 19/19 成功。IFT 泄漏为 0。validation 有 1
个事件没有 WithoutIFT candidate（选择损失，第 0 阶段已见）。落到具体
wafer 时出现的 ACTS `PropagatorError:3`（`StepCountLimitReached`）或
`SurfaceError:1`（`GlobalPositionNotOnSurface`）是 residual 级失败，
不是删径迹。这些 ERROR 计数不是重建损失。

全部 IFT cluster 的 residual 混有未关联 cluster。带
`reconstruction_associated` 标记的（唯一四站 TSOS IFT identifier）只
是诊断族；第 3 阶段必须预注册哪一族是正式 observable。它们的均值
**不是** Yasu 偏置结论。
