# CKF q/p 协方差导出契约（Stage B / Task A2）

Workbook 96。本任务从 WB87/95 已经使用的重建 xAOD 中导出持久化
`CKFTrackCollection` 的 5×5。不重跑 SegmentFit，不写 geometry，未通过前不进入
Task B。

原生状态：Athena TrackParameters `(loc1, loc2, phi, theta, q/p)`，q/p 单位
`1/MeV`。导出状态：`(x, y, tx, ty, q/p)`，用已冻结的 CurvilinearUVT Jacobian，
q/p 本身不变换。

允许：Calypso reconstruction chain、`CKFTrackCollection`、已持久化的 ACTS
TrackState。

禁止：truth q/p、SegmentFit dummy 协方差、合成 q/p 注入、新建 source campaign、
写 geometry、进入 alignment、进入 Measurement Model V2。

PASS 要求 construction / validation 都物化完整 5×5，协方差对称且正定，不等于
SegmentFit dummy（`q/p = 1e-5 /MeV`，`var = 5e-6 /MeV²`），并且导出非零 q/p
交叉项。

FAIL 则继续保持 `faseracts_transport_covariance_not_validated`，不打开 Task B。
项目停留在 residual / DQ monitoring 加可复现负结果。
