# CKF 尾巴溯源 / association 审计（Stage B / Task B5）

Workbook 101。WB100 已说明 mean χ² 由错误 track state 主导。
本任务只问这些状态从哪里来。

使用 WB99/WB100 冻结的 top 1% / 0.1% 名单。不重新筛选，不删除，
不根据结果设计 cut，不调 C/Q，不进入 Measurement Model V2 / alignment。

truth q/p 只作诊断，不进入重建。

## 审计内容

- **B5.1** 径迹身份：source、run/event、CKF `track_index`、collection、
  站击中、测量数、seed 站、dump 与 ntuple long-track 是否一致。
- **B5.2** 拟合 q/p 对 truth q/p（仅诊断）：残差、pull、动量比、电荷、
  拟合质量。
- **B5.3** 关联：多个 CKF candidate、冲突的 tracklet truth id、未匹配
  tracklet、同一 state 出现在多个 station pair。
- **B5.4** 质量变量（若有）：χ²、ndof、测量数、层数、站图案。
  smoothing / outlier / material interaction 若 ntuple 没有则记为 null。

旗标只用于分类，不是选择 cut。

## 失败分类

| 情况 | 含义 | 下一步 |
| --- | --- | --- |
| A | 错误 association / candidate / identity | association 质量控制 |
| B | CKF 不是 long track、缺站、或拟合失败 | tracking reconstruction 审计 |
| C | 剩余物理 / 材料 / 传播 | ACTS material diagnosis |

正式 run `sbb5_ckf_tail_provenance_20260906T183700Z_f95fb941` 为
**Case B**：冻结 1% 的 14/24 条是 CKF 重建失败，主要是
`CKFTrackCollection` 里有、ntuple long track 里没有的状态。
association 问题是次因（7/24）。未调 C/Q。
