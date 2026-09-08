# CKF 重建输入契约审计（Stage B / Task B6）

Workbook 102。WB101 已说明冻结尾巴主要由不满足 long-track 重建契约的
CKF 状态主导。本任务只问：当前哪些 CKF 状态被送进 transport
covariance validation / alignment likelihood。

这是对现有 pipeline 的审计。不设计 cut，不实现过滤器，不删尾巴，
不调 C/Q，不进入 Measurement Model V2 / alignment。

## 审计内容

- **B6.1** 全部 WB87 源上 `CKFTrackCollection` 对 ntuple `longTracks`：
  唯一 CKF 径迹、被接受的 long track、描述性拒绝原因。
- **B6.2** 契约定义：`raw_CKF_collection` 与
  `eligible_for_transport_validation`。后者只记录，不落地。
- **B6.3** 使用 WB101 冻结尾巴，不重新筛选。把 official pair 样本与
  long-track 等价子集做诊断对照。
- **B6.4** 冻结 15/24 条非 long-track 尾巴：站图案、segment、缺站、
  径迹长度、q/p 不确定度。

旗标和阈值只用于分类，不是选择 cut。

## 失败分类

| 情况 | 含义 | 下一步 |
| --- | --- | --- |
| A | 原始 CKF 含大量不完整 state，输入契约缺失 | 先定义输入范围，再做 Transport covariance V3 |
| B | long-track 样本仍有错误 q/p 或冻结尾巴 | CKF 拟合质量审计 |
| C | 质量契约完整但尾巴仍在 | ACTS 材料 / 物理诊断 |

正式 run `sbb6_ckf_reconstruction_contract_20260906T191538Z_afbc27fc`
为 **Case A**，次因 **Case B**。未宣称 closure。long-track 子集只作
诊断对照。

## B6.1 输入范围

九个 WB87 源。CKF 唯一身份为
`(source_id, run_id, event_id, track_index)`。

| 量 | 计数 |
| --- | --- |
| 唯一 CKF 径迹 | 895 |
| 对应 ntuple long track | 779 |
| `longTracks = 0` 的 CKF | 116（13.0%） |
| 有 long track 但无 CKF dump 的事例 | 0 |

116 条被拒 CKF 的互斥描述性原因：

| 原因 | N |
| --- | --- |
| `missing_station` | 59 |
| `other` | 46 |
| `no_truth_match` | 9 |
| `fit_failure` | 2 |
| `short_segment` | 0（全部同时缺站，已计入上一行） |
| `low_measurement_count` | 0（互斥） |

独立旗标：`short_segment = 59`（与缺站同一批），
`low_measurement_count = 17`，`no_truth_match = 17`。
`other` 表示四站、击中数足够，但仍没有 ntuple long track。
生产 long-track 选择比这些下限更严。B6 不补写那条 cut。

## B6.2 契约

当前 production / transport validation 的输入是
`raw_CKF_collection`：所有带 5×5 协方差和有限带符号 q/p 的
`CKFTrackCollection` 状态。不要求站数、测量数或拟合状态。

`eligible_for_transport_validation` **在 production 中未定义**。
文档中的重建契约（本任务不实现为过滤器）是：

- 四站 / 四个 segment
- state 与 5×5 协方差可用
- 有限带符号 q/p
- 拟合状态可用
- 与 long-track 等价

ntuple 的 `Track_*` 已经是 long-track 过滤后的子集。alignment 与
transport validation 当前用的是原始 collection。

## B6.3 尾巴复现（仅诊断）

冻结 official pair top 1% = 24 行。未重新筛选，未删除。

| 样本 | N | χ² 均值 | χ² 中位数 | 剩余冻结 1% | q/p pull RMS |
| --- | --- | --- | --- | --- | --- |
| 原始 official pair | 2352 | 33.33 | 0.150 | 24（1.02%） | 4.82 |
| long-track 等价 | 2094 | 24.64 | 0.116 | 9（0.43%） | 1.15 |

均值下降是因为 15/24 条冻结尾巴随不完整 CKF 状态离开。这 **不是**
closure PASS：最大值仍是 \(3.7 \times 10^4\)，`100043/37` 仍在。

剩余冻结 long-track 身份：`100043/50`、`100043/13`、`100043/37`
（三个 pair）、`100044/3`、`100044/2`、`100047/80`、`100047/68`。
剩余 3 条 `|q/p pull| ≥ 5` 全部是 `100043/37`。

## B6.4 短径迹 / 缺站尾巴

冻结 15/24 条非 long-track 行：

| 互斥原因 | N |
| --- | --- |
| `missing_station` | 5 |
| `no_truth_match` | 4 |
| `other` | 6 |

典型缺站：缺 2、3 站，两个 segment，约 10 击中，q/p pull 很大。
六条 `other` 已有四站且 ≥19 击中，仍不在 `longTracks` 里。
q/p 不确定度仍是 dump 的 Cin 量级（\(10^{-7}\)–\(10^{-6}\) / MeV），
解释不了这些 pull。

## 判定

Case A：重建输入契约缺失。13% 的唯一 CKF 径迹、以及 15/24 条冻结
尾巴，是 long-track 选择从未接受的状态。

Case B 是次因，不能靠收窄范围消掉：long-track 对照后冻结尾巴和
`100043/37` 的 q/p 失败仍在。这是 CKF 拟合 / 剩余高动量物理，
不是缩放 C 或 Q 的许可。

不要写成“删掉不完整事例后 closure PASS”。
